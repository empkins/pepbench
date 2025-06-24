import re
from collections.abc import Sequence
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import ClassVar

import pandas as pd
from biopsykit.signals.ecg.event_extraction import QPeakExtractionVanLien2013
from biopsykit.signals.ecg.preprocessing import EcgPreprocessingNeurokit
from biopsykit.signals.ecg.segmentation import HeartbeatSegmentationNeurokit
from biopsykit.signals.icg.preprocessing import IcgPreprocessingBandpass
from biopsykit.utils.dtypes import EcgRawDataFrame, IcgRawDataFrame

from pepbench.datasets import BasePepDatasetWithAnnotations
from pepbench.datasets._base_pep_extraction_dataset import base_pep_extraction_docfiller
from pepbench.datasets.time_window_icg._helper import _get_match_heartbeat_label_ids, _load_txt_data
from pepbench.utils._types import path_t

_cached_get_txt_data = lru_cache(maxsize=10)(_load_txt_data)

__all__ = ["TimeWindowIcgDataset"]


@base_pep_extraction_docfiller
class TimeWindowIcgDataset(BasePepDatasetWithAnnotations):
    base_path: Path
    use_cache: bool

    exclude_r_peak_detection_errors: bool

    SAMPLING_RATE: ClassVar[int] = 2000

    PHASES: ClassVar[Sequence[str]] = ["Baseline", "EmotionInduction"]

    SUBSET_R_PEAK_DETECTION_ERRORS: ClassVar[Sequence[str]] = ["IDN_17"]

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Sequence[str] | None = None,
        subset_index: Sequence[str] | None = None,
        *,
        return_clean: bool = True,
        use_cache: bool = True,
        exclude_r_peak_detection_errors: bool = True,
        only_labeled: bool = False,
    ) -> None:
        self.base_path = base_path
        self.use_cache = use_cache
        self.exclude_r_peak_detection_errors = exclude_r_peak_detection_errors
        self.data_to_exclude = self._find_data_to_exclude()
        super().__init__(
            groupby_cols=groupby_cols, subset_index=subset_index, return_clean=return_clean, only_labeled=only_labeled
        )

    def _sanitize_params(self) -> None:
        # ensure pathlib
        self.base_path = Path(self.base_path)

    def create_index(self) -> pd.DataFrame:
        self._sanitize_params()

        file_list = sorted(self.base_path.joinpath("signals").glob("*.txt"))
        p_ids = [re.findall(r"IDN(\d+).txt", f.name)[0] for f in file_list]
        p_ids = sorted([f"IDN_{p_id.zfill(2)}" for p_id in p_ids])
        index = product(p_ids, self.PHASES)
        index = pd.DataFrame(index, columns=["participant", "phase"])
        index = index.set_index("participant")

        index = index.drop(self.data_to_exclude)
        index = index.reset_index()

        return index

    def _find_data_to_exclude(self) -> Sequence[tuple[str, str]]:
        data_to_exclude = []
        if self.exclude_r_peak_detection_errors:
            data_to_exclude.extend(self.SUBSET_R_PEAK_DETECTION_ERRORS)

        return data_to_exclude

    @property
    def sampling_rate_ecg(self) -> int:
        return self.SAMPLING_RATE

    @property
    def sampling_rate_icg(self) -> int:
        return self.SAMPLING_RATE

    @property
    def data(self) -> pd.DataFrame:
        if not self.is_single("participant"):
            raise ValueError("Data can only be loaded for a single participant.")

        p_id = self.subset_index.iloc[0]["participant"]

        file_path = self.base_path.joinpath("signals").joinpath(f"IDN{int(p_id.split('_')[1])}.txt")

        data = _cached_get_txt_data(file_path) if self.use_cache else _load_txt_data(file_path)

        data = data.copy()
        data.index = pd.to_timedelta(data.index / self.SAMPLING_RATE, unit="s")
        data.index.name = "t"

        if self.is_single("phase"):
            phase = self.subset_index.iloc[0]["phase"]
            split_idx = int(120 * self.sampling_rate_icg)
            # Baseline phase is 0-120s, EmotionInduction phase is 120-end
            data = data.iloc[:split_idx] if phase == "Baseline" else data.iloc[split_idx:]

        return data

    @property
    def ecg(self) -> EcgRawDataFrame:
        """Return the ECG data.

        Returns
        -------
        :class:`~biopsykit.utils.dtypes.EcgRawDataFrame`
            If `return_clean` is `True`, the cleaned ECG data, otherwise the raw ECG data.

        """
        ecg = self.data[["ecg"]]
        if self.return_clean:
            algo = EcgPreprocessingNeurokit()
            algo.clean(ecg=ecg, sampling_rate_hz=self.sampling_rate_ecg)
            ecg = algo.ecg_clean_
        return ecg

    @property
    def icg(self) -> IcgRawDataFrame:
        """Return the ICG data.

        Returns
        -------
        :class:`~biopsykit.utils.dtypes.IcgRawDataFrame`
            If `return_clean` is `True`, the cleaned ICG data, otherwise the raw ICG data.

        """
        icg = self.data[["icg_der"]]
        if self.return_clean:
            algo = IcgPreprocessingBandpass()
            algo.clean(icg=icg, sampling_rate_hz=self.sampling_rate_icg)
            icg = algo.icg_clean_
        return icg

    @property
    def heartbeats(self) -> pd.DataFrame:
        """Segment heartbeats from the ECG data and return the heartbeat borders.

        Returns
        -------
        :class:`~pandas.DataFrame`
            Heartbeats as a pandas DataFrame.

        """
        heartbeat_algo = HeartbeatSegmentationNeurokit()
        heartbeat_algo.extract(ecg=self.ecg, sampling_rate_hz=self.sampling_rate_ecg)
        heartbeats = heartbeat_algo.heartbeat_list_
        return heartbeats

    @property
    def labeling_borders(self) -> pd.DataFrame:
        """Return the labeling borders for a selected participant and phase.

        Returns
        -------
        :class:`~pandas.DataFrame`
            Labeling borders as :class:`~pandas.DataFrame`.

        """
        if not self.is_single("participant"):
            raise ValueError("Labeling borders can only be loaded for a single participant.")

        data = self.data
        labeling_borders = pd.DataFrame({"start_sample": 0, "end_sample": data.shape[0] - 1}, index=[0])
        return labeling_borders

    @property
    def reference_heartbeats(self) -> pd.DataFrame:
        if not self.is_single("participant"):
            raise ValueError("Reference heartbeats can only be loaded for a single participant.")

        reference_heartbeat_folder = self.base_path.joinpath("reference_heartbeats")
        if not reference_heartbeat_folder.exists():
            raise ValueError(
                "Reference heartbeats not found. Please generate them first by calling "
                "`pepbench.datasets.time_window_icg.generate_heartbeat_borders()`."
            )

        p_id = self.subset_index.iloc[0]["participant"]

        file_path = reference_heartbeat_folder.joinpath(f"IDN{int(p_id.split('_')[1])}.csv")
        reference_heartbeats = pd.read_csv(file_path)
        reference_heartbeats = reference_heartbeats.set_index("heartbeat_id")
        if self.is_single("phase"):
            phase = self.subset_index.iloc[0]["phase"]
            split_idx = int(120 * self.sampling_rate_icg)
            if phase == "Baseline":
                # Baseline phase is 0-120s
                reference_heartbeats = reference_heartbeats[reference_heartbeats["start_sample"] <= split_idx]
            else:
                # EmotionInduction phase is 120-end
                reference_heartbeats = reference_heartbeats.loc[reference_heartbeats["start_sample"] > split_idx]
                reference_heartbeats = reference_heartbeats.assign(
                    start_sample=reference_heartbeats["start_sample"] - split_idx,
                    start_time=reference_heartbeats["start_time"] - split_idx / self.sampling_rate_icg,
                    end_sample=reference_heartbeats["end_sample"] - split_idx,
                    r_peak_sample=reference_heartbeats["r_peak_sample"] - split_idx,
                )
                reference_heartbeats.index -= reference_heartbeats.index[0]
        reference_heartbeats = reference_heartbeats.astype(
            {
                "start_sample": "Int64",
                "end_sample": "Int64",
                "r_peak_sample": "Int64",
                "rr_interval_sample": "Int64",
                "rr_interval_ms": "Float64",
            }
        )
        return reference_heartbeats

    @property
    def reference_labels_icg(self) -> pd.DataFrame:
        if not self.is_single("participant"):
            raise ValueError("Reference labels can only be loaded for a single participant.")

        p_id = self.subset_index.iloc[0]["participant"]

        file_path = self.base_path.joinpath("annotations").joinpath(f"IDN{int(p_id.split('_')[1])}.csv")

        reference_heartbeats = self.reference_heartbeats[["start_sample", "end_sample"]]
        b_points = self._load_reference_labels(file_path)
        if self.is_single("phase"):
            phase = self.subset_index.iloc[0]["phase"]
            split_idx = int(120 * self.sampling_rate_icg)
            if phase == "Baseline":
                # Baseline phase is 0-120s
                b_points = b_points[b_points["sample_relative"] <= split_idx]
            else:
                # EmotionInduction phase is 120-end
                b_points = b_points.loc[b_points["sample_relative"] > split_idx]
                b_points -= split_idx
                b_points.index -= b_points.index[0]

        ret = _get_match_heartbeat_label_ids(heartbeats=reference_heartbeats, b_points=b_points)
        # drop duplicate values
        ret = ret.drop_duplicates()

        heartbeats_match = reference_heartbeats.loc[ret]
        heartbeats_match.index.name = "heartbeat_id"
        bpoints_match = b_points.loc[ret.index]
        bpoints_match.index = heartbeats_match.index

        heartbeats_match.columns = ["start", "end"]
        bpoints_match.columns = ["B-point"]

        reference_labels = pd.concat({"heartbeats": heartbeats_match, "ICG": bpoints_match}, axis=1)
        reference_labels = reference_labels.stack([0, 1], future_stack=True)
        reference_labels.index = reference_labels.index.set_names(["heartbeat_id", "channel", "label"])
        # sort values within each heartbeat
        reference_labels = reference_labels.groupby("heartbeat_id", group_keys=False).apply(lambda x: x.sort_values())
        reference_labels = reference_labels.to_frame(name="sample_relative")
        return reference_labels

    @property
    def reference_labels_ecg(self) -> pd.DataFrame:
        """Return the reference labels for the ECG data.

        Returns
        -------
        :class:`~pandas.DataFrame`
            Reference labels for the ECG data as a pandas DataFrame

        """
        ecg = self.ecg
        q_peak_algo = QPeakExtractionVanLien2013(time_interval_ms=32)
        q_peak_algo.extract(ecg=ecg, heartbeats=self.reference_heartbeats, sampling_rate_hz=self.sampling_rate_ecg)
        q_peaks = q_peak_algo.points_
        q_peaks = q_peaks[["q_peak_sample"]]
        heartbeats = self.reference_heartbeats[["start_sample", "end_sample"]]
        heartbeats.columns = ["start", "end"]
        q_peaks.columns = ["Q-peak"]

        res = pd.concat({"heartbeat": heartbeats, "ECG": q_peaks}, axis=1)
        res = res.stack([0, 1], future_stack=True).sort_values().to_frame(name="sample_relative")
        res.index = res.index.set_names(["heartbeat_id", "channel", "label"])

        return res

    @staticmethod
    def _load_reference_labels(file_path: path_t) -> pd.DataFrame:
        data = pd.read_csv(file_path)
        data = data[["B"]]
        data.columns = ["sample_relative"]
        data.index.name = "heartbeat_id"

        return data
