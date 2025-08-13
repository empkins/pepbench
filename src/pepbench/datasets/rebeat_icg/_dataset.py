import re
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import ClassVar

import pandas as pd
from biopsykit.signals.ecg.event_extraction import QPeakExtractionVanLien2013
from biopsykit.signals.ecg.segmentation import HeartbeatSegmentationNeurokit
from biopsykit.signals.icg.preprocessing import IcgPreprocessingBandpass
from biopsykit.utils.dtypes import EcgRawDataFrame, IcgRawDataFrame

from pepbench.datasets import BasePepDatasetWithAnnotations
from pepbench.datasets._base_pep_extraction_dataset import base_pep_extraction_docfiller

__all__ = ["ReBeatIcgDataset"]

from pepbench.datasets.rebeat_icg._helper import _load_b_point_annotations, _load_mat_data
from pepbench.utils._types import path_t

_cached_get_mat_data = lru_cache(maxsize=10)(_load_mat_data)


@base_pep_extraction_docfiller
class ReBeatIcgDataset(BasePepDatasetWithAnnotations):
    base_path: Path
    use_cache: bool

    exclude_annotation_errors: bool

    SAMPLING_RATE: ClassVar[int] = 250

    SUBSET_ANNOTATION_ERRORS = (
        ("11", "Baseline"),
        ("11", "CognitiveWorkload"),
        ("12", "CognitiveWorkload"),
        ("16", "Baseline"),
        ("23", "Baseline"),
        ("23", "CognitiveWorkload"),
    )

    PHASES: ClassVar[dict[str, str]] = {"BL": "Baseline", "CW": "CognitiveWorkload"}
    PHASES_INVERSE: ClassVar[dict[str, str]] = {v: k for k, v in PHASES.items()}

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Sequence[str] | None = None,
        subset_index: Sequence[str] | None = None,
        *,
        return_clean: bool = True,
        exclude_annotation_errors: bool = True,
        use_cache: bool = True,
        only_labeled: bool = False,
    ) -> None:
        """Initialize a new ``GuardianDataset`` instance.

        Parameters
        ----------
        base_path : :class:`~pathlib.Path` or str
            Path to the root directory of the Guardian dataset.
        return_clean : bool
            Whether to return the preprocessed/cleaned ECG and ICG data when accessing the respective properties.
            Default: ``True``.
        use_cache : bool, optional
            Whether to use caching for loading TFM data. Default: ``True``.
        only_labeled : bool, optional
            Whether to only return segments that are labeled (i.e., cut the data to the labeling borders).
            This is necessary when using the dataset for evaluating the performance of PEP extraction algorithms or for
            training ML-based PEP extraction algorithms. Default: ``False``.
        """
        self.base_path = base_path
        self.use_cache = use_cache
        self.exclude_annotation_errors = exclude_annotation_errors
        self.data_to_exclude = self._find_data_to_exclude()
        super().__init__(
            groupby_cols=groupby_cols, subset_index=subset_index, return_clean=return_clean, only_labeled=only_labeled
        )

    def _sanitize_params(self) -> None:
        # ensure pathlib
        self.base_path = Path(self.base_path)

    def create_index(self) -> pd.DataFrame:
        self._sanitize_params()

        if self.only_labeled:
            label_border_folder = self.base_path.joinpath("04_LabelingAndHeartBeatBorders")
            file_list = sorted(label_border_folder.glob("*.csv"))
            label_region_ids = list(
                filter(
                    lambda x: len(x) > 0,
                    [
                        re.findall(r"LabelingBorders_Subject_(\d+)_task_(\w+)_label_(\d+).csv", path.name)
                        for path in file_list
                    ],
                )
            )
            matches = [label_region_id[0] for label_region_id in label_region_ids]
            matches = [(participant.zfill(2), phase, int(label_period)) for participant, phase, label_period in matches]
            index = pd.DataFrame(matches, columns=["participant", "phase", "label_period"])
            index = index.set_index(["participant", "phase"]).rename(index=self.PHASES, level="phase").sort_index()
        else:
            if self.return_clean:
                file_prefix = "FilteredData"
                data_path = self.base_path.joinpath(f"02_{file_prefix}")
            else:
                file_prefix = "RawData"
                data_path = self.base_path.joinpath(f"01_{file_prefix}")

            file_list = sorted(data_path.glob("*.mat"))
            matches = [re.findall(rf"{file_prefix}_Subject_(\d+)_task_(\w+).mat", path.name)[0] for path in file_list]
            # add leading zeros to participant numbers
            matches = [(participant.zfill(2), phase) for participant, phase in matches]
            index = pd.DataFrame(matches, columns=["participant", "phase"])
            index = index.set_index(["participant", "phase"]).rename(index=self.PHASES, level="phase").sort_index()

        index = index.reset_index()
        for item in self.data_to_exclude:
            index = index.drop(index[(index["participant"] == item[0]) & (index["phase"] == item[1])].index)
            index = index.reset_index(drop=True)
        return index

    def _find_data_to_exclude(self) -> Sequence[tuple[str, str]]:
        data_to_exclude = []
        if self.exclude_annotation_errors:
            data_to_exclude.extend(self.SUBSET_ANNOTATION_ERRORS)

        return data_to_exclude

    @property
    def sampling_rate_ecg(self) -> int:
        """Return the sampling rate of the ECG signal.

        Returns
        -------
        int
            Sampling rate of the ECG signal in Hz.

        """
        return self.SAMPLING_RATE

    @property
    def sampling_rate_icg(self) -> int:
        """Return the sampling rate of the ICG signal.

        Returns
        -------
        int
            Sampling rate of the ICG signal in Hz.

        """
        return self.SAMPLING_RATE

    @property
    def data(self) -> pd.DataFrame:
        if not self.is_single(None):
            if self.only_labeled:
                error_msg = "Data can only be accessed for a single participant, phase, and label period."
            else:
                error_msg = "Data can only be accessed for a single participant and phase."
            raise ValueError(error_msg)
        p_id = self.index["participant"][0]
        phase = self.index["phase"][0]

        if self.return_clean:
            file_prefix = "FilteredData"
            folder_path = self.base_path.joinpath(f"02_{file_prefix}")
        else:
            file_prefix = "RawData"
            folder_path = self.base_path.joinpath(f"01_{file_prefix}")

        file_path = folder_path.joinpath(f"{file_prefix}_Subject_{int(p_id)}_task_{self.PHASES_INVERSE[phase]}.mat")
        data = _cached_get_mat_data(file_path) if self.use_cache else _load_mat_data(file_path)

        if data["samplFreq"] != self.SAMPLING_RATE:
            raise ValueError(f"Sampling rate of {data['samplFreq']} does not match expected {self.SAMPLING_RATE}.")

        data = pd.DataFrame({"ecg": data["ECG"], "icg_der": data["ICG"]})
        data.index /= self.SAMPLING_RATE
        data.index.name = "t"
        data.index = pd.to_timedelta(data.index, unit="s")

        if self.only_labeled:
            labeling_borders = self.labeling_borders.iloc[0]
            data = data.iloc[labeling_borders["start_sample"] : labeling_borders["end_sample"]]
            data.index -= data.index[0]

        return data

    @property
    def ecg(self) -> EcgRawDataFrame:
        return self.data[["ecg"]]

    @property
    def icg(self) -> IcgRawDataFrame:
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
        labeling_border_folder = self.base_path.joinpath("04_LabelingAndHeartBeatBorders")
        if not labeling_border_folder.exists():
            raise ValueError(
                "Labeling borders not found. Please generate them first by calling "
                "`generate_labeling_and_heartbeat_borders()`."
            )

        p_id = self.index["participant"][0]
        phase = self.index["phase"][0]
        label_period = int(self.index["label_period"][0])
        file_path = labeling_border_folder.joinpath(
            f"LabelingBorders_Subject_{int(p_id)}_task_{self.PHASES_INVERSE[phase]}_label_{label_period}.csv"
        )
        labeling_borders = pd.read_csv(file_path)
        labeling_borders = labeling_borders.astype(int)
        return labeling_borders

    @property
    def reference_heartbeats(self) -> pd.DataFrame:
        """Return the reference heartbeats.

        Returns
        -------
        :class:`~pandas.DataFrame`
            Reference heartbeats as a pandas DataFrame

        """
        reference_heartbeat_folder = self.base_path.joinpath("04_LabelingAndHeartBeatBorders")
        if not reference_heartbeat_folder.exists():
            raise ValueError(
                "Reference heartbeats not found. Please generate them first by calling "
                "`pepbench.datasets.rebeat_icg.generate_labeling_and_heartbeat_borders()`."
            )

        p_id = self.index["participant"][0]
        phase = self.index["phase"][0]
        label_period = self.index["label_period"][0]
        file_path = reference_heartbeat_folder.joinpath(
            f"Heartbeats_Subject_{int(p_id)}_task_{self.PHASES_INVERSE[phase]}_label_{label_period}.csv"
        )
        heartbeats = pd.read_csv(file_path)
        heartbeats = heartbeats.set_index("heartbeat_id")
        heartbeats.index -= heartbeats.index[0]

        heartbeats = heartbeats.assign(
            start_sample=heartbeats["start_sample"] - self.labeling_borders["start_sample"].iloc[0],
            end_sample=heartbeats["end_sample"] - self.labeling_borders["start_sample"].iloc[0],
            r_peak_sample=heartbeats["r_peak_sample"] - self.labeling_borders["start_sample"].iloc[0],
        )

        return heartbeats

    @property
    def reference_labels_icg(self) -> pd.DataFrame:
        """Return the reference labels for the ICG data.

        Returns
        -------
        :class:`~pandas.DataFrame`
            Reference labels for the ICG data as a pandas DataFrame

        """
        heartbeats = self.reference_heartbeats[["start_sample", "end_sample"]]
        b_points = self._load_b_point_annotations()
        labeling_borders = self.labeling_borders.iloc[0]

        b_points = b_points[
            (b_points["sample_relative"] > labeling_borders["start_sample"])
            & (b_points["sample_relative"] < labeling_borders["end_sample"])
        ]
        b_points -= labeling_borders["start_sample"]
        b_points.index -= b_points.index[0]

        data = pd.concat(
            {
                ("heartbeats", "start"): heartbeats["start_sample"],
                ("heartbeats", "end"): heartbeats["end_sample"],
                ("ICG", "B-point"): b_points["sample_relative"],
            },
            axis=1,
        )
        data = data.stack([0, 1], future_stack=True).astype("Int64")
        data = pd.DataFrame(data, columns=["sample_relative"])
        data.index = data.index.set_names(["heartbeat_id", "channel", "label"])
        data = data.sort_values(by="sample_relative")
        return data

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

    def _load_b_point_annotations(self) -> pd.DataFrame:
        p_id = self.index["participant"][0]
        phase = self.index["phase"][0]
        file_path = self.base_path.joinpath(
            f"03_ExpertAnnotations/Annotat_Subject_{int(p_id)}_task_{self.PHASES_INVERSE[phase]}.mat"
        )
        return _load_b_point_annotations(file_path, load_func=_cached_get_mat_data if self.use_cache else None)
