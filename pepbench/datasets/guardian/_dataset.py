from collections.abc import Sequence
from functools import lru_cache
from itertools import product
from typing import ClassVar, Optional, Union

import pandas as pd
from biopsykit.metadata import bmi
from biopsykit.signals.ecg.preprocessing import EcgPreprocessingNeurokit
from biopsykit.signals.ecg.segmentation import HeartbeatSegmentationNeurokit
from biopsykit.signals.icg.preprocessing import IcgPreprocessingBandpass

from pepbench.datasets import BaseUnifiedPepExtractionDataset
from pepbench.datasets._helper import compute_reference_heartbeats, compute_reference_pep, load_labeling_borders
from pepbench.datasets.guardian._helper import _load_tfm_data
from pepbench.utils._types import path_t

__all__ = ["GuardianDataset"]


_cached_get_tfm_data = lru_cache(maxsize=4)(_load_tfm_data)
# cache_dir = "./cachedir"
# memory = Memory(location=cache_dir, verbose=0)
# _cached_get_tfm_data = memory.cache(_load_tfm_data)


class GuardianDataset(BaseUnifiedPepExtractionDataset):
    """Guardian dataset."""

    base_path: path_t
    use_cache: bool

    SAMPLING_RATES: ClassVar[dict[str, int]] = {"ecg_1": 500, "ecg_2": 500, "icg_der": 500}
    PHASES: ClassVar[tuple[str, ...]] = ["Pause", "Valsalva", "HoldingBreath", "TiltUp", "TiltDown"]

    GENDER_MAPPING: ClassVar[dict[str, str]] = {"M": "Male", "F": "Female"}

    SUBSET_NO_RECORDED_DATA = (
        ("GDN0006", "HoldingBreath"),
        ("GDN0009", "HoldingBreath"),
        ("GDN0010", "Valsalva"),
        ("GDN0017", "Pause"),
        ("GDN0018", "TiltDown"),
        ("GDN0020", "TiltUp"),
        ("GDN0022", "TiltUp"),
        ("GDN0024", "TiltDown"),
        ("GDN0025", "Valsalva"),
        ("GDN0028", "TiltUp"),
        ("GDN0030", "Pause"),
        ("GDN0030", "TiltUp"),
    )
    SUBSET_NOISY_DATA = (
        ("GDN0025", "TiltUp"),
        ("GDN0025", "TiltDown"),
    )

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Optional[Sequence[str]] = None,
        subset_index: Optional[Sequence[str]] = None,
        *,
        exclude_no_recorded_data: bool = True,
        exclude_noisy_data: bool = True,
        use_cache: bool = True,
        only_labeled: bool = False,
    ) -> None:
        self.base_path = base_path
        self.exclude_no_recorded_data = exclude_no_recorded_data
        self.exclude_noisy_data = exclude_noisy_data
        self.data_to_exclude = self._find_data_to_exclude()
        self.use_cache = use_cache
        self.only_labeled = only_labeled
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self) -> pd.DataFrame:
        overview_df = pd.read_csv(self.base_path.joinpath("dataset_overview.csv"), sep=";")
        pids = list(overview_df["participant"])
        index = list(product(pids, self.PHASES))
        index = pd.DataFrame(index, columns=["participant", "phase"])
        for item in self.data_to_exclude:
            index = index.drop(index[(index["participant"] == item[0]) & (index["phase"] == item[1])].index)
        index = index.reset_index(drop=True)

        return index

    def _find_data_to_exclude(self) -> Sequence[tuple[str, str]]:
        data_to_exclude = []
        if self.exclude_no_recorded_data:
            data_to_exclude.extend(self.SUBSET_NO_RECORDED_DATA)
        if self.exclude_noisy_data:
            data_to_exclude.extend(self.SUBSET_NOISY_DATA)

        return data_to_exclude

    @property
    def sampling_rates(self) -> dict[str, int]:
        return self.SAMPLING_RATES

    @property
    def sampling_rate_ecg(self) -> int:
        return self.SAMPLING_RATES["ecg_2"]

    @property
    def sampling_rate_icg(self) -> int:
        return self.SAMPLING_RATES["icg_der"]

    @property
    def date(self) -> Union[pd.Series, pd.Timestamp]:
        metadata_path = self.base_path.joinpath("metadata/recording_timestamps.xlsx")
        metadata = pd.read_excel(metadata_path)
        metadata = metadata.set_index("participant")["date"]
        if self.is_single("participant"):
            return metadata[self.index["participant"][0]]
        return metadata

    @property
    def tfm_data(self) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
        participant = self.index["participant"][0]
        phases = self.index["phase"]

        if not (self.is_single(None) or len(phases) == len(self.PHASES)):
            raise ValueError(
                "TFM data can only be accessed for a single participant and ALL phases or "
                "for a single participant and a SINGLE phase."
            )

        tfm_path = self.base_path.joinpath(f"data_raw/{participant}/tfm_data/{participant.lower()}_no01.mat")

        if self.use_cache:
            tfm_data_dict = _cached_get_tfm_data(tfm_path, self.date)
        else:
            tfm_data_dict = _load_tfm_data(tfm_path, self.date)

        if self.is_single(None):
            tfm_data_dict = tfm_data_dict[phases[0]]

        if self.only_labeled:
            labeling_borders = self.labeling_borders

            if self.is_single(None):
                tfm_data_dict = self._cut_to_labeling_borders(tfm_data_dict, labeling_borders)
            else:
                for phase in phases:
                    borders = labeling_borders[labeling_borders["description"].apply(lambda x, ph=phase: ph in x)]
                    tfm_data = tfm_data_dict[phase]
                    tfm_data_dict[phase] = self._cut_to_labeling_borders(tfm_data, borders)

        return tfm_data_dict

    @property
    def icg(self) -> pd.DataFrame:
        if not self.is_single(None):
            raise ValueError("ICG data can only be accessed for a single participant and a single phase!")
        return self.tfm_data[["icg_der"]]

    @property
    def icg_clean(self) -> pd.DataFrame:
        algo = IcgPreprocessingBandpass()
        algo.clean(icg=self.icg, sampling_rate_hz=self.sampling_rate_icg)
        return algo.icg_clean_

    @property
    def ecg(self) -> pd.DataFrame:
        if not self.is_single(None):
            raise ValueError("ECG data can only be accessed for a single participant and a single phase!")
        data = self.tfm_data[["ecg_2"]]
        data.columns = ["ecg"]
        return data

    @property
    def ecg_clean(self) -> pd.DataFrame:
        algo = EcgPreprocessingNeurokit()
        algo.clean(ecg=self.ecg, sampling_rate_hz=self.sampling_rate_ecg)
        return algo.ecg_clean_

    @property
    def labeling_borders(self) -> pd.DataFrame:
        participant = self.index["participant"][0]

        if not self.is_single("participant"):
            raise ValueError("Labeling borders can only be accessed for a single participant.")

        file_path = self.base_path.joinpath(
            f"data_raw/{participant}/tfm_data/reference_labels/{participant}_labeling_borders.csv"
        )
        data = load_labeling_borders(file_path)

        if self.is_single(None):
            phase = self.index["phase"][0]
            data = data[data["description"].apply(lambda x: phase in x)]

        return data

    @property
    def reference_heartbeats(self) -> pd.DataFrame:
        return self._load_reference_heartbeats()

    @property
    def reference_labels_ecg(self) -> pd.DataFrame:
        return self._load_reference_labels("ECG")

    @property
    def reference_labels_icg(self) -> pd.DataFrame:
        return self._load_reference_labels("ICG")

    def _load_reference_heartbeats(self) -> pd.DataFrame:
        reference_ecg = self.reference_labels_ecg
        reference_heartbeats = reference_ecg.reindex(["heartbeat"], level="channel")
        reference_heartbeats = compute_reference_heartbeats(reference_heartbeats)
        return reference_heartbeats

    def _load_reference_labels(self, channel: str) -> pd.DataFrame:
        participant = self.index["participant"][0]
        phases = self.index["phase"]

        if not (self.is_single(None) or len(phases) == len(self.PHASES)):
            raise ValueError(
                "Reference data can only be accessed for a single participant and ALL phases or "
                "for a single participant and a SINGLE phase."
            )
        reference_data_dict = {}
        for phase in phases:
            file_path = self.base_path.joinpath(
                f"data_raw/{participant}/tfm_data/reference_labels/{participant}_reference_labels_{phase}_{channel}.csv"
            )
            reference_data = pd.read_csv(file_path)
            reference_data = reference_data.set_index(["heartbeat_id", "channel", "label"])

            reference_data_dict[phase] = reference_data

        if self.is_single(None):
            return reference_data_dict[phases[0]]
        return pd.concat(reference_data_dict, names=["phase"])

    @property
    def reference_pep(self) -> pd.DataFrame:
        return compute_reference_pep(self)

    @property
    def heartbeats(self) -> pd.DataFrame:
        heartbeat_algo = HeartbeatSegmentationNeurokit()
        ecg_clean = self.ecg_clean
        heartbeat_algo.extract(ecg=ecg_clean, sampling_rate_hz=self.sampling_rate_ecg)
        heartbeats = heartbeat_algo.heartbeat_list_
        return heartbeats

    @staticmethod
    def _cut_to_labeling_borders(data: pd.DataFrame, borders: pd.DataFrame) -> pd.DataFrame:
        start = borders.index[0]
        end = borders.index[-1]
        data = data.loc[start:end]
        return data

    @property
    def metadata(self) -> pd.DataFrame:
        data = pd.read_csv(self.base_path.joinpath("metadata/demographics.csv"))
        data = data.set_index("participant")

        return data.loc[self.index["participant"].unique()]

    @property
    def age(self) -> pd.DataFrame:
        return self.metadata[["Age"]]

    @property
    def gender(self) -> pd.DataFrame:
        return self.metadata[["Gender"]].replace(self.GENDER_MAPPING)

    @property
    def bmi(self) -> pd.DataFrame:
        return bmi(self.metadata[["Weight", "Height"]])
