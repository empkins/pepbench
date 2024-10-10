from collections.abc import Sequence
from functools import cached_property, lru_cache
from itertools import product
from typing import ClassVar, Optional, Union

import pandas as pd
from biopsykit.signals.ecg.preprocessing._preprocessing import clean_ecg
from biopsykit.signals.ecg.segmentation import HeartbeatSegmentationNeurokit
from biopsykit.signals.icg.preprocessing import clean_icg_deriv
from biopsykit.utils.file_handling import get_subject_dirs
from biopsykit.metadata import bmi

from pepbench.datasets import BaseUnifiedPepExtractionDataset
from pepbench.datasets._helper import compute_reference_heartbeats, compute_reference_pep, load_labeling_borders
from pepbench.datasets.empkins._helper import _load_biopac_data, _load_timelog
from pepbench.utils._types import path_t

_cached_get_biopac_data = lru_cache(maxsize=4)(_load_biopac_data)
# cache_dir = "./cachedir"
# memory = Memory(location=cache_dir, verbose=0)
# _cached_get_biopac_data = memory.cache(_load_biopac_data)


class EmpkinsDataset(BaseUnifiedPepExtractionDataset):
    base_path: path_t
    use_cache: bool
    SAMPLING_RATES: ClassVar[dict[str, int]] = {"ecg": 1000, "icg": 1000}

    PHASES: ClassVar[Sequence[str]] = ["Prep", "Pause_1", "Talk", "Math", "Pause_5"]

    CONDITIONS: ClassVar[Sequence[str]] = ["tsst", "ftsst"]

    GENDER_MAPPING: ClassVar[dict[int, str]] = {1: "Female", 2: "Male"}

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Optional[Sequence[str]] = None,
        subset_index: Optional[Sequence[str]] = None,
        *,
        exclude_missing_data: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        only_labeled: bool = False,
    ) -> None:
        # ensure pathlib
        self.base_path = base_path
        self.exclude_missing_data = exclude_missing_data
        self.use_cache = use_cache
        self.only_labeled = only_labeled
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self) -> pd.DataFrame:
        # data is located in a folder named "Data" and data per participant is located in folders named "VP_xx"
        participant_ids = [
            participant_dir.name
            for participant_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_[0-9]{3}")
        ]

        # excludes participants where parts of data are missing
        if self.exclude_missing_data:
            for p_id in self.MISSING_DATA:
                if p_id in participant_ids:
                    participant_ids.remove(p_id)

        index = list(product(participant_ids, self.CONDITIONS, self.PHASES))
        index = pd.DataFrame(index, columns=["participant", "condition", "phase"])
        return index

    @property
    def sampling_rate(self) -> dict[str, float]:
        return self.SAMPLING_RATES

    @property
    def sampling_rate_ecg(self) -> int:
        return self.SAMPLING_RATES["ecg"]

    @property
    def sampling_rate_icg(self) -> int:
        return self.SAMPLING_RATES["icg"]

    @cached_property
    def biopac(self) -> pd.DataFrame:
        participant = self.index["participant"][0]
        condition = self.index["condition"][0]

        if self.is_single(None):
            phase = self.index["phase"][0]
        elif self.is_single(["participant", "condition"]):
            phase = "all"
        else:
            raise ValueError("Biopac data can only be accessed for one single participant and condition at once!")

        data, fs = self._get_biopac_data(participant, condition, phase)

        if self.only_labeled:
            biopac_data_dict = {}
            labeling_borders = self.labeling_borders

            if self.is_single(None):
                biopac_data_dict = self._cut_to_labeling_borders(data, labeling_borders)
            else:
                for phase in self.PHASES:
                    borders = labeling_borders[labeling_borders["description"].apply(lambda x, ph=phase: ph in x)]
                    biopac_data_dict[phase] = self._cut_to_labeling_borders(data, borders)
            return biopac_data_dict

        return data

    def _cut_to_labeling_borders(self, data: pd.DataFrame, labeling_borders: pd.DataFrame) -> pd.DataFrame:
        start_index = labeling_borders["sample_relative"].iloc[0]
        end_index = labeling_borders["sample_relative"].iloc[-1]
        return data.iloc[start_index:end_index]

    @property
    def icg(self) -> pd.DataFrame:
        if not self.is_single(None):
            raise ValueError(
                "ICG data can only be accessed for a single participant, a single condition, and a single phase!"
            )
        return self.biopac[["icg_der"]]

    @property
    def icg_clean(self) -> pd.DataFrame:
        icg = self.icg
        fs = self.sampling_rate_icg
        return pd.DataFrame(clean_icg_deriv(raw_signal=icg["icg_der"], sampling_rate_hz=fs), columns=["icg_der"])

    @property
    def ecg(self) -> pd.DataFrame:
        if not self.is_single(None):
            raise ValueError(
                "ECG data can only be accessed for a single participant, a single condition, and a single phase!"
            )
        return self.biopac[["ecg"]]

    @property
    def ecg_clean(self) -> pd.DataFrame:
        ecg = self.ecg
        fs = self.sampling_rate_ecg
        return pd.DataFrame(clean_ecg(raw_signal=ecg["ecg"], sampling_rate_hz=fs, method="biosppy"), columns=["ecg"])

    @property
    def timelog(self) -> pd.DataFrame:
        if self.is_single(None):
            participant = self.index["participant"][0]
            condition = self.index["condition"][0]
            phase = self.index["phase"][0]
            return self._get_timelog(participant, condition, phase)

        if self.is_single(["participant", "condition"]):
            if not self._all_phases_selected():
                raise ValueError("Timelog can only be accessed for all phases or one specific phase!")

            participant = self.index["participant"][0]
            condition = self.index["condition"][0]
            return self._get_timelog(participant, condition, "all")

        raise ValueError("Timelog can only be accessed for a single participant and a single condition at once!")

    def _get_biopac_data(self, participant_id: str, condition: str, phase: str) -> tuple[pd.DataFrame, int]:
        if self.use_cache:
            data, fs = _cached_get_biopac_data(self.base_path, participant_id, condition)
        else:
            data, fs = _load_biopac_data(self.base_path, participant_id, condition)

        if phase == "all":
            return data, fs
        # cut biopac data to specified phase
        timelog = self.timelog
        phase_start = timelog[phase]["start"][0]
        phase_end = timelog[phase]["end"][0]
        data = data.loc[phase_start:phase_end]
        return data, fs

    def _get_timelog(self, participant_id: str, condition: str, phase: str) -> pd.DataFrame:
        return _load_timelog(self.base_path, participant_id, condition, phase)

    def _all_phases_selected(self) -> bool:
        # check if all phases are selected
        return len(self.index["phase"]) == len(self.PHASES)

    @property
    def labeling_borders(self) -> pd.DataFrame:
        participant = self.index["participant"][0]
        condition = self.index["condition"][0]

        if not self.is_single("participant"):
            raise ValueError("Labeling borders can only be accessed for a single participant.")

        file_path = self.base_path.joinpath(
            f"data_per_subject/{participant}/{condition}/biopac/reference_labels/{participant}_{condition}_labeling_borders.csv"
        )
        data = load_labeling_borders(file_path)

        if self.is_single(None):
            phase = self.index["phase"][0]
            data = data[data["description"].apply(lambda x, ph=phase: ph in x)]

        return data

    @property
    def reference_heartbeats(self) -> pd.DataFrame:
        return self._load_reference_heartbeats()

    @property
    def reference_labels_ecg(self) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
        return self._load_reference_labels("ECG")

    @property
    def reference_labels_icg(self) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
        return self._load_reference_labels("ICG")

    def _load_reference_heartbeats(self) -> pd.DataFrame:
        reference_ecg = self.reference_labels_ecg
        reference_heartbeats = reference_ecg.reindex(["heartbeat"], level="channel")
        reference_heartbeats = compute_reference_heartbeats(reference_heartbeats)
        return reference_heartbeats

    def _load_reference_labels(self, channel: str) -> pd.DataFrame:
        participant = self.index["participant"][0]
        condition = self.index["condition"][0]
        phases = self.index["phase"]

        if not (self.is_single(None) or len(phases) == len(self.PHASES)):
            raise ValueError(
                "Reference data can only be accessed for a single participant and ALL phases or "
                "for a single participant and a SINGLE phase."
            )

        reference_data_dict = {}
        for phase in phases:
            file_path = self.base_path.joinpath(
                f"data_per_subject/{participant}/{condition}/biopac/reference_labels/"
                f"{participant}_{condition}_reference_labels_{phase.lower()}_{channel.lower()}.csv"
            )
            reference_data = pd.read_csv(file_path)
            reference_data = reference_data.set_index(["heartbeat_id", "channel", "label"])

            start_idx = self.get_subset(phase=phase).labeling_borders.iloc[0]
            reference_data = reference_data.assign(
                sample_relative=reference_data["sample_absolute"] - start_idx["sample_absolute"]
            )

            reference_data_dict[phase] = reference_data

        if self.is_single(None):
            return reference_data_dict[phases[0]]
        return pd.concat(reference_data_dict, names=["phase"])

    @property
    def reference_pep(self) -> pd.DataFrame:
        return compute_reference_pep(self)

    @property
    def heartbeats(self) -> pd.DataFrame:
        heartbeat_algo = HeartbeatSegmentationNeurokit(variable_length=True)
        ecg_clean = self.ecg_clean
        ecg_clean.columns = ["ECG_Clean"]
        heartbeat_algo.extract(ecg=ecg_clean, sampling_rate_hz=self.sampling_rate_ecg)
        heartbeats = heartbeat_algo.heartbeat_list_
        return heartbeats

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
