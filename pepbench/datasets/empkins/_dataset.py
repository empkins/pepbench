import warnings
from functools import cached_property, lru_cache
from itertools import product
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union
import ast
import json
import numpy as np

import pandas as pd
from biopsykit.io.io import load_long_format_csv
from biopsykit.signals.ecg.preprocessing._preprocessing import clean_ecg
from biopsykit.utils.file_handling import get_subject_dirs
from biopsykit.signals.icg.preprocessing import clean_icg_deriv

from pepbench._utils._types import path_t
from pepbench.datasets import BaseUnifiedPepExtractionDataset
from pepbench.datasets._helper import load_labeling_borders
from pepbench.datasets.empkins._helper import _load_biopac_data, _load_timelog

_cached_get_biopac_data = lru_cache(maxsize=4)(_load_biopac_data)


class EmpkinsDataset(BaseUnifiedPepExtractionDataset):
    base_path: path_t
    use_cache: bool
    SAMPLING_RATES: Dict[str, int] = {"ecg": 1000, "icg": 1000}

    PHASES = [
        "Prep",
        "Pause_1",
        "Talk",
        "Math",
        "Pause_5",
    ]

    CONDITIONS = ["tsst", "ftsst"]

    MISSING_DATA: Sequence[str] = [
        "VP_045",
    ]  # Missing data (add participant IDs here)

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Optional[Sequence[str]] = None,
        subset_index: Optional[Sequence[str]] = None,
        *,
        exclude_missing_data: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        only_labeled: bool = False,
    ):
        # ensure pathlib
        self.base_path = base_path
        self.exclude_missing_data = exclude_missing_data
        self.use_cache = use_cache
        self.only_labeled = only_labeled
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self):
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
    def sampling_rate(self) -> Dict[str, float]:
        return self.SAMPLING_RATES

    @property
    def sampling_rate_ecg(self):
        return self.SAMPLING_RATES["ecg"]

    @property
    def sampling_rate_icg(self):
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
                    borders = labeling_borders[labeling_borders["description"].apply(lambda x: phase in x.keys())]
                    biopac_data_dict[phase] = self._cut_to_labeling_borders(data, borders)
            return biopac_data_dict

        return data

    def _cut_to_labeling_borders(self, data, labeling_borders):
        start_index = labeling_borders["sample_relative"].iloc[0]
        end_index = labeling_borders["sample_relative"].iloc[-1]
        return data.iloc[start_index:end_index]

    @property
    def icg(self):
        if not self.is_single(None):
            raise ValueError(
                "ICG data can only be accessed for a single participant, a single condition, and a single phase!"
            )
        return self.biopac[["icg_der"]]

    @property
    def icg_clean(self):
        icg = self.icg
        fs = self.sampling_rate_icg
        return pd.DataFrame(clean_icg_deriv(raw_signal=icg["icg_der"], sampling_rate_hz=fs), columns=["icg_der"])

    @property
    def ecg(self):
        if not self.is_single(None):
            raise ValueError(
                "ECG data can only be accessed for a single participant, a single condition, and a single phase!"
            )
        return self.biopac[["ecg"]]

    @property
    def ecg_clean(self):
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

    def _get_biopac_data(self, participant_id: str, condition: str, phase: str) -> Tuple[pd.DataFrame, int]:
        if self.use_cache:
            data, fs = _cached_get_biopac_data(self.base_path, participant_id, condition)
        else:
            data, fs = _load_biopac_data(self.base_path, participant_id, condition)

        if phase == "all":
            return data, fs
        else:
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
            data = data[data["description"].apply(lambda x: phase in x.keys())]

        return data

    @property
    def reference_labels_ecg(self) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        return self._load_reference_labels("ECG")

    @property
    def reference_labels_icg(self) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        return self._load_reference_labels("ICG")

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

            # reference_data = reference_data.drop(index=reference_data[reference_data["Channel"] == phase].index)
            # reference_data = reference_data.set_index(["Samples", "Label"]).sort_index()
            # reference_data = reference_data.reset_index()
            # reference_data = reference_data[["Samples", "Channel", "Label"]]
            # reference_data.columns = ["Sample", "Channel", "Label"]
            # if reference_data.iloc[0]["Label"] != "start":
            #     reference_data = reference_data.iloc[1:]
            # if reference_data.iloc[-1]["Label"] != "end":
            #     reference_data = reference_data.iloc[:-1]
            #
            # heartbeat_id = 0
            # reference_data["Heartbeat_ID"] = None
            # # Iterate over the DataFrame to assign heartbeat IDs
            # for index, row in reference_data.iterrows():
            #     if row["Label"] == "start":
            #         # Start of a new heartbeat
            #         heartbeat_id += 1
            #     # Assign the current heartbeat ID
            #     reference_data.at[index, "Heartbeat_ID"] = heartbeat_id
            #
            # reference_data = reference_data.set_index(["Heartbeat_ID", "Channel", "Label"])

            reference_data_dict[phase] = reference_data

        if self.is_single(None):
            return reference_data_dict[phases[0]]
        return pd.concat(reference_data_dict, names=["phase"])

    @property
    def reference_pep(self) -> pd.DataFrame:
        reference_icg = self.reference_labels_icg
        reference_ecg = self.reference_labels_ecg

        # artefact_ids = pd.concat(
        #     [reference_icg.reindex(["Artefact"], level="label"), reference_ecg.reindex(["Artefact"], level="label")]
        # )
        # artefact_ids = artefact_ids.index.get_level_values("heartbeat_id")

        b_points = reference_icg.xs("ICG", level="channel")
        qwave_onsets = reference_ecg.xs("ECG", level="channel")

        pep_reference = pd.concat([qwave_onsets, b_points]).sort_index()
        pep_reference = pep_reference["sample_relative"].unstack("label")

        pep_reference.columns = ["q_wave_onset_sample", "b_point_sample"]
        pep_reference = pep_reference.assign(pep_sample=-1 * pep_reference.diff(axis=1).dropna(axis=1, how="all"))
        pep_reference = pep_reference.assign(pep_ms=pep_reference["pep_sample"] / self.sampling_rate_ecg * 1000)
        return pep_reference

    # def calculate_pep_manual_labeled(self, ecg_clean, ecg_whole, heartbeats):
    #     # calculate pep out of the manual labels
    #
    #     fs = self.sampling_rates["biopac"]
    #
    #     ecg_start = ecg_whole
    #     start = (ecg_clean.index[0] - ecg_start.index[0]).total_seconds()
    #     start = int(start * fs)
    #     phase = self.index["phase"][0]
    #
    #     # load the manual labeled data
    #     data_ICG, data_ECG = self.load_manual_labeled
    #
    #     if data_ICG is None or data_ECG is None:
    #         return None, None, None
    #     row = data_ECG[(data_ECG["Channel"] == phase) & (data_ECG["Label"] == "start")]
    #
    #     row_end = data_ECG[(data_ECG["Channel"] == phase) & (data_ECG["Label"] == "end")]
    #
    #     # selct only part of the labels within the specific phase
    #     data_ECG = data_ECG.iloc[row.index[0] : row_end.index[0] + 2]
    #     data_ICG = data_ICG.iloc[row.index[0] : row_end.index[0] + 2]
    #     start_value = row["Samples"].values[0]
    #     end_value = row_end["Samples"].values[0]
    #
    #     data_ICG = data_ICG[(data_ICG["Channel"] == "ICG") | (data_ICG["Channel"] == "Artefact")]
    #
    #     data_ECG = data_ECG[(data_ECG["Channel"] == "ECG") | (data_ECG["Channel"] == "Artefact")]
    #
    #     heartbeats["start_sample"] = heartbeats["start_sample"] + start
    #     heartbeats["end_sample"] = heartbeats["end_sample"] + start
    #
    #     # exclude labeled points that are part of uncomplete heartbeats
    #     heartbeats = heartbeats[(heartbeats["start_sample"] >= start_value) & (heartbeats["end_sample"] <= end_value)]
    #
    #     if data_ECG["Samples"].values[0] < heartbeats["start_sample"].values[0]:
    #         data_ECG = data_ECG[1:]
    #
    #     if data_ECG["Samples"].values[-1] > heartbeats["end_sample"].values[-1]:
    #         data_ECG = data_ECG[:-1]
    #
    #     if data_ICG["Samples"].values[0] < heartbeats["start_sample"].values[0]:
    #         data_ICG = data_ICG[1:]
    #
    #     if data_ICG["Samples"].values[-1] > heartbeats["end_sample"].values[-1]:
    #         data_ICG = data_ICG[:-1]
    #
    #     # insert nan for all artefacts
    #     data_ICG.loc[data_ICG["Channel"] == "Artefact", "Samples"] = np.nan
    #     data_ECG.loc[data_ECG["Channel"] == "Artefact", "Samples"] = np.nan
    #
    #     b_points = data_ICG["Samples"].values
    #
    #     q_onset = data_ECG["Samples"].values
    #
    #     if b_points[0] < q_onset[0]:
    #         b_points = b_points[1:]
    #     if b_points[-1] < q_onset[-1]:
    #         q_onset = q_onset[:-1]
    #
    #     # insert nan values for heartbeats in which no points were labeled
    #     count = 0
    #     average_time = np.mean(heartbeats["rr_interval_samples"])
    #     outliers = heartbeats[heartbeats["rr_interval_samples"] < (0.7 * average_time)]
    #
    #     for h in heartbeats.index:
    #         # ignore if the heartbeat may be not a real heartbeat
    #         if h in outliers.index:
    #             count += 1
    #             continue
    #         start = heartbeats.loc[h]["start_sample"]
    #         end = heartbeats.loc[h]["end_sample"]
    #
    #         if not any(start <= x <= end for x in b_points):
    #             if not (pd.isna(b_points[count])):
    #                 b_points = np.insert(b_points, count, np.nan)
    #         if not any(start <= x <= end for x in q_onset):
    #             if not (pd.isna(q_onset[count])):
    #                 q_onset = np.insert(q_onset, count, np.nan)
    #         count += 1
    #     # calculate pep from start and end points
    #     pep_df = pd.DataFrame((b_points - q_onset) / fs * 1000, columns=["pep"])
    #     pep_df.index = range(len(pep_df))
    #     pep_df.index.name = "heartbeat_id"
    #
    #     return b_points, q_onset, pep_df, start_value
    #

    #
    # def correct_start_points(
    #     self, ecg_clean, ecg_start, heartbeats, b_points=[], q_points=[], c_points=[], pep_results=[]
    # ):
    #     # used to correct the start points of the calculated points to match the manually labeled points
    #     # get the start sample of the phase (needed since all phases are combined after one another and sample count is based on the whole data and not just the phase)
    #
    #     fs = self.sampling_rates["biopac"]
    #     start = (ecg_clean.index[0] - ecg_start.index[0]).total_seconds()
    #     start_phase = int(start * fs)
    #     rows = self.load_annotations()
    #
    #     # start and end of the random selected part of the phase
    #     start = rows["pos"][0]
    #     end = rows["pos"][1]
    #
    #     heartbeats["start_sample"] = heartbeats["start_sample"] + start_phase
    #     heartbeats["end_sample"] = heartbeats["end_sample"] + start_phase
    #
    #     heartbeats = heartbeats.loc[(heartbeats["start_sample"] >= start)]
    #     heartbeats = heartbeats.loc[(heartbeats["end_sample"] <= end)]
    #     ids = heartbeats.index
    #
    #     # correct the sample count of the calculated points to match the manually labeled points
    #     if len(b_points) != 0:
    #         b_points = b_points.iloc[ids]
    #         b_points = b_points + start_phase
    #
    #     if len(q_points) != 0:
    #         q_points = q_points.iloc[ids]
    #         q_points = q_points + start_phase
    #
    #     if len(c_points) != 0:
    #         c_points = c_points.iloc[ids]
    #         c_points = c_points + start_phase
    #     if len(pep_results) != 0:
    #         pep_results = pep_results.iloc[ids]
    #
    #     return b_points, q_points, heartbeats, c_points, pep_results
