import ast
import json
from pathlib import Path
from itertools import product
from typing import Optional, Sequence, Union, Dict, Tuple

from functools import lru_cache

import pandas as pd
import numpy as np
from biopsykit.signals.ecg.preprocessing._preprocessing import clean_ecg
from biopsykit.signals.ecg.segmentation import HeartbeatSegmentationNeurokit
from biopsykit.signals.icg.preprocessing import clean_icg_deriv

from pepbench._utils._types import path_t
from pepbench.datasets import BaseUnifiedPepExtractionDataset
from pepbench.datasets._helper import load_labeling_borders, compute_reference_pep
from pepbench.datasets.guardian._helper import _load_tfm_data

__all__ = ["GuardianDataset"]


_cached_get_tfm_data = lru_cache(maxsize=4)(_load_tfm_data)


class GuardianDataset(BaseUnifiedPepExtractionDataset):
    """Guardian dataset."""

    base_path: path_t
    use_cache: bool

    SAMPLING_RATES = {"ecg_1": 500, "ecg_2": 500, "icg_der": 500}
    PHASES = ["Pause", "Valsalva", "HoldingBreath", "TiltUp", "TiltDown"]

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

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Optional[Sequence[str]] = None,
        subset_index: Optional[Sequence[str]] = None,
        *,
        exclude_no_recorded_data: bool = True,
        use_cache: bool = True,
        only_labeled: bool = False,
    ):
        self.base_path = base_path
        self.exclude_no_recorded_data = exclude_no_recorded_data
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

    def _find_data_to_exclude(self) -> Sequence[Tuple[str, str]]:
        data_to_exclude = []
        if self.exclude_no_recorded_data:
            data_to_exclude = self.SUBSET_NO_RECORDED_DATA

        return data_to_exclude

    @property
    def sampling_rates(self) -> Dict[str, int]:
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
    def tfm_data(self) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
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
                    borders = labeling_borders[labeling_borders["description"].apply(lambda x: phase in x.keys())]
                    tfm_data = tfm_data_dict[phase]
                    tfm_data_dict[phase] = self._cut_to_labeling_borders(tfm_data, borders)

        return tfm_data_dict

    @property
    def icg(self):
        if not self.is_single(None):
            raise ValueError("ICG data can only be accessed for a single participant and a single phase!")
        return self.tfm_data[["icg_der"]]

    @property
    def icg_clean(self):
        icg = self.icg
        fs = self.sampling_rate_icg
        return pd.DataFrame(clean_icg_deriv(raw_signal=icg["icg_der"], sampling_rate_hz=fs), columns=["icg_der"])

    @property
    def ecg(self):
        if not self.is_single(None):
            raise ValueError("ECG data can only be accessed for a single participant and a single phase!")
        data = self.tfm_data[["ecg_2"]]
        data.columns = ["ecg"]
        return data

    @property
    def ecg_clean(self):
        ecg = self.ecg
        fs = self.sampling_rate_ecg
        return pd.DataFrame(clean_ecg(raw_signal=ecg["ecg"], sampling_rate_hz=fs, method="biosppy"), columns=["ecg"])

    # def calculate_pep_manual_labeled(self, ecg_clean):
    #     # calculate the pep from manually labeled points
    #     fs = self.SAMPLING_RATES["ecg_2"]
    #     phase = self.index["phase"][0]
    #     heartbeat_algo = HeartBeatSegmentation()
    #     heartbeat_algo.extract(ecg_clean=ecg_clean, sampling_rate_hz=fs)
    #     heartbeats = heartbeat_algo.heartbeat_list_
    #
    #     # load manually labeled points
    #     data_ICG, data_ECG = self.load_manual_labeled
    #
    #     if data_ICG is None or data_ECG is None:
    #         return None, None, None
    #     row = data_ECG[(data_ECG["Channel"] == phase) & (data_ECG["Label"] == "start")]
    #
    #     # get start of first heartbeat
    #     start_value = row["Samples"].values[0]
    #
    #     data_ICG.loc[data_ICG["Channel"] == "Artefact", "Samples"] = np.nan
    #     data_ECG.loc[data_ECG["Channel"] == "Artefact", "Samples"] = np.nan
    #     data_ICG = data_ICG[(data_ICG["Channel"] == "ICG") | (data_ICG["Channel"] == "Artefact")]
    #
    #     data_ECG = data_ECG[(data_ECG["Channel"] == "ECG") | (data_ECG["Channel"] == "Artefact")]
    #     b_points = data_ICG["Samples"].values
    #
    #     q_onset = data_ECG["Samples"].values
    #
    #     if b_points[0] < q_onset[0]:
    #
    #         b_points = b_points[1:]
    #     if b_points[-1] < q_onset[-1]:
    #
    #         q_onset = q_onset[:-1]
    #
    #     # calculate pep from manually labeled points
    #     pep_df = pd.DataFrame((b_points - q_onset) / fs * 1000, columns=["pep"])
    #     pep_df.index = range(len(pep_df))
    #     pep_df.index.name = "heartbeat_id"
    #
    #     # correcct sample number to match the start of the phase (needed since the heartbeats are only calculated inside the random selected part of the phase and not in realtion to the whole phase)
    #     heartbeats["start_sample"] = heartbeats["start_sample"] + start_value
    #     heartbeats["end_sample"] = heartbeats["end_sample"] + start_value
    #
    #     # exclude labeled points that are part of uncomplete heartbeats
    #
    #     if q_onset[0] < heartbeats["start_sample"].values[0]:
    #
    #         q_onset = q_onset[1:]
    #         b_points = b_points[1:]
    #         pep_df = pep_df[1:]
    #
    #     if q_onset[-1] > heartbeats["end_sample"].values[-1]:
    #
    #         q_onset = q_onset[:-1]
    #         b_points = b_points[:-1]
    #         pep_df = pep_df[:-1]
    #
    #     return b_points, q_onset, pep_df, start_value

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
            data = data[data["description"].apply(lambda x: phase in x.keys())]

        return data

    @property
    def reference_labels_ecg(self) -> pd.DataFrame:
        return self._load_reference_labels("ECG")

    @property
    def reference_labels_icg(self) -> pd.DataFrame:
        return self._load_reference_labels("ICG")

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
        return compute_reference_pep(self)

    @property
    def heartbeats(self):
        heartbeat_algo = HeartbeatSegmentationNeurokit()
        ecg_clean = self.ecg_clean
        ecg_clean.columns = ["ECG_Clean"]
        heartbeat_algo.extract(ecg=ecg_clean, sampling_rate_hz=self.sampling_rate_ecg)
        heartbeats = heartbeat_algo.heartbeat_list_
        return heartbeats

    # def correct_start_point(self, heartbeats, b_points=[], q_points=[], c_points=[], pep_results=[]):
    #     # correct samples such manually labeled and calculated ones match
    #
    #     rows = self.labeling_borders
    #
    #     start = rows["pos"][0]
    #
    #     # adding start to get the sample number for the complete phase and not just random selected part
    #
    #     b_points = b_points + start
    #
    #     q_points = q_points + start
    #     c_points = c_points + start
    #
    #     heartbeats["start_sample"] = heartbeats["start_sample"] + start
    #     heartbeats["end_sample"] = heartbeats["end_sample"] + start
    #     return b_points, q_points, heartbeats, c_points, pep_results

    @staticmethod
    def _cut_to_labeling_borders(data, borders) -> pd.DataFrame:
        start = borders.index[0]
        end = borders.index[-1]
        data = data.loc[start:end]
        return data
