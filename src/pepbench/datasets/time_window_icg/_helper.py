import re

import numpy as np
import pandas as pd
from biopsykit.signals.ecg.segmentation import HeartbeatSegmentationNeurokit

from pepbench.utils._types import path_t


def _load_txt_data(file_path: path_t) -> pd.DataFrame:
    data = pd.read_csv(file_path, header=None)
    data.columns = ["icg", "icg_der", "ecg"]
    return data


def _get_match_heartbeat_label_ids(heartbeats: pd.DataFrame, b_points: pd.DataFrame) -> pd.Series:
    heartbeat_ids = pd.Series(index=heartbeats.index, name="heartbeat_id")
    heartbeat_ids.index.name = "heartbeat_id_b_point"
    for i, b_point in b_points.iterrows():
        idx = np.where(
            (heartbeats["start_sample"] <= b_point["sample_relative"])
            & (heartbeats["end_sample"] >= b_point["sample_relative"])
        )[0]
        if len(idx) == 0:
            # If no match is found, add NaN
            heartbeat_ids[i] = np.nan
        else:
            # If a match is found, add the index of the heartbeat
            heartbeat_ids[i] = idx[0]

    heartbeat_ids = heartbeat_ids.dropna().astype(int)
    return heartbeat_ids


def generate_heartbeat_borders(base_path: path_t) -> None:
    data_folder = base_path.joinpath("signals")
    annotation_folder = base_path.joinpath("annotations")
    heartbeat_folder = base_path.joinpath("reference_heartbeats")
    heartbeat_folder.mkdir(parents=True, exist_ok=True)
    for annotation_path in sorted(annotation_folder.glob("*.csv")):
        matches = re.findall(r"IDN(\d+).csv", str(annotation_path.name))
        p_id = matches[0]
        data_path = data_folder.joinpath(f"IDN{p_id}.txt")
        data = _load_txt_data(data_path)
        fs = 2000

        data.index /= fs
        data.index.name = "t"

        heartbeat_algo = HeartbeatSegmentationNeurokit()
        heartbeat_algo.extract(ecg=data[["ecg"]], sampling_rate_hz=fs)
        heartbeats = heartbeat_algo.heartbeat_list_

        heartbeats = heartbeats.round(2)

        heartbeat_path = heartbeat_folder.joinpath(f"IDN{p_id}.csv")
        heartbeats.to_csv(heartbeat_path)
