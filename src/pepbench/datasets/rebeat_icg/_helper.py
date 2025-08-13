import re
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
from biopsykit.signals.ecg.segmentation import HeartbeatSegmentationNeurokit
from scipy.io import loadmat

from pepbench.utils._types import path_t


def _load_mat_data(file_path: path_t) -> dict[str, np.ndarray]:
    return loadmat(file_path, squeeze_me=True)


def generate_labeling_and_heartbeat_borders(base_path: path_t) -> None:
    data_folder = base_path.joinpath("01_RawData")
    annotation_folder = base_path.joinpath("03_ExpertAnnotations")
    labeling_border_folder = base_path.joinpath("04_LabelingAndHeartBeatBorders")
    labeling_border_folder.mkdir(parents=True, exist_ok=True)
    for annotation_path in sorted(annotation_folder.glob("*.mat")):
        matches = re.findall(r"Annotat_Subject_(\d+)_task_(\w+).mat", str(annotation_path.name))
        p_id = matches[0][0]
        phase = matches[0][1]
        b_points = _load_b_point_annotations(annotation_path)

        data_path = data_folder.joinpath(f"RawData_Subject_{int(p_id)}_task_{phase}.mat")
        data = _load_mat_data(data_path)
        fs = data["samplFreq"]

        data = pd.DataFrame({"ecg": data["ECG"], "icg_der": data["ICG"]})
        data.index /= fs
        data.index.name = "t"

        heartbeat_algo = HeartbeatSegmentationNeurokit()
        heartbeat_algo.extract(ecg=data[["ecg"]], sampling_rate_hz=fs)
        heartbeats = heartbeat_algo.heartbeat_list_

        warnings.simplefilter("ignore", FutureWarning)
        b_point_groups = np.split(b_points, np.where(b_points.diff() > 5 * fs)[0])
        b_point_groups = pd.concat(dict(enumerate(b_point_groups)))
        b_point_groups.index = b_point_groups.index.set_names("label_region", level=0)
        b_point_groups = b_point_groups.reset_index()

        heartbeat_ids = []
        for _i, b_point in b_point_groups.iterrows():
            idx = np.where(
                (heartbeats["start_sample"] <= b_point["sample_relative"])
                & (heartbeats["end_sample"] >= b_point["sample_relative"])
            )[0]
            if len(idx) != 0:
                heartbeat_ids.append(idx[0])
        heartbeat_ids = pd.Series(heartbeat_ids)

        heartbeats = heartbeats.loc[heartbeat_ids]
        heartbeats.index = b_point_groups.index
        heartbeats.index -= heartbeats.index[0]
        heartbeats = heartbeats.assign(label_region=b_point_groups["label_region"].astype(int))
        heartbeats.index.name = "heartbeat_id"

        labeling_borders = heartbeats.groupby("label_region").apply(
            lambda s, sampling_rate=fs: {
                "start_sample": s.iloc[0]["start_sample"] - int(0.5 * sampling_rate),
                "end_sample": s.iloc[-1]["end_sample"] + int(0.5 * sampling_rate),
            },
        )
        labeling_borders = labeling_borders.apply(pd.Series).astype("Int64")

        for label_id, df in heartbeats.groupby("label_region"):
            df_out = df.drop(columns=["label_region"])
            heartbeat_path = labeling_border_folder.joinpath(
                f"Heartbeats_Subject_{int(p_id)}_task_{phase}_label_{label_id}.csv"
            )
            df_out = df_out.astype(
                {
                    "start_sample": "Int64",
                    "end_sample": "Int64",
                    "r_peak_sample": "Int64",
                    "rr_interval_sample": "Int64",
                }
            )
            df_out.to_csv(heartbeat_path)
            labeling_borders_path = labeling_border_folder.joinpath(
                f"LabelingBorders_Subject_{int(p_id)}_task_{phase}_label_{label_id}.csv"
            )
            labeling_borders.iloc[[label_id]].to_csv(labeling_borders_path, index=False)


def _load_b_point_annotations(file_path: path_t, load_func: Callable | None = None) -> pd.DataFrame:
    if load_func is None:
        load_func = _load_mat_data
    annotations = load_func(file_path)
    annotations = pd.DataFrame(annotations["annotPoints"], columns=["B", "C", "X", "Other"])
    annotations = annotations[["B"]].sort_values(by="B").reset_index(drop=True)
    annotations.columns = ["sample_relative"]
    annotations.index.name = "heartbeat_id"
    return annotations
