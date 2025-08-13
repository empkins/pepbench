import pandas as pd
from tqdm.auto import tqdm

from pepbench.datasets import BasePepDatasetWithAnnotations
from pepbench.heartbeat_matching import match_heartbeat_lists

__all__ = [
    "compute_annotation_differences",
    "load_annotations_from_dataset",
    "match_annotations",
    "normalize_annotations_to_heartbeat_start",
]


def load_annotations_from_dataset(
    dataset_01: BasePepDatasetWithAnnotations, dataset_02: BasePepDatasetWithAnnotations
) -> pd.DataFrame:
    labels_ecg_dict = {}
    labels_icg_dict = {}
    for subset_01, subset_02 in tqdm(list(zip(dataset_01.groupby(None), dataset_02.groupby(None), strict=False))):
        labels_ecg = match_annotations(
            subset_01.reference_labels_ecg, subset_02.reference_labels_ecg, dataset_01.sampling_rate_ecg
        )
        labels_icg = match_annotations(
            subset_01.reference_labels_icg, subset_02.reference_labels_icg, dataset_02.sampling_rate_ecg
        )

        labels_ecg_dict[subset_01.group_label] = labels_ecg
        labels_icg_dict[subset_01.group_label] = labels_icg

    labels_ecg_total = pd.concat(labels_ecg_dict, names=dataset_01.group_labels[0]._fields)
    labels_icg_total = pd.concat(labels_icg_dict, names=dataset_01.group_labels[0]._fields)

    labels_ecg_total = labels_ecg_total.xs("sample_relative", level="sample", axis=1)
    labels_icg_total = labels_icg_total.xs("sample_relative", level="sample", axis=1)

    return pd.concat({"ECG": labels_ecg_total, "ICG": labels_icg_total}, names=["signal"])


def match_annotations(
    annotations_01: pd.DataFrame, annotations_02: pd.DataFrame, sampling_rate_hz: float
) -> pd.DataFrame:
    heartbeats_01 = annotations_01.unstack("label")["sample_relative"][["start", "end"]].dropna()
    # heartbeats_01 = annotations_01.reindex(["start", "end"], level="label")["sample_relative"].unstack()
    heartbeats_01 = heartbeats_01.droplevel(-1)
    heartbeats_01.columns = ["start_sample", "end_sample"]

    heartbeats_02 = annotations_02.unstack("label")["sample_relative"][["start", "end"]].dropna()
    # heartbeats_02 = annotations_02.reindex(["start", "end"], level="label")["sample_relative"].unstack()
    heartbeats_02 = heartbeats_02.droplevel(-1)
    heartbeats_02.columns = ["start_sample", "end_sample"]

    heartbeat_matching = match_heartbeat_lists(
        heartbeats_reference=heartbeats_01,
        heartbeats_extracted=heartbeats_02,
        tolerance_ms=100,
        sampling_rate_hz=sampling_rate_hz,
    )

    tp_matches = heartbeat_matching.query("match_type == 'tp'")

    annotations_01_tp = annotations_01.loc[tp_matches["heartbeat_id_reference"]]
    annotations_02_tp = annotations_02.loc[tp_matches["heartbeat_id"]]
    annotations_01_tp = annotations_01_tp.reset_index(["channel", "label"])
    annotations_02_tp = annotations_02_tp.reset_index(["channel", "label"])

    annotations_02_tp.index = annotations_01_tp.index

    annotations_01_tp_new = annotations_01_tp.set_index(["channel", "label"], append=True)
    annotations_02_tp_new = annotations_02_tp.set_index(["channel", "label"], append=True)
    annotations_01_tp_new = annotations_01_tp_new.dropna()
    annotations_02_tp_new = annotations_02_tp_new.dropna()

    annotations = pd.concat(
        {"rater_01": annotations_01_tp_new, "rater_02": annotations_02_tp_new}, names=["rater", "sample"], axis=1
    )

    return annotations


def compute_annotation_differences(annotations: pd.DataFrame, sampling_rate_hz: float | None = None) -> pd.DataFrame:
    """Compute the difference in samples between the two raters."""
    if annotations.columns.nlevels == 1:
        annotations = annotations["rater_01"] - annotations["rater_02"]
    else:
        annotations = annotations["rater_01"]["sample_relative"] - annotations["rater_02"]["sample_relative"]

    if sampling_rate_hz is not None:
        annotations = annotations / sampling_rate_hz * 1000
        annotations = annotations.to_frame("difference_ms")
    else:
        annotations = annotations.to_frame("difference_samples")
    annotations = annotations.drop("heartbeat", level="channel")
    annotations = annotations.drop("Artefact", level="label")
    annotations = annotations.droplevel(["label", "channel"])

    return annotations


def normalize_annotations_to_heartbeat_start(
    annotations: pd.DataFrame, sampling_rate_hz: float | None = None
) -> pd.DataFrame:
    """Normalize the annotations to the start of the heartbeat."""
    annotations = annotations.drop("end", level="label").drop("Artefact", level="label")
    annotations = annotations.droplevel("channel").unstack()
    annotations = annotations.T.groupby("rater").diff().dropna(how="all")
    annotations = annotations.droplevel("label")
    annotations = annotations.T
    annotations = annotations.stack()
    if sampling_rate_hz is not None:
        annotations = annotations / sampling_rate_hz * 1000
        annotations = annotations.to_frame("difference_ms")
    else:
        annotations = annotations.to_frame("difference_samples")

    return annotations
