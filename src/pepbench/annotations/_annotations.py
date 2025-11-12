# Module for handling and comparing annotations from different raters or datasets.

"""Utilities for loading, matching and comparing annotations.

This module provides functions to load annotations from two datasets, match
heartbeats between two annotation sets, compute pointwise annotation
differences and normalize annotations relative to heartbeat start.

Functions
---------
load_annotations_from_dataset
    Load and align annotations from two datasets per-signal (ECG, ICG).
match_annotations
    Match annotations between two raters/datasets and return paired annotations.
compute_annotation_differences
    Compute pointwise differences between two raters' annotations.
normalize_annotations_to_heartbeat_start
    Normalize annotation times to the heartbeat start for each rater.
"""
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
    """Load and align annotations from two datasets.

    This function iterates over matching subsets (groups) of the two datasets,
    matches annotations per-signal (ECG and ICG) using ``match_annotations``,
    and concatenates the results into a single DataFrame with a top-level column
    index ``signal`` containing ``ECG`` and ``ICG``.

    Parameters
    ----------
    dataset_01 : BasePepDatasetWithAnnotations
        Reference dataset providing ``reference_labels_ecg`` and
        ``reference_labels_icg``, and ``group_labels``.
    dataset_02 : BasePepDatasetWithAnnotations
        Dataset to compare against ``dataset_01``.

    Returns
    -------
    pandas.DataFrame
        Concatenated DataFrame of matched annotations for ECG and ICG with a
        MultiIndex column where the top level is ``signal`` and lower levels
        represent ``rater`` and ``sample`` (as produced by ``match_annotations``).

    Notes
    -----
    Iteration uses ``dataset.groupby(None)`` and expects group labels to be
    compatible between datasets.
    """
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
    """Match annotations between two raters/datasets and return paired annotations.

        Extract heartbeat start/end sample indices from each annotation DataFrame,
        use ``match_heartbeat_lists`` to find matching heartbeats (true positives),
        and return the matched annotations aligned by heartbeat.

        Parameters
        ----------
        annotations_01 : pandas.DataFrame
            Annotation table for reference rater/dataset. Must contain a
            ``sample_relative`` column and a ``label`` level with ``start`` and ``end``.
        annotations_02 : pandas.DataFrame
            Annotation table for the other rater/dataset (same structure as
            ``annotations_01``).
        sampling_rate_hz : float
            Sampling rate used when matching heartbeats (passed to
            ``match_heartbeat_lists``).

        Returns
        -------
        pandas.DataFrame
            Concatenated annotations for matched heartbeats with MultiIndex columns
            ``rater`` (``rater_01`` and ``rater_02``) and ``sample`` (annotation fields).
            Only matched (true positive) heartbeats are included.

        Raises
        ------
        KeyError, IndexError
            If expected index/column levels (``label``, ``sample_relative``, ``channel``)
            are missing.
        """
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
    """Compute pointwise differences between two raters' annotations.

    For matched annotations (as produced by ``match_annotations``) this function
    computes the difference between ``rater_01`` and ``rater_02`` for the
    ``sample_relative`` values (or direct values if columns are single-leveled).
    The result is returned as a single-column DataFrame containing either
    ``difference_ms`` (if ``sampling_rate_hz`` is provided) or
    ``difference_samples``.

    Parameters
    ----------
    annotations : pandas.DataFrame
        Annotations DataFrame with rater columns (``rater_01``, ``rater_02``) and
        sample values under ``sample_relative`` or as single-level columns.
    sampling_rate_hz : float or None
        If provided, convert sample differences to milliseconds using this rate.

    Returns
    -------
    pandas.DataFrame
        Single-column DataFrame with the computed differences. Index preserves
        heartbeat/sample identifiers after dropping label/channel where appropriate.

    Notes
    -----
    The function drops entries labeled ``Artefact`` and removes ``label`` and
    ``channel`` levels from the index before returning.
    """
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
    """Normalize annotation times to the heartbeat start for each rater.

    Remove ``end`` and ``Artefact`` labels, collapse the ``channel`` level,
    unstack so that ``sample`` becomes columns, compute per-rater differences
    relative to the first annotation (``groupby('rater').diff()``), and return
    the differences either in samples or milliseconds.

    Parameters
    ----------
    annotations : pandas.DataFrame
        Annotation DataFrame with ``label`` and ``channel`` levels and rater columns.
    sampling_rate_hz : float or None
        If provided, convert sample-based differences to milliseconds.

    Returns
    -------
    pandas.DataFrame
        DataFrame of differences normalized to heartbeat start, with a single
        column named ``difference_ms`` or ``difference_samples`` depending on input.

    Notes
    -----
    Rows that become all-NaN after differencing are dropped.
    """
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
