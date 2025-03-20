import ast

import pandas as pd

from pepbench.datasets import BasePepDatasetWithAnnotations
from pepbench.utils._types import path_t

__all__ = ["load_labeling_borders", "compute_reference_heartbeats", "compute_reference_pep"]


def load_labeling_borders(file_path: path_t) -> pd.DataFrame:
    """Load the labeling borders from a csv file.

    Parameters
    ----------
    file_path : :class:`pathlib.Path` or str
        The path to the csv file.

    Returns
    -------
    :class:`pandas.DataFrame`
        The labeling borders.

    """
    data = pd.read_csv(file_path)
    data = data.assign(description=data["description"].apply(lambda s: ast.literal_eval(s)))

    data = data.set_index("timestamp").sort_index()
    return data


def compute_reference_heartbeats(heartbeats: pd.DataFrame) -> pd.DataFrame:
    """Reformat the heartbeats DataFrame.

    Parameters
    ----------
    heartbeats : :class:`pandas.DataFrame`
        DataFrame containing the heartbeats.

    Returns
    -------
    :class:`pandas.DataFrame`
        DataFrame containing the reformatted heartbeats.

    """
    heartbeats = heartbeats.droplevel("channel")["sample_relative"].unstack("label")
    heartbeats.columns = [f"{col}_sample" for col in heartbeats.columns]
    return heartbeats


def _fill_unlabeled_artefacts(
    points: pd.DataFrame,
    reference_data: pd.DataFrame,
    heartbeats: pd.DataFrame,  # noqa: ARG001
) -> pd.DataFrame:
    # get the indices of reference_icg that are not in b_points.index => they are artefacts but were not labeled
    heartbeat_ids = reference_data.index.get_level_values("heartbeat_id").unique()
    # insert "Artefact" label for artefacts that were not labeled to b_points,
    # set the sample to the middle of the heartbeat
    artefact_ids = list(heartbeat_ids.difference(points.droplevel("channel").index))
    for artefact_id in artefact_ids:
        start_abs, end_abs = reference_data.xs(artefact_id, level="heartbeat_id")["sample_absolute"]
        start_rel, end_rel = reference_data.xs(artefact_id, level="heartbeat_id")["sample_relative"]
        points.loc[(artefact_id, "Artefact"), :] = (int((start_abs + end_abs) / 2), int((start_rel + end_rel) / 2))

    points = points.sort_index()
    return points


def compute_reference_pep(subset: BasePepDatasetWithAnnotations) -> pd.DataFrame:
    """Compute the reference PEP values between the reference Q-peak and B-point labels.

    Parameters
    ----------
    subset : :class:`pepbench.datasets.BasePepDatasetWithAnnotations`
        Subset of a dataset containing the reference labels.

    Returns
    -------
    :class:`pandas.DataFrame`
        DataFrame containing the computed PEP values.

    """
    heartbeats = subset.reference_heartbeats
    reference_icg = subset.reference_labels_icg
    reference_ecg = subset.reference_labels_ecg

    b_points = reference_icg.reindex(["ICG", "Artefact"], level="channel").droplevel("label")
    b_points = _fill_unlabeled_artefacts(b_points, reference_icg, heartbeats)
    b_point_artefacts = b_points.reindex(["Artefact"], level="channel").droplevel("channel")
    b_points = b_points.reindex(["ICG"], level="channel").droplevel("channel")

    q_peaks = reference_ecg.reindex(["ECG", "Artefact"], level="channel").droplevel("label")
    q_peaks = _fill_unlabeled_artefacts(q_peaks, reference_ecg, heartbeats)
    q_peak_artefacts = q_peaks.reindex(["Artefact"], level="channel").droplevel("channel")
    q_peaks = q_peaks.reindex(["ECG"], level="channel").droplevel("channel")

    pep_reference = heartbeats.copy()
    pep_reference.columns = [
        f"heartbeat_{col}" if col != "r_peak_sample" else "r_peak_sample" for col in heartbeats.columns
    ]

    pep_reference = pep_reference.assign(
        q_peak_sample=q_peaks["sample_relative"],
        b_point_sample=b_points["sample_relative"],
        nan_reason=pd.NA,
    )
    # fill nan_reason column with artefact information
    pep_reference.loc[b_point_artefacts.index, "nan_reason"] = "icg_artefact"
    pep_reference.loc[q_peak_artefacts.index, "nan_reason"] = "ecg_artefact"

    pep_reference = pep_reference.assign(pep_sample=pep_reference["b_point_sample"] - pep_reference["q_peak_sample"])
    pep_reference = pep_reference.assign(pep_ms=pep_reference["pep_sample"] / subset.sampling_rate_ecg * 1000)

    # reorder columns
    pep_reference = pep_reference[
        [
            "heartbeat_start_sample",
            "heartbeat_end_sample",
            "q_peak_sample",
            "b_point_sample",
            "pep_sample",
            "pep_ms",
            "nan_reason",
        ]
    ]

    return pep_reference.convert_dtypes(infer_objects=True)
