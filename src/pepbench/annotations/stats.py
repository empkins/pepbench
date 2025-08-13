"""Functions for analyzing and processing annotation differences.

The functions include utilities for computing descriptive statistics, binning annotation differences, calculating the
Intraclass Correlation Coefficient (ICC), and integrating annotation agreement information into results dataframes.

"""

__all__ = [
    "add_annotation_agreement_to_results_dataframe",
    "bin_annotation_differences",
    "compute_icc",
    "describe_annotation_differences",
]

from collections.abc import Sequence

import pandas as pd
import pingouin as pg

from pepbench.annotations._annotations import compute_annotation_differences, normalize_annotations_to_heartbeat_start
from pepbench.data_handling import add_unique_id_to_results_dataframe


def describe_annotation_differences(annotation_diffs: pd.DataFrame, include_absolute: bool = True) -> pd.DataFrame:
    """Generate descriptive statistics for annotation differences.

    This function computes descriptive statistics for the provided annotation differences dataframe.
    Optionally, it can include the absolute values of the differences as an additional column.

    Parameters
    ----------
    annotation_diffs : :class:`~pandas.DataFrame`
        A dataframe containing annotation differences with a column named "difference_ms".
    include_absolute : bool, optional
        If True, includes the absolute values of the differences in the descriptive statistics.
        Default is True.

    Returns
    -------
    :class:`~pandas.DataFrame`
        A transposed dataframe containing descriptive statistics for the annotation differences.
    """
    annotation_diffs_describe = annotation_diffs.copy()
    if include_absolute:
        annotation_diffs_describe = annotation_diffs_describe.assign(
            difference_ms_absolute=annotation_diffs_describe["difference_ms"].abs()
        )
    return annotation_diffs_describe.describe().T


def bin_annotation_differences(
    annotation_diffs: pd.DataFrame, bins: Sequence[int] | None = None, labels: Sequence[str] | None = None
) -> pd.DataFrame:
    """Bin annotation differences into specified categories.

    This function categorizes annotation differences into bins and assigns labels to each bin.
    If no bins are provided, default bins are used. The resulting bins are returned as a dataframe.

    Parameters
    ----------
    annotation_diffs : :class:`~pandas.DataFrame`
        A dataframe containing annotation differences to be binned.
    bins : list of int, optional
        A sequence of bin edges. If not provided, default bins [0, 4, 10] are used.
    labels : list of str, optional
        A sequence of labels corresponding to the bins. If not provided, no labels are assigned.

    Returns
    -------
    :class:`~pandas.DataFrame`
        A dataframe with a single column named "annotation_bins" containing the binned annotation differences.
    """
    if bins is None:
        bins = [0, 4, 10]
    annotation_bins = pd.cut(
        annotation_diffs.abs().squeeze(),
        bins=[*bins, annotation_diffs.max().squeeze()],
        include_lowest=True,
        labels=labels,
    )
    return annotation_bins.to_frame(name="annotation_bins")


def compute_icc(annotation_diffs: pd.DataFrame, sampling_rate_hz: float) -> pd.DataFrame:
    """Compute the Intraclass Correlation Coefficient (ICC) for annotation differences.

    This function normalizes annotation differences to the heartbeat start, adds unique IDs
    for each rater, and calculates the ICC using the Pingouin library.

    Parameters
    ----------
    annotation_diffs : :class:`~pandas.DataFrame`
        A dataframe containing annotation differences.
    sampling_rate_hz : float
        The sampling rate in Hertz used to normalize annotation differences.

    Returns
    -------
    :class:`~pandas.DataFrame`
        A dataframe containing the computed ICC values.
    """
    annotation_diffs_normalized = normalize_annotations_to_heartbeat_start(
        annotation_diffs, sampling_rate_hz=sampling_rate_hz
    )
    annotation_diffs_normalized = add_unique_id_to_results_dataframe(annotation_diffs_normalized, algo_levels=["rater"])

    return pg.intraclass_corr(
        data=annotation_diffs_normalized.reset_index(),
        targets="id_concat",
        ratings="difference_ms",
        raters="rater",
        nan_policy="omit",
    )


def add_annotation_agreement_to_results_dataframe(
    results_per_sample: pd.DataFrame, annotations: pd.DataFrame, sampling_rate_hz: float
) -> pd.DataFrame:
    """Add annotation agreement information to the results dataframe.

    This function computes annotation differences, bins them into categories (e.g., high, medium, low),
    and integrates this information into the provided results dataframe. The resulting dataframe
    includes an additional index level for annotation agreement bins.

    Parameters
    ----------
    results_per_sample : :class:`~pandas.DataFrame`
        A dataframe containing results for each sample.
    annotations : :class:`~pandas.DataFrame`
        A dataframe containing annotation data to compute differences.
    sampling_rate_hz : float
        The sampling rate in Hertz used to normalize annotation differences.

    Returns
    -------
    :class:`~pandas.DataFrame`
        A dataframe with the original results and added annotation agreement information.
    """
    annotation_diffs = compute_annotation_differences(annotations, sampling_rate_hz=sampling_rate_hz)
    annotation_bins = bin_annotation_differences(annotation_diffs, labels=["high", "medium", "low"])

    annotation_bins.index = annotation_bins.index.rename({"heartbeat_id": "id"})
    annotation_bins = pd.concat(
        {"Annotation Agreement": pd.concat({"annotation_agreement": annotation_bins}, axis=1)}, axis=1
    )

    results_per_sample = results_per_sample.join(annotation_bins)

    results_per_sample = results_per_sample.reindex(
        ["absolute_error_per_sample_ms", "annotation_agreement"], level=1, axis=1
    ).set_index(("Annotation Agreement", "annotation_agreement", "annotation_bins"), append=True)
    results_per_sample.index = results_per_sample.index.rename(
        {("Annotation Agreement", "annotation_agreement", "annotation_bins"): "agreement_bins"}
    )
    return results_per_sample
