r"""Module for handling PEP benchmark data.

Provides utility functions to extract reference PEP values, compute performance metrics,
aggregate and compare results across algorithms and annotators, and compute derived
signals used throughout the PEP benchmark evaluation pipeline.

Functions
---------
:func:`~pepbench.data_handling._data_handling.get_reference_data`
    Extract the reference columns from a results-per-sample :class:`~pandas.DataFrame`.
:func:`~pepbench.data_handling._data_handling.get_reference_pep`
    Return the ``pep_ms`` reference column from results.
:func:`~pepbench.data_handling._data_handling.get_data_for_algo`
    Select data for a specific algorithm combination from a results frame.
:func:`~pepbench.data_handling._data_handling.get_pep_for_algo`
    Select estimated PEP values for a specified algorithm combination.
:func:`~pepbench.data_handling._data_handling.describe_pep_values`
    Compute descriptive statistics for PEP values using :meth:`pandas.DataFrame.describe`.
:func:`~pepbench.data_handling._data_handling.compute_pep_performance_metrics`
    Aggregate performance metrics (e.g., mean, std) across algorithm combinations.
:func:`~pepbench.data_handling._data_handling.get_performance_metric`
    Extract a single performance metric column from a results frame.
:func:`~pepbench.data_handling._data_handling.rr_interval_to_heart_rate`
    Convert ``rr_interval_ms`` to ``heart_rate_bpm`` and join to the input frame.
:func:`~pepbench.data_handling._data_handling.correlation_reference_pep_heart_rate`
    Compute linear regression and Pearson correlation between reference PEP and heart rate.
:func:`~pepbench.data_handling._data_handling.get_error_by_group`
    Aggregate error metrics by specified grouping columns (mean and std).
:func:`~pepbench.data_handling._data_handling.add_unique_id_to_results_dataframe`
    Add a unique concatenated identifier to the results index for sample-level merging.
:func:`~pepbench.data_handling._data_handling.compute_improvement_outlier_correction`
    Compute the percentage of samples that improved, deteriorated, or remained unchanged after outlier correction.
:func:`~pepbench.data_handling._data_handling.compute_improvement_pipeline`
    Compute percentage of samples showing sign changes in the error metric between two pipelines.
:func:`~pepbench.data_handling._data_handling.merge_result_metrics_from_multiple_annotators`
    Combine metric tables from multiple annotators and optionally compute annotation differences.
:func:`~pepbench.data_handling._data_handling.merge_results_per_sample_from_different_annotators`
    Concatenate per-sample results from multiple annotators into a single frame.

Notes
-----
- Many functions expect or return :class:`~pandas.DataFrame` objects using the project\'
  conventions: MultiIndex columns/indices with algorithm-level names
  ``q_peak_algorithm``, ``b_point_algorithm``, and ``outlier_correction_algorithm``.
- Columns commonly referenced include ``pep_ms``, ``absolute_error_per_sample_ms``,
  and ``rr_interval_ms``; heartbeat/algorithm indexing is assumed for per-sample frames.
- Correlation and regression helpers rely on :mod:`pingouin` for statistics.
- Functions preserve dtypes and index names where possible to facilitate downstream
  grouping and joining operations.

See Also
--------
:mod:`pepbench.pipelines`
    Pipeline helpers and end-to-end execution utilities for PEP extraction.
"""
import contextlib
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pingouin as pg

from pepbench.utils._types import check_data_is_df, check_data_is_str_t, str_t

__all__ = [
    "add_unique_id_to_results_dataframe",
    "compute_improvement_outlier_correction",
    "compute_improvement_pipeline",
    "compute_pep_performance_metrics",
    "correlation_reference_pep_heart_rate",
    "describe_pep_values",
    "get_data_for_algo",
    "get_error_by_group",
    "get_pep_for_algo",
    "get_reference_data",
    "get_reference_pep",
    "merge_result_metrics_from_multiple_annotators",
    "merge_results_per_sample_from_different_annotators",
    "rr_interval_to_heart_rate",
]


_algo_levels = ["q_peak_algorithm", "b_point_algorithm", "outlier_correction_algorithm"]

_pep_error_metric_map = {
    "absolute_error_per_sample_ms": "Mean Absolute Error [ms]",
    "error_per_sample_ms": "Mean Error [ms]",
    "absolute_relative_error_per_sample_percent": "Mean Absolute Relative Error [%]",
}
_pep_number_map = {
    "num_pep_valid": "Valid PEPs",
    "num_pep_invalid": "Invalid PEPs",
    "num_pep_total": "Total PEPs",
}


def get_reference_data(results_per_sample: pd.DataFrame) -> pd.DataFrame:
    """Extract the reference data from the *results-per-sample* dataframe.

    Parameters
    ----------
    results_per_sample : :class:`pandas.DataFrame`
        The results-per-sample dataframe.

    Returns
    -------
    :class:`pandas.DataFrame`
        The reference data.

    Raises
    ------
    TypeError
        If the input data is not a :class:`pandas.DataFrame`.

    """
    check_data_is_df(results_per_sample)
    reference_pep = results_per_sample.xs("reference", level=-1, axis=1)
    reference_pep = reference_pep.groupby(_algo_levels)
    reference_pep = reference_pep.get_group(next(iter(reference_pep.groups))).droplevel(_algo_levels)

    return reference_pep


def get_reference_pep(results_per_sample: pd.DataFrame) -> pd.DataFrame:
    """Extract the reference PEP values from the *results-per-sample* dataframe.

    Parameters
    ----------
    results_per_sample : :class:`pandas.DataFrame`
        The results-per-sample dataframe.

    Returns
    -------
    :class:`pandas.DataFrame`
        The reference PEP values.

    """
    check_data_is_df(results_per_sample)
    return get_reference_data(results_per_sample)[["pep_ms"]]


def get_data_for_algo(results_per_sample: pd.DataFrame, algo_combi: str_t) -> pd.DataFrame:
    """Extract the data for a specific algorithm combination from the *results-per-sample* dataframe.

    Parameters
    ----------
    results_per_sample : :class:`pandas.DataFrame`
        The results-per-sample dataframe.
    algo_combi : str or tuple of str
        The algorithm combination for which the data should be extracted.

    Returns
    -------
    :class:`pandas.DataFrame`
        The data for the specified algorithm combination.

    Raises
    ------
    TypeError
        If the input data is not a :class:`pandas.DataFrame`.

    """
    check_data_is_df(results_per_sample)
    algo_levels = [s for s in results_per_sample.index.names if s in _algo_levels]

    # Validate algo_combi: should be a sequence of strings (tuple/list) matching the number of algo levels
    if isinstance(algo_combi, str) or not isinstance(algo_combi, Sequence):
        from pepbench.utils.exceptions import ValidationError

        raise ValidationError("algo_combi must be a tuple/list of algorithm level names")

    if len(algo_combi) != len(algo_levels):
        from pepbench.utils.exceptions import ValidationError

        raise ValidationError(f"algo_combi must have length {len(algo_levels)} matching algorithm levels")

    data = results_per_sample.xs(tuple(algo_combi), level=algo_levels)
    return data


def get_pep_for_algo(results_per_sample: pd.DataFrame, algo_combi: Sequence[str]) -> pd.DataFrame:
    """Extract the PEP values for a specific algorithm combination from the *results-per-sample* dataframe.

    Parameters
    ----------
    results_per_sample : :class:`pandas.DataFrame`
        The results-per-sample dataframe.
    algo_combi : str or tuple of str
        The algorithm combination for which the PEP values should be extracted.

    Returns
    -------
    :class:`pandas.DataFrame`
        The PEP values for the specified algorithm combination.

    Raises
    ------
    TypeError
        If the input data is not a :class:`pandas.DataFrame`.

    """
    check_data_is_df(results_per_sample)
    # get_data_for_algo performs validation of algo_combi
    pep = get_data_for_algo(results_per_sample, algo_combi)
    pep = pep.loc[:, pep.columns.get_level_values(0) == "pep_ms"]
    # select estimated column if present
    if ("pep_ms", "estimated") in pep.columns:
        pep = pep[[("pep_ms", "estimated")]].droplevel(level=-1, axis=1)
    else:
        # fallback: if only single-level pep_ms column present, return it
        # suppress common errors from droplevel when columns are already single-level
        with contextlib.suppress(IndexError, KeyError, AttributeError, ValueError):
            pep = pep.droplevel(level=-1, axis=1)

    return pep


def describe_pep_values(
    data: pd.DataFrame, group_cols: str_t | None = None, metrics: Sequence[str] | None = None
) -> pd.DataFrame:
    """Compute the descriptive statistics for the PEP values using the :meth:`pandas.DataFrame.describe` method.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        The PEP values.
    group_cols : str or list of str, optional
        The column(s) to group the data by, if any. Default: "phase".
    metrics : list of str, optional
        List of metrics to display from the descriptive statistics. Default: ["mean", "std", "min", "max"].

    """
    check_data_is_df(data)
    if group_cols is None:
        group_cols = ["phase"]
    else:
        # validate provided group columns (str or sequence of str)
        check_data_is_str_t(group_cols)
    if metrics is None:
        metrics = ["mean", "std", "min", "max"]

    return data.groupby(group_cols).describe().reindex(metrics, level=-1, axis=1)


def compute_pep_performance_metrics(
    results_per_sample: pd.DataFrame,
    *,
    num_heartbeats: pd.DataFrame | None = None,
    metrics: Sequence[str] | None = None,
    sortby: str_t | None = ("absolute_error_per_sample_ms", "mean"),
    ascending: bool | None = True,
) -> pd.DataFrame:
    """Compute the performance metrics for the PEP values.

    Parameters
    ----------
    results_per_sample : :class:`pandas.DataFrame`
        The results-per-sample dataframe.
    num_heartbeats : :class:`pandas.DataFrame`, optional
        Dataframe containing the number of heartbeats (to include in the output). Default: None.
    metrics : list of str, optional
        List of metrics to compute. Default: ["mean", "std"].
    sortby : str, optional
        The column to sort the results by. Default: ("absolute_error_per_sample_ms", "mean").
    ascending : bool, optional
        Whether to sort the results in ascending order. Default: True.

    """
    if metrics is None:
        metrics = ["mean", "std"]
    results_per_sample = results_per_sample.copy()
    algo_levels = [s for s in results_per_sample.index.names if s in _algo_levels]
    # select only the error metric columns that actually exist in the frame
    present_metrics = [k for k in _pep_error_metric_map if k in results_per_sample.columns.get_level_values(0)]
    results_per_sample = (results_per_sample.loc[:, results_per_sample.columns.get_level_values(0).
                          isin(present_metrics)].droplevel(level=-1, axis=1))
    results_per_sample = results_per_sample.groupby(algo_levels)
    results_per_sample = results_per_sample.agg(metrics)

    num_heartbeats = num_heartbeats.unstack()
    # Ensure columns are MultiIndex so swaplevel works; if single-level, promote to MultiIndex using the column name
    if not isinstance(num_heartbeats.columns, pd.MultiIndex):
        # column names may be an Index of participants; use the Series/DataFrame name as top level if available
        # top_level_name = getattr(num_heartbeats, "columns", None)
        # If num_heartbeats came from a Series.unstack() it may be a Series; ensure DataFrame
        if isinstance(num_heartbeats, pd.Series):
            num_heartbeats = num_heartbeats.to_frame()
        # create a top level name
        top_name = num_heartbeats.columns.name if (getattr(num_heartbeats.columns, "name", None)
                                                   is not None) else "num_heartbeats"
        num_heartbeats.columns = pd.MultiIndex.from_product([[top_name], list(num_heartbeats.columns)])
    num_heartbeats = num_heartbeats.swaplevel(axis=1)
    results_per_sample = results_per_sample.join(num_heartbeats)

    if sortby is not None:
        # validate sortby: accepts a string or sequence of strings
        check_data_is_str_t(sortby)
        results_per_sample = results_per_sample.sort_values(sortby, ascending=ascending)

    rename_map = _pep_error_metric_map.copy()
    rename_map.update(_pep_number_map)
    results_per_sample = results_per_sample.rename(rename_map, level=0, axis=1)
    results_per_sample = results_per_sample.reindex(rename_map.values(), level=0, axis=1)

    return results_per_sample


def get_performance_metric(results_per_sample: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Extract a specific performance metric from the *results-per-sample* dataframe.

    Parameters
    ----------
    results_per_sample : :class:`pandas.DataFrame`
        The results-per-sample dataframe.
    metric : str

    Returns
    -------
    :class:`pandas.DataFrame`
        The extracted performance metric.

    """
    # validate metric name (string or sequence of strings)
    check_data_is_str_t(metric)
    return results_per_sample[[metric]].droplevel(level=-1, axis=1)


def rr_interval_to_heart_rate(data: pd.DataFrame) -> pd.DataFrame:
    """Convert RR intervals in milliseconds to heart rate in beats per minute.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        The data containing the RR intervals in milliseconds. The column name must be "rr_interval_ms".

    Returns
    -------
    :class:`pandas.DataFrame`
        The data with the heart rate in beats per minute

    Raises
    ------
    AssertionError
        If the input DataFrame does not contain the "rr_interval_ms" column.
    ValidationError
        If the input data is not a :class:`pandas.DataFrame`.

    """
    check_data_is_df(data)
    assert "rr_interval_ms" in data.columns, 'Input DataFrame must contain "rr_interval_ms" column.'
    heart_rate_bpm = 60 * 1000 / data[["rr_interval_ms"]]
    heart_rate_bpm = heart_rate_bpm.rename(columns={"rr_interval_ms": "heart_rate_bpm"})
    return data.join(heart_rate_bpm)


def correlation_reference_pep_heart_rate(data: pd.DataFrame, groupby: str | None = None) -> dict[str, pd.DataFrame]:
    """Compute the correlation between the reference PEP values and the heart rate.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        The data containing the reference PEP values and the heart rate.
    groupby : str, optional
        The column to group the data by. If None, no grouping is applied. Default: None.

    Returns
    -------
    dict
        A dictionary containing the linear regression model and the correlation coefficient.

    """
    data = get_reference_data(data)

    # compute a linear regression model
    if groupby is None:
        linreg = pg.linear_regression(X=data["heart_rate_bpm"], y=data["pep_ms"], remove_na=True)
        corr = pg.corr(data["heart_rate_bpm"], data["pep_ms"], method="pearson")
    else:
        linreg = data.groupby(groupby).apply(
            lambda df: pg.linear_regression(X=df["heart_rate_bpm"], y=df["pep_ms"], remove_na=True)
        )
        corr = data.groupby(groupby).apply(lambda df: pg.corr(df["heart_rate_bpm"], df["pep_ms"], method="pearson"))

    return {"linear_regression": linreg, "correlation": corr}


def get_error_by_group(
    results_per_sample: pd.DataFrame, error_metric: str = "absolute_error_per_sample_ms", grouper: str_t = "participant"
) -> pd.DataFrame:
    """Compute mean and standard deviation of the error metric by group.

    Parameters
    ----------
    results_per_sample : :class:`pandas.DataFrame`
        The results-per-sample dataframe.
    error_metric : str, optional
        The error metric to extract. Default: "absolute_error_per_sample_ms".
    grouper : str or list of str, optional
        The column(s) to group the data by. Default: "participant".

    Returns
    -------
    :class:`pandas.DataFrame`
        The error metric aggregated by group.

    Raises
    ------
    ValidationError
        If the grouper argument is not a string or a sequence of strings.

    """
    algo_levels = [s for s in results_per_sample.index.names if s in _algo_levels]
    # validate grouper argument (str or sequence of str)
    check_data_is_str_t(grouper)
    if isinstance(grouper, str):
        grouper = [grouper]

    error = results_per_sample[[error_metric]].groupby([*algo_levels, *grouper]).agg(["mean", "std"])
    error = (
        error.droplevel(1, axis=1)
        .unstack(algo_levels)
        .reorder_levels([0, *range(2, 2 + len(algo_levels)), 1], axis=1)
        .sort_index(axis=1)
    )
    error.columns = error.columns.set_names("metric", level=0)
    return error


def add_unique_id_to_results_dataframe(data: pd.DataFrame, algo_levels: Sequence[str] | None = None) -> pd.DataFrame:
    """Add a unique ID to the results dataframe.

    The unique ID is created by concatenating the values of the specified algorithm levels and the heartbeat IDs.
    This is then added as a new index level named "id_concat".

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        The results dataframe.
    algo_levels : list of str, optional
        The algorithm levels to use for the unique ID. If None, the default algorithm levels
        (["q_peak_algorithm", "b_point_algorithm", "outlier_correction_algorithm"]) are used.

    Returns
    -------
    :class:`pandas.DataFrame`
        The results dataframe with the unique IDs added as new index level.

    Raises
    ------
    ValidationError
        If the input data is not a :class:`pandas.DataFrame`.
    """
    check_data_is_df(data)
    data = data.copy()
    if data.columns.nlevels > 1:
        data = data.droplevel(axis=1, level=-1).rename(index=str)
    if algo_levels is None:
        algo_levels = _algo_levels
    if isinstance(algo_levels, str):
        algo_levels = [algo_levels]

    algo_levels = [s for s in data.index.names if s in algo_levels]
    data = data.reset_index(level=algo_levels)
    id_concat = pd.Index(["_".join([str(i) for i in idx]) for idx in data.index], name="id_concat")

    data = data.assign(id_concat=id_concat)
    data = data.set_index([*algo_levels, "id_concat"])
    return data


def compute_improvement_outlier_correction(data: pd.DataFrame, outlier_algos: Sequence[str]) -> pd.DataFrame:
    """Compute the percentage of samples which improved, deteriorated, or remained unchanged after outlier correction.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        The data containing the PEP values before and after outlier correction.
    outlier_algos : list of str
        The outlier correction algorithms to consider.

    Returns
    -------
    :class:`pandas.DataFrame`
        The percentage of samples which improved, deteriorated, or remained unchanged after outlier correction.

    Raises
    ------
    ValidationError
        If the input data is not a :class:`pandas.DataFrame`.
    """
    check_data_is_df(data)
    data = data.copy()

    # Ensure columns indexed by outlier algorithm are available; accept either MultiIndex or simple DataFrame
    if "outlier_correction_algorithm" in (data.columns.names or []):
        # select columns for the requested outlier algorithms
        # let any unexpected exceptions bubble up; explicitly handle common lookup errors
        left = data.xs(outlier_algos[0], level="outlier_correction_algorithm", axis=1)
        right = data.xs(outlier_algos[1], level="outlier_correction_algorithm", axis=1)
    else:
        # assume simple columns named by outlier_algos
        left = data[outlier_algos[0]]
        right = data[outlier_algos[1]]

    # compute difference (after - before), sign: negative -> improvement
    diff = right - left
    # if diff is DataFrame with multiple columns, reduce to a single series by summing/mean?
    # Here expect single metric column
    if isinstance(diff, pd.DataFrame):
        # if multiple columns, attempt to reduce to a single series by taking the first column
        diff = diff.iloc[:, 0]

    signs = np.sign(diff.dropna())
    counts = signs.value_counts(normalize=True) * 100
    counts = counts.rename({-1: "improvement", 0: "no change", 1: "deterioration"})
    # ensure all three categories present
    for k in ["improvement", "no change", "deterioration"]:
        if k not in counts.index:
            counts.loc[k] = 0.0

    # return as a single-row DataFrame with ordered indices
    counts = counts.reindex(["improvement", "no change", "deterioration"]) * 1.0
    out = counts.to_frame().T
    out.columns.name = None
    return out


def compute_improvement_pipeline(data: pd.DataFrame, pipelines: Sequence[str]) -> pd.DataFrame:
    """Compute the percentage of samples which showed sign changes in the error metric between two pipelines.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        The data containing the PEP extraction results from different pipelines.
    pipelines : list of str
        The pipelines to compare.

    Returns
    -------
    :class:`pandas.DataFrame`
        Overview of the percentage of samples which showed a change in the sign of the error metric
        (i.e., either positive to negative, vice versa, or no change) between two pipelines.

    Raises
    ------
    ValidationError
        If the input data is not a :class:`pandas.DataFrame` or :class:`pandas.Series`.
    ValueError
        If less than two pipelines are provided.

    """
    # Accept Series or DataFrame input
    if isinstance(data, pd.Series):
        # convert to DataFrame so unstack/pivot operations work consistently
        ser = data.copy()
        data = ser.to_frame(name="value")
    elif not isinstance(data, pd.DataFrame):
        from pepbench.utils.exceptions import ValidationError

        raise ValidationError(f"Expected data to be a pandas DataFrame or Series, got {type(data)} instead.")

    data = data.copy()
    pipelines = ["_".join(i) for i in pipelines]

    # After unstack, columns may be a simple Index of pipeline keys or a MultiIndex with a 'pipeline' level
    try:
        unstacked = data.unstack("pipeline")
    except (KeyError, ValueError):
        # If unstack by level name fails, assume index first level already contains pipeline keys and pivot accordingly
        unstacked = data.unstack(level=0)

    # If unstacked has MultiIndex columns with a 'pipeline' level, select by that, otherwise reindex simple columns
    if isinstance(unstacked.columns, pd.MultiIndex) and "pipeline" in (unstacked.columns.names or []):
        df = unstacked.reindex(pipelines, level="pipeline", axis=1)
        # collapse to simple DataFrame of values for comparison
        # take the first metric if multiple present
        df_values = df.iloc[:, :2] if isinstance(df.columns, pd.MultiIndex) else df
    else:
        # simple columns: ensure order matches pipelines
        df_values = unstacked.reindex(columns=pipelines)

    if df_values.shape[1] < 2:
        raise ValueError("Need at least two pipelines to compare")

    a = df_values.iloc[:, 0]
    b = df_values.iloc[:, 1]

    df_flags = pd.DataFrame(
        {
            "change_pos_neg": (a > 0) & (b < 0),
            "change_pos_pos": (a > 0) & (b > 0),
            "change_neg_pos": (a < 0) & (b > 0),
            "change_neg_neg": (a < 0) & (b < 0),
        }
    )
    df_flags["change_diff"] = df_flags["change_pos_neg"] | df_flags["change_neg_pos"]
    df_flags["change_same"] = df_flags["change_pos_pos"] | df_flags["change_neg_neg"]

    res = df_flags.apply(pd.Series.value_counts, normalize=True) * 100
    return res


def merge_result_metrics_from_multiple_annotators(
    results: Sequence[pd.DataFrame], add_annotation_difference: bool = True
) -> pd.DataFrame:
    """
    Merge result metrics from multiple annotators into a single dataframe.

    Parameters
    ----------
    results: list of :class:`pandas.DataFrame`
        List of result metrics dataframes from different annotators.
    add_annotation_difference: bool, optional
        Whether to compute and add the difference between annotators. Default is True.

    Returns
    -------
    :class:`pandas.DataFrame`
        Merged result metrics from multiple annotators.

    """
    metrics_combined = pd.concat({f"Annotator {i + 1}": result_df for i, result_df in enumerate(results)}, axis=1)
    metrics_combined = metrics_combined.reindex(["Mean Absolute Error [ms]", "Mean Error [ms]"], level=1, axis=1)

    if add_annotation_difference:
        if len(results) != 2:
            raise ValueError("Annotation difference can only be computed for two annotators.")

        # compute direct difference between annotator 2 and annotator 1
        a1 = results[0]
        a2 = results[1]
        # align indices and columns
        a2_aligned, a1_aligned = a2.align(a1, join="outer", axis=None)
        diff = a2_aligned - a1_aligned
        # wrap the diff with a top-level key 'Annotator Difference'
        diff_wrapped = pd.concat({"Annotator Difference": diff}, axis=1)

        metrics_combined = pd.concat([metrics_combined, diff_wrapped], axis=1)

    return metrics_combined


def merge_results_per_sample_from_different_annotators(
    results: Sequence[pd.DataFrame], selected_algorithm: tuple[str] | None = None
) -> pd.DataFrame:
    """
    Merge results-per-sample dataframes from different annotators into a single dataframe.

    Parameters
    ----------
    results: list of :class:`pandas.DataFrame`
        List of results-per-sample dataframes from different annotators.
    selected_algorithm: tuple of str, optional
        Specific algorithm combination to extract from each annotator's results. If None, all algorithms are

    Returns
    -------
    :class:`pandas.DataFrame`
        Merged results-per-sample from different annotators.

    """
    if selected_algorithm is None:
        results_combined = pd.concat({f"Annotator {i + 1}": result_df for i, result_df in enumerate(results)}, axis=1)
    else:
        results_combined = pd.concat(
            {f"Annotator {i + 1}": result_df.loc[selected_algorithm] for i, result_df in enumerate(results)}, axis=1
        )

    results_combined = results_combined.sort_index()

    return results_combined
