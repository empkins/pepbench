from collections.abc import Sequence

import numpy as np
import pandas as pd
import pingouin as pg

from pepbench.utils._types import str_t

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

    """
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

    """
    algo_levels = [s for s in results_per_sample.index.names if s in _algo_levels]
    if isinstance(algo_combi, str):
        algo_combi = (algo_combi,)
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

    """
    pep = get_data_for_algo(results_per_sample, algo_combi)
    pep = pep[[("pep_ms", "estimated")]].droplevel(level=-1, axis=1)

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
    if group_cols is None:
        group_cols = ["phase"]
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
    results_per_sample = results_per_sample[_pep_error_metric_map.keys()].droplevel(level=-1, axis=1)
    results_per_sample = results_per_sample.groupby(algo_levels)
    results_per_sample = results_per_sample.agg(metrics)

    num_heartbeats = num_heartbeats.unstack().swaplevel(axis=1)
    results_per_sample = results_per_sample.join(num_heartbeats)

    if sortby is not None:
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

    """
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

    """
    algo_levels = [s for s in results_per_sample.index.names if s in _algo_levels]
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

    """
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

    """
    data = data.copy()
    if "outlier_correction_algorithm" not in data.columns.names:
        data = data.unstack("outlier_correction_algorithm")
    data = data.reindex(outlier_algos, axis=1, level="outlier_correction_algorithm")
    data = data.diff(axis=1).dropna(how="all", axis=1)
    data = np.sign(data)
    data.columns = ["improvement_percent"]
    # negative values = improvement through outlier correction
    # count the number of positive and negative values
    improvement_percent = data.value_counts(normalize=True) * 100
    improvement_percent = improvement_percent.rename({-1: "improvement", 1: "deterioration", 0: "no change"})
    improvement_percent = improvement_percent.to_frame().reindex(["improvement", "no change", "deterioration"], level=0)
    improvement_percent.columns = ["improvement_percent"]
    improvement_percent.index.names = [None]
    improvement_percent = improvement_percent.T
    return improvement_percent


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

    """
    data = data.copy()
    pipelines = ["_".join(i) for i in pipelines]

    data = data.unstack("pipeline").reindex(pipelines, level="pipeline", axis=1)
    # compute the percentage of sample which have a positive value in the first column
    # and a negative value in the second column
    data = data.assign(change_pos_neg=(data.iloc[:, 0] > 0) & (data.iloc[:, 1] < 0))
    data = data.assign(change_pos_pos=(data.iloc[:, 0] > 0) & (data.iloc[:, 1] > 0))
    data = data.assign(change_neg_pos=(data.iloc[:, 0] < 0) & (data.iloc[:, 1] > 0))
    data = data.assign(change_neg_neg=(data.iloc[:, 0] < 0) & (data.iloc[:, 1] < 0))
    data = data.assign(change_diff=data["change_pos_neg"] | data["change_neg_pos"])
    data = data.assign(change_same=data["change_pos_pos"] | data["change_neg_neg"])

    data = data.filter(like="change", axis=1)
    data = data.apply(pd.Series.value_counts, normalize=True) * 100

    return data


def merge_result_metrics_from_multiple_annotators(
    results: Sequence[pd.DataFrame], add_annotation_difference: bool = True
) -> pd.DataFrame:
    metrics_combined = pd.concat({f"Annotator {i + 1}": result_df for i, result_df in enumerate(results)}, axis=1)
    metrics_combined = metrics_combined.reindex(["Mean Absolute Error [ms]", "Mean Error [ms]"], level=1, axis=1)

    if add_annotation_difference:
        if len(results) != 2:
            raise ValueError("Annotation difference can only be computed for two annotators.")

        metrics_combined_diff = (
            metrics_combined.reindex(["Mean Absolute Error [ms]"], level=1, axis=1)
            .T.groupby(level=[1, 2])
            .diff()
            .dropna(how="all")
        )
        metrics_combined_diff = metrics_combined_diff.rename({"Annotator 2": "Annotator Difference"}).T

        metrics_combined = pd.concat([metrics_combined, metrics_combined_diff], axis=1)

    return metrics_combined


def merge_results_per_sample_from_different_annotators(
    results: Sequence[pd.DataFrame], selected_algorithm: tuple[str] | None = None
) -> pd.DataFrame:
    if selected_algorithm is None:
        results_combined = pd.concat({f"Annotator {i + 1}": result_df for i, result_df in enumerate(results)}, axis=1)
    else:
        results_combined = pd.concat(
            {f"Annotator {i + 1}": result_df.loc[selected_algorithm] for i, result_df in enumerate(results)}, axis=1
        )

    results_combined = results_combined.sort_index()

    return results_combined
