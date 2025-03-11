from collections.abc import Sequence
from typing import Optional

import numpy as np
import pandas as pd
import pingouin as pg

from pepbench.utils._types import str_t

__all__ = [
    "get_error_by_group",
    "get_reference_data",
    "get_reference_pep",
    "get_pep_for_algo",
    "get_data_for_algo",
    "describe_pep_values",
    "compute_pep_performance_metrics",
    "rr_interval_to_heart_rate",
    "correlation_reference_pep_heart_rate",
    "add_unique_id_to_results_dataframe",
    "compute_improvement_outlier_correction",
    "compute_improvement_pipeline",
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
    reference_pep = results_per_sample.xs("reference", level=-1, axis=1)
    reference_pep = reference_pep.groupby(_algo_levels)
    reference_pep = reference_pep.get_group(next(iter(reference_pep.groups))).droplevel(_algo_levels)

    return reference_pep


def get_reference_pep(results_per_sample: pd.DataFrame) -> pd.DataFrame:
    return get_reference_data(results_per_sample)[["pep_ms"]]


def get_data_for_algo(results_per_sample: pd.DataFrame, algo_combi: str_t) -> pd.DataFrame:
    algo_levels = [s for s in results_per_sample.index.names if s in _algo_levels]
    if isinstance(algo_combi, str):
        algo_combi = (algo_combi,)
    data = results_per_sample.xs(tuple(algo_combi), level=algo_levels)
    return data


def get_pep_for_algo(results_per_sample: pd.DataFrame, algo_combi: Sequence[str]) -> pd.DataFrame:
    pep = get_data_for_algo(results_per_sample, algo_combi)
    pep = pep[[("pep_ms", "estimated")]].droplevel(level=-1, axis=1)

    return pep


def describe_pep_values(
    data: pd.DataFrame, group_cols: Optional[str_t] = None, metrics: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    if group_cols is None:
        group_cols = ["phase"]
    if metrics is None:
        metrics = ["mean", "std", "min", "max"]

    return data.groupby(group_cols).describe().reindex(metrics, level=-1, axis=1)


def compute_pep_performance_metrics(
    results_per_sample: pd.DataFrame,
    *,
    num_heartbeats: Optional[pd.DataFrame] = None,
    metrics: Optional[Sequence[str]] = None,
    sortby: Optional[str_t] = ("absolute_error_per_sample_ms", "mean"),
    ascending: Optional[bool] = True,
) -> pd.DataFrame:
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
    return results_per_sample[[metric]].droplevel(level=-1, axis=1)


def rr_interval_to_heart_rate(data: pd.DataFrame) -> pd.DataFrame:
    heart_rate_bpm = 60 * 1000 / data[["rr_interval_ms"]]
    heart_rate_bpm = heart_rate_bpm.rename(columns={"rr_interval_ms": "heart_rate_bpm"})
    return data.join(heart_rate_bpm)


def correlation_reference_pep_heart_rate(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    data = get_reference_data(data)

    # compute a linear regression model
    linreg = pg.linear_regression(X=data["heart_rate_bpm"], y=data["pep_ms"], remove_na=True)
    corr = pg.corr(data["heart_rate_bpm"], data["pep_ms"], method="pearson")

    return {"linear_regression": linreg, "correlation": corr}


def get_error_by_group(
    results_per_sample: pd.DataFrame, error_metric: str = "absolute_error_per_sample_ms", grouper: str_t = "participant"
) -> pd.DataFrame:
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


def add_unique_id_to_results_dataframe(data: pd.DataFrame, algo_levels: Optional[Sequence[str]] = None) -> pd.DataFrame:
    data = data.copy()
    data = data.droplevel(axis=1, level=-1).rename(index=str)
    if algo_levels is None:
        algo_levels = _algo_levels
    if isinstance(algo_levels, str):
        algo_levels = [algo_levels]

    algo_levels = [s for s in data.index.names if s in algo_levels]
    data = data.reset_index(level=algo_levels)
    id_concat = pd.Index(["_".join(i) for i in data.index], name="id_concat")
    data = data.assign(id_concat=id_concat)
    data = data.set_index([*algo_levels, "id_concat"])
    return data


def compute_improvement_outlier_correction(data: pd.DataFrame, outlier_algos: Sequence[str]) -> pd.DataFrame:
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
