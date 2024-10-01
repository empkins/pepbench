from collections.abc import Sequence
from typing import Optional

import pandas as pd

__all__ = [
    "get_reference_data",
    "get_reference_pep",
    "get_pep_for_algo",
    "get_data_for_algo",
    "describe_pep_values",
    "compute_pep_performance_metrics",
]

from pepbench.utils._types import str_t

_algo_levels = ["q_wave_algorithm", "b_point_algorithm", "outlier_correction_algorithm"]

_pep_error_metric_map = {
    "absolute_error_per_sample_ms": "Mean Absolute Error [ms]",
    "error_per_sample_ms": "Mean Error [ms]",
    "absolute_relative_error_per_sample_percent": "Mean Relative Error [%]",
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


def get_data_for_algo(results_per_sample: pd.DataFrame, algo_combi: Sequence[str]) -> pd.DataFrame:
    pep = results_per_sample.xs(tuple(algo_combi), level=_algo_levels)

    return pep


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
    results_per_sample = results_per_sample[_pep_error_metric_map.keys()].droplevel(level=-1, axis=1)
    results_per_sample = results_per_sample.groupby(_algo_levels)
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
