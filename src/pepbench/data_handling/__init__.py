"""Module for various data handling helper functions."""

from pepbench.data_handling import utils
from pepbench.data_handling._data_handling import (
    add_unique_id_to_results_dataframe,
    compute_improvement_outlier_correction,
    compute_improvement_pipeline,
    compute_pep_performance_metrics,
    correlation_reference_pep_heart_rate,
    describe_pep_values,
    get_data_for_algo,
    get_error_by_group,
    get_pep_for_algo,
    get_reference_data,
    get_reference_pep,
    merge_result_metrics_from_multiple_annotators,
    merge_results_per_sample_from_different_annotators,
    rr_interval_to_heart_rate,
)

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
    "utils",
]
