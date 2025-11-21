"""Module for various data handling helper functions.

This package provides helpers for:
- Adding a unique identifier column to results dataframes to track samples across datasets and enable reliable merging.
- Computing improvement metrics and running the evaluation pipeline with optional outlier handling.
- Producing performance summaries tailored to PEP estimation tasks for evaluation and reporting.
- Assessing and quantifying the relationship between reference PEP measurements and heart rate signals.
- Generating descriptive statistics and diagnostic summaries for PEP values across datasets or cohorts.
- Loading and filtering input data specific to a chosen algorithm or experimental setup.
- Computing error metrics grouped by experimental factors and aggregating those error statistics for analysis.
- Extracting PEP annotations or reference signals from dataset records for downstream processing.
- Retrieving canonical reference datasets or records used for benchmarking and validation.
- Merging metric outputs and reconciling per-sample results produced by multiple annotators into a unified view.
- Converting series of RR intervals into instantaneous heart rate values.
- Exposing miscellaneous low-level utility helpers used across the data handling code.
"""

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
