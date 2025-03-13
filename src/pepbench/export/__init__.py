"""Module to export results to LaTeX tables."""

from pepbench.export._latex import (
    convert_to_latex,
    create_algorithm_result_table,
    create_nan_reason_table,
    create_outlier_correction_table,
    create_reference_pep_table,
)

__all__ = [
    "create_reference_pep_table",
    "create_algorithm_result_table",
    "create_nan_reason_table",
    "create_outlier_correction_table",
    "convert_to_latex",
]
