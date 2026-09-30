"""Input/output helpers for challenge results and unit conversions.

This module provides convenience functions for loading and saving evaluation challenge results and for
simple unit conversions related to sampling rates. The helper functions are thin wrappers around the
implementations in :mod:`pepbench.io._io` and are intended to provide a stable public API for users of
the package.

The primary public functions are:

- :func:`pepbench.io.convert_hz_to_ms` — convert a frequency in Hertz to a period in milliseconds.
- :func:`pepbench.io.load_challenge_results_from_folder` — load evaluation results stored in a folder
  and return them in the package's canonical format.

See :mod:`pepbench.io._io` for implementation details.

"""

from pepbench.io._io import load_challenge_results_from_folder
from pepbench.io._io import load_best_performing_algos_b_point
from pepbench.io._io import load_best_performing_algos_q_wave
from pepbench.io._io import get_best_pipeline_results
from pepbench.io._io import get_best_estimator
from pepbench.io._io import get_pipeline_steps
from pepbench.io._io import convert_hz_to_ms
from pepbench.io._ml_helper import load_preprocessed_training_data
from pepbench.io._ml_helper import compute_mae_std_from_permuter
from pepbench.io._ml_helper import compute_mae_std_from_metric_summary
from pepbench.io._ml_helper import compute_abs_error
from pepbench.io._ml_helper import compute_error
from pepbench.io._ml_helper import impute_missing_values

from pepbench.io._io import get_best_estimator
__all__ = ["load_challenge_results_from_folder", "load_best_performing_algos_b_point", "load_best_performing_algos_q_wave",
           "get_best_pipeline_results", "get_best_estimator", "get_pipeline_steps", "convert_hz_to_ms",
           "load_preprocessed_training_data", "compute_mae_std_from_permuter", "compute_mae_std_from_metric_summary", "compute_abs_error", "compute_error",
           "impute_missing_values"]
