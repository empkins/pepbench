"""Input/output helpers for challenge results and unit conversions.

This module provides utilities to load evaluation challenge results stored in a folder and to perform
simple unit conversions related to sampling rates. The routines here are intended to support the
package's evaluation workflow by returning results in the canonical :class:`pepbench.evaluation.ChallengeResults`
format (either as pandas DataFrames or as dictionaries).

The primary public functions are:

- :func:`pepbench.io._io.load_challenge_results_from_folder` — load and parse challenge result CSV files from a
  folder and return them as :class:`pepbench.evaluation.ChallengeResults`.
- :func:`pepbench.io._io.convert_hz_to_ms` — convert a sampling frequency in Hertz to a period in milliseconds.

See Also
--------
:mod:`pepbench.evaluation` for the :class:`pepbench.evaluation.ChallengeResults` container used by the loader.
"""

from collections.abc import Sequence

import pandas as pd

from pepbench.evaluation import ChallengeResults
from pepbench.utils._types import path_t

__all__ = ["convert_hz_to_ms", "load_challenge_results_from_folder"]


def load_challenge_results_from_folder(
    folder_path: path_t,
    index_cols_single: Sequence[str] | None = None,
    index_cols_per_sample: Sequence[str] | None = None,
    return_as_df: bool | None = True,
) -> ChallengeResults:
    """Load challenge results exported as CSV files from a folder.

    The function searches the provided folder for files following the naming conventions used by the
    evaluation pipeline (``*_agg_mean_std.csv``, ``*_agg_total.csv``, ``*_single.csv`` and ``*_per-sample.csv``),
    reads them, normalizes index/column names and returns a :class:`pepbench.evaluation.ChallengeResults`
    object.

    Parameters
    ----------
    folder_path : str or :class:`pathlib.Path`
        Path to the folder containing the result CSV files. The function asserts that the folder exists.
    index_cols_single : sequence of str or None, optional
        Column names to use as the index when reading ``*_single.csv`` files. If ``None``, defaults to
        the value of ``index_cols_per_sample`` (which by default is ``["participant"]``).
    index_cols_per_sample : sequence of str or None, optional
        Column names to use for the multi-level index when reading ``*_per-sample.csv`` files.
        If ``None``, defaults to ``["participant"]``.
    return_as_df : bool or None, optional
        If ``True`` (default) the function will return pandas DataFrames inside the returned
        :class:`pepbench.evaluation.ChallengeResults`. If ``False``, dictionaries mapping algorithm-tuples
        to DataFrames are returned instead.

    Returns
    -------
    ChallengeResults
        A :class:`pepbench.evaluation.ChallengeResults` instance containing the aggregated mean/std results,
        aggregated totals, per-participant single results and per-sample results. Each element will be either
        a pandas DataFrame (when ``return_as_df`` is True) or a dictionary mapping algorithm identifier tuples
        to DataFrames.

    Notes
    -----
    - The function expects files to follow the naming convention used by the evaluation pipeline and will
      build tuple keys from filename parts to identify algorithm combinations.
    - When ``return_as_df`` is ``True``, per-sample columns with suffixes like ``_id`` or ``_sample`` are
      coerced to pandas' nullable ``Int64`` dtype where possible. Conversion of ``_ms`` and ``_percent``
      columns to nullable float dtypes is currently disabled due to compatibility issues with some plotting
      libraries.
    - The function asserts that ``folder_path`` exists and is a directory.

    Examples
    --------
    >>> from pathlib import Path
    >>> results = load_challenge_results_from_folder(Path("/path/to/results"), return_as_df=True)
    >>> isinstance(results, ChallengeResults)
    True
    """
    assert folder_path.is_dir(), f"Folder '{folder_path}' does not exist!"

    if index_cols_per_sample is None:
        index_cols_per_sample = ["participant"]

    if index_cols_single is None:
        index_cols_single = index_cols_per_sample

    result_files_agg_mean_std = sorted(folder_path.glob("*_agg_mean_std.csv"))
    result_files_agg_total = sorted(folder_path.glob("*_agg_total.csv"))
    result_files_single = sorted(folder_path.glob("*_single.csv"))
    result_files_per_sample = sorted(folder_path.glob("*_per-sample.csv"))
    dict_agg_mean_std = {}
    dict_agg_total = {}
    dict_single = {}
    dict_per_sample = {}

    for file in result_files_agg_mean_std:
        file_paras = file.stem.split("_")
        algo_types = tuple(file_paras[3:6])
        data = pd.read_csv(file, index_col=0)
        data.index.name = "metric"
        dict_agg_mean_std[algo_types] = data

    for file in result_files_agg_total:
        file_paras = file.stem.split("_")
        algo_types = tuple(file_paras[3:6])
        data = pd.read_csv(file, index_col=0)
        data.index.name = "metric"
        dict_agg_total[algo_types] = data

    for file in result_files_single:
        file_paras = file.stem.split("_")
        algo_types = tuple(file_paras[3:6])
        data = pd.read_csv(file, index_col=index_cols_single)
        dict_single[algo_types] = data

    for file in result_files_per_sample:
        file_paras = file.stem.split("_")
        algo_types = tuple(file_paras[3:6])
        index_cols = list(range(len(index_cols_per_sample) + 1))
        data = pd.read_csv(file, header=[0, 1], index_col=index_cols)
        data.index = data.index.set_names("id", level=-1)
        dict_per_sample[algo_types] = data

    if return_as_df:
        results_agg_mean_std = pd.concat(
            dict_agg_mean_std, names=["q_peak_algorithm", "b_point_algorithm", "outlier_correction_algorithm"]
        )
        results_agg_total = pd.concat(
            dict_agg_total, names=["q_peak_algorithm", "b_point_algorithm", "outlier_correction_algorithm"]
        )
        results_single = pd.concat(
            dict_single, names=["q_peak_algorithm", "b_point_algorithm", "outlier_correction_algorithm"]
        )
        results_per_sample = pd.concat(
            dict_per_sample, names=["q_peak_algorithm", "b_point_algorithm", "outlier_correction_algorithm"]
        )
        # all columns with suffix "_sample" or "_id" should be "Int64"
        # all columns with suffix "_ms" or "_percent" should be "Float64"
        dtype_dict = {col: "Int64" for col in results_per_sample.columns if col[0].endswith("_id")}
        dtype_dict.update({col: "Int64" for col in results_per_sample.columns if col[0].endswith("_sample")})
        # TODO: this is, for now, commented out because the nan-safe pandas data types fail with the currently
        #  installed seaborn and pingouin versions => this should be fixed in the future
        # dtype_dict.update({col: "Float64" for col in results_per_sample.columns if col[0].endswith("_ms")})
        # dtype_dict.update({col: "Float64" for col in results_per_sample.columns if col[0].endswith("_percent")})
        results_per_sample = results_per_sample.astype(dtype_dict)

        return ChallengeResults(results_agg_mean_std, results_agg_total, results_single, results_per_sample)

    return ChallengeResults(dict_agg_mean_std, dict_agg_total, dict_single, dict_per_sample)


def convert_hz_to_ms(sampling_frequency: float) -> float:
    """Convert a sampling frequency in Hertz to a period in milliseconds.

    Parameters
    ----------
    sampling_frequency : float
        Sampling frequency in Hz.

    Returns
    -------
    float
        Period in milliseconds equal to ``1000 / sampling_frequency``.

    Examples
    --------
    >>> convert_hz_to_ms(100.0)
    10.0
    """
    conversion_factor = 1000 / sampling_frequency
    return conversion_factor
