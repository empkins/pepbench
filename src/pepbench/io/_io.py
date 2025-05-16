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
    """Load challenge results from a folder.

    Parameters
    ----------
    folder_path : str or :class:`pathlib.Path`
        The folder path containing the results.
    return_as_df : bool, optional
        ``True`` to return the results as DataFrames, ``False`` to return the results as dictionaries. Default: ``True``
    index_cols_single : list[str], optional
        The index columns for the single results. Default: ``["participant"]``
    index_cols_per_sample : list[str], optional
        The index columns for the per-sample results. Default: ``["participant"]``

    Returns
    -------
    tuple of dict or tuple of pd.DataFrame
        The results as a tuple of dictionaries or as a tuple of DataFrames.

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
    """Convert a given sampling frequency to milliseconds.

    Parameters
    ----------
    sampling_frequency: int
        The sampling frequency in Hz.

    Returns
    -------
    float
        The conversion factor from Hz to milliseconds.

    """
    conversion_factor = 1000 / sampling_frequency
    return conversion_factor
