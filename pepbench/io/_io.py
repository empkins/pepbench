from collections.abc import Sequence
from typing import Optional, Union

import pandas as pd

from pepbench.utils._types import path_t

__all__ = ["load_challenge_results_from_folder"]


def load_challenge_results_from_folder(
    folder_path: path_t,
    index_cols_single: Optional[Sequence[str]] = None,
    index_cols_per_sample: Optional[Sequence[str]] = None,
    return_as_df: Optional[bool] = True,
) -> Union[tuple[dict[str, pd.DataFrame], ...], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
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

    if index_cols_single is None:
        index_cols_single = ["participant"]

    if index_cols_per_sample is None:
        index_cols_per_sample = ["participant"]

    result_files_agg = sorted(folder_path.glob("*_agg.csv"))
    result_files_single = sorted(folder_path.glob("*_single.csv"))
    result_files_per_sample = sorted(folder_path.glob("*_per-sample.csv"))
    dict_agg = {}
    dict_single = {}
    dict_per_sample = {}

    for file in result_files_agg:
        file_paras = file.stem.split("_")
        algo_types = tuple(file_paras[3:6])
        data = pd.read_csv(file, index_col=0)
        data.index.name = "metric"
        dict_agg[algo_types] = data

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
        dict_per_sample[algo_types] = data

    if return_as_df:
        results_agg = pd.concat(
            dict_agg, names=["q_wave_algorithm", "b_point_algorithm", "outlier_correction_algorithm"]
        )
        results_single = pd.concat(
            dict_single, names=["q_wave_algorithm", "b_point_algorithm", "outlier_correction_algorithm"]
        )
        results_per_sample = pd.concat(
            dict_per_sample, names=["q_wave_algorithm", "b_point_algorithm", "outlier_correction_algorithm"]
        )
        return results_agg, results_single, results_per_sample

    return dict_agg, dict_single, dict_per_sample
