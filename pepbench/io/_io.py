from collections.abc import Sequence
from typing import Optional

import pandas as pd

from pepbench.evaluation import ChallengeResults
from pepbench.utils._types import path_t
from biopsykit.classification.model_selection import SklearnPipelinePermuter

__all__ = ["load_challenge_results_from_folder", "load_best_performing_algos_b_point", "load_best_performing_algos_q_wave", "get_best_pipeline_results", "get_best_estimator", "get_pipeline_steps", "convert_hz_to_ms"]


def load_challenge_results_from_folder(
    folder_path: path_t,
    index_cols_single: Optional[Sequence[str]] = None,
    index_cols_per_sample: Optional[Sequence[str]] = None,
    return_as_df: Optional[bool] = True,
) -> ChallengeResults:
    """
    Load challenge results from a folder.

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
        dict_per_sample[algo_types] = data

    if return_as_df:
        results_agg_mean_std = pd.concat(
            dict_agg_mean_std, names=["q_wave_algorithm", "b_point_algorithm", "outlier_correction_algorithm"]
        )
        results_agg_total = pd.concat(
            dict_agg_total, names=["q_wave_algorithm", "b_point_algorithm", "outlier_correction_algorithm"]
        )
        results_single = pd.concat(
            dict_single, names=["q_wave_algorithm", "b_point_algorithm", "outlier_correction_algorithm"]
        )
        results_per_sample = pd.concat(
            dict_per_sample, names=["q_wave_algorithm", "b_point_algorithm", "outlier_correction_algorithm"]
        )
        return ChallengeResults(results_agg_mean_std, results_agg_total, results_single, results_per_sample)

    return ChallengeResults(dict_agg_mean_std, dict_agg_total, dict_single, dict_per_sample)

def load_best_performing_algos_b_point(
        folder_path: path_t,
        n_best: Optional[int] = 5,
        outlier_correction: Optional[bool] = False,
):
    """
    Load the best performing B-Point Detection algorithms from folder based on their mean absolute error.
    
    Parameters
    ----------
    folder_path : str or :class:`pathlib.Path`
        The folder path containing the results.
    n_best: int, optional
        The amount of algorithms that should be returned. Default: ``5``
    outlier_correction: bool, optional
        Specifies whether the outlier correction algorithms should be taken into account. Default: ```False``

    Returns
    -------
    pd.Dataframe
        The n_best algorihtms with the lowest mean absolute error as a pd.Dataframe.

    """
    assert folder_path.is_dir(), f"Folder '{folder_path}' does not exist!"

    if type(n_best) is not int:
        raise TypeError(f"Expected type int, received type {type(n_best)}!")
    
    result_files_agg_mean_std = sorted(folder_path.glob("*_agg_mean_std.csv"))
    dict_agg_mean_std = {}

    for file in result_files_agg_mean_std:
        file_paras = file.stem.split("_")
        algo_types = tuple(file_paras[3:6])
        data = pd.read_csv(file, index_col=0)
        data.index.name = "metric"
        dict_agg_mean_std[algo_types] = data

    agg_mean_std = pd.concat(dict_agg_mean_std, names=["q_wave_algorithm", "b_point_algorithm", "outlier_correction_algorithm"]).droplevel("q_wave_algorithm")

    if outlier_correction == False:
        agg_mean_std = agg_mean_std.xs(key='none', level='outlier_correction_algorithm', drop_level=True)

    best_agg_mean_std = agg_mean_std.xs(key="absolute_error_ms", level="metric").nsmallest(n_best, "mean")

    return best_agg_mean_std

def load_best_performing_algos_q_wave(
        folder_path: path_t,
        n_best: Optional[int] = 5,
):
    """
    Load the best performing B-Point Detection algorithms from folder based on their mean absolute error.
    
    Parameters
    ----------
    folder_path : str or :class:`pathlib.Path`
        The folder path containing the results.
    n_best: int, optional
        The amount of algorithms that should be returned. Default: ``5``

    Returns
    -------
    pd.Dataframe
        The n_best algorihtms with the lowest mean absolute error as a pd.Dataframe.

    """
    assert folder_path.is_dir(), f"Folder '{folder_path}' does not exist!"

    if type(n_best) is not int:
        raise TypeError(f"Expected type int, received type {type(n_best)}!")
    
    result_files_agg_mean_std = sorted(folder_path.glob("*_agg_mean_std.csv"))
    dict_agg_mean_std = {}

    for file in result_files_agg_mean_std:
        file_paras = file.stem.split("_")
        algo_types = tuple(file_paras[3:6])
        data = pd.read_csv(file, index_col=0)
        data.index.name = "metric"
        dict_agg_mean_std[algo_types] = data

    agg_mean_std = pd.concat(dict_agg_mean_std, names=["q_wave_algorithm", "b_point_algorithm", "outlier_correction_algorithm"]).droplevel(["b_point_algorithm", "outlier_correction_algorithm"])
    agg_mean_std = agg_mean_std.drop(index="scipy-findpeaks")

    best_agg_mean_std = agg_mean_std.xs(key="absolute_error_ms", level="metric").nsmallest(n_best, "mean")

    return best_agg_mean_std

def get_best_pipeline_results(
        pipeline_permuter: SklearnPipelinePermuter,
        metric: Optional[str] = "mean_absolute_error",
):  
    """
    Get the best performing algorithm from the metric summary of the SklearnPipelinePermuter.

    Parameters
    ----------
    pipeline_permuter: class biopsykit.classification.model_selection.SklearnPipelinePermuter
        The pipeline permuter object containing the prediction results and metric performances.
    metric: str, optional
        The metric that was used for scoring in the SklearnPipelinePermuter. Default ``"mean_absolute_error"``

    Returns
    -------
    pd.DataFrame
        The prediction and metric performance results of the best estimator.

    """
    if metric == "mean_absolute_error":
        sklearn_metric = "mean_test_neg_mean_absolute_error"
    else:
        raise KeyError(f"Specified metric is not implemented yet!")
    
    assert sklearn_metric in pipeline_permuter.metric_summary().columns, "Specified metric is not used in the pipeline permuter!"
    return pipeline_permuter.metric_summary().iloc[pipeline_permuter.metric_summary()["mean_test_neg_mean_absolute_error"].argmin()].to_frame().T

def get_best_estimator(
        pipeline_permuter: SklearnPipelinePermuter,
        metric: Optional[str] = "mean_absolute_error",
):
    """
    Get the best estimator object and its underlying pipeline combination.

    Parameters
    ----------
    pipeline_permuter: class biopsykit.classification.model_selection.SklearnPipelinePermuter
        The pipeline permuter object containing the prediction results and metric performances.
    metric: str, optional
        The metric that was used for scoring in the SklearnPipelinePermuter. Default ``"mean_absolute_error"``

    Returns
    -------
    tuple of _PipelineWrapper and tuple
        The results as a tuple containing of the biopsykit.classification.utils._PipelineWrapper object and a tuple of 
        strings containing the underlying pipeline combination.
    """

    if metric == "mean_absolute_error":
        sklearn_metric = "mean_test_neg_mean_absolute_error"
    else:
        raise KeyError(f"Specified metric is not implemented yet!")
    
    assert sklearn_metric in pipeline_permuter.metric_summary().columns, "Specified metric is not used in the pipeline permuter!"
    best_estimator_name = pipeline_permuter.metric_summary().iloc[pipeline_permuter.metric_summary()[sklearn_metric].argmin()].name
    return pipeline_permuter.best_estimator_summary().loc[best_estimator_name].values[0], best_estimator_name

def get_pipeline_steps(
        pipeline_permuter: SklearnPipelinePermuter,
        input_data: pd.DataFrame,
        metric: Optional[str] = "mean_absolute_error",
        step: Optional[str] = "reduce_dim",
        scaler: Optional[bool] = True,
):
    """
    Gain further insights in the components of your model. E.g. recieve the features selected by the model.

    Parameters
    ----------
    pipeline_permuter: class biopsykit.classification.model_selection.SklearnPipelinePermuter
        The pipeline permuter object containing the prediction results and metric performances.
    input_data: pd.DataFrame
        The data that was used to train the models.
    metric: str, optional
        The metric that was used for scoring in the SklearnPipelinePermuter. Default ``"mean_absolute_error"``
    step: str, optional
        The step of interest. Default ``"reduce_dim"``
    scaler: str, optional
        Specifies whether scaling was considered in the model. Default ``True``

    Results
    -------
    list
        List of selected features for each pipeline in the best_estimator object.
    
    """
    best_estimator, best_estimator_name = get_best_estimator(pipeline_permuter=pipeline_permuter, metric=metric)
    reduce_dim_model = ""
    selected_features = []
    selected_feature_mask = None

    if scaler:
        if step == "reduce_dim":
            reduce_dim_model = best_estimator_name[1]
        else:
            raise KeyError("Specified step is not supported yet!")
    else:
        if step == "reduce_dim":
            reduce_dim_model = best_estimator_name[0]
        else:
            raise KeyError("Specified step is not supported yet!")
    
    for pipeline in best_estimator.pipeline:
        if reduce_dim_model == "RFE":
            selected_feature_mask = pipeline.named_steps["reduce_dim"].get_support()
        elif reduce_dim_model == "SelectKBest":
            selected_feature_mask = pipeline.named_steps["reduce_dim"].get_support()
        else:
            raise KeyError("The model used in your pipeline is not supported yet!")
        selected_features.append(input_data.columns[selected_feature_mask].to_list())
    return selected_features
    
    

def convert_hz_to_ms(sampling_frequency):
    """
    Convert Hz to ms.

    Parameters
    ----------
    sampling_frequency: int
    The sampling freqency that should be converted in milliseconds.

    Returns
    -------
    float 
        The conversion factor from samples to ms.
    """
    conversion_factor = 1000 / sampling_frequency
    return conversion_factor
