from typing import Optional

import pandas as pd
import numpy as np

from pepbench.utils._types import path_t
from biopsykit.classification.model_selection import SklearnPipelinePermuter
import shap

__all__ = ["load_preprocessed_training_data", "compute_mae_std_from_permuter", "compute_mae_std_from_metric_summary", "compute_abs_error", "compute_error",
           "impute_missing_values", "compute_shap_values"]


def load_preprocessed_training_data(
        file_path: path_t,
        algorithms: Optional[pd.MultiIndex] = None,
        index_col: Optional[list] = None,
        include_reference: Optional[bool] = True,
        event: Optional[str] = 'b-point'
):
    """
    Load preprocessed training data for evaluation of the model output.

    Parameters
    ----------
    file_path: str or :class:`pathlib.Path
        The file path containing the preprocessed training data.
    algorithms: pd.MultiIndex
        The index specifying the algorihtms that should be returned for comparision.
    index_col: list
        List containing the columns that should serve as the index of the dataframe.
    include_reference: bool, optional
        ``True`` to include the reference B-Points, ``False`` to exclude the reference B-Points. Default: ``True``
    event: str, optional
        Event that is represented in the training data. Can be either ``'b-point'`` or ``'q_wave'``. Default: ``'b-point'``

    Returns
    -------
    pd.DataFrame
        The B-Points extracted by the algorithms in ms.
    """

    supported_events = ['b-point', 'q_wave']
    assert file_path.is_file(), f"File '{file_path}' does not exist!"
    assert event in supported_events, f"Event '{event}' is not supported. Supported events: {supported_events}.\n"

    data = pd.read_csv(file_path, index_col=index_col)

    if algorithms is not None:
        algos = None
        if event == 'b-point':
            if 'outlier_correction_algorithm' in data.columns:
                algos = algorithms.get_level_values('b_point_algorithm') + "_" + algorithms.get_level_values(
                    'outlier_correction_algorithm')
            else:
                algos = algorithms.get_level_values('b_point_algorithm')
            if include_reference:
                return data[["b_point_sample_reference"] + list(algos.values)]
            else:
                return data[algos.values]
        elif event == 'q_wave':
            algos = algorithms.get_level_values('q_wave_algorithm')
            if include_reference:
                return data[['q_wave_onset_sample_reference'] + list(algos.values)]
            else:
                return data[algos.values]
        else:
            raise KeyError(f"Event: '{event}' is not supported. Supported events: {supported_events}.\n")
    else:
        return data


def compute_mae_std_from_permuter(
        pipeline_permuter: SklearnPipelinePermuter,
):
    """
    Compute mae and std from permuter.
    Parameters
    ----------
    input_data: SklearnPipelinePermuter
        Pipeline permuter containing the regression results.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the mae, std, true_labels, predicted_labels, and the absolute_error of the predictions.
    """
    if pipeline_permuter is not None:
        permuter_metrics = pd.DataFrame(data=pipeline_permuter.metric_summary()[['true_labels', 'predicted_labels']],
                                        columns=['mae', 'std', 'true_labels', 'predicted_labels'],
                                        index=pipeline_permuter.metric_summary().index)


    permuter_metrics['absolute_error'] = np.abs(permuter_metrics['true_labels'] - permuter_metrics['predicted_labels'])

    for index in permuter_metrics.index:
        permuter_metrics.at[index, 'mae'] = np.mean(permuter_metrics.loc[index]['absolute_error'])
        permuter_metrics.at[index, 'std'] = np.std(permuter_metrics.loc[index]['absolute_error'])
    return permuter_metrics

def compute_mae_std_from_metric_summary(
        pipeline_permuter: pd.DataFrame,
):
    """
    Compute mae and std from permuter.
    Parameters
    ----------
    input_data: pd.DataFrame
        DataFrame containing the regression results.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the mae, std, true_labels, predicted_labels, and the absolute_error of the predictions.
    """
    permuter_metrics = pd.DataFrame(data=pipeline_permuter[['true_labels', 'predicted_labels']],
                                        columns=['mae', 'std', 'true_labels', 'predicted_labels'],
                                        index=pipeline_permuter.index)

    permuter_metrics['absolute_error'] = np.abs(permuter_metrics['true_labels'] - permuter_metrics['predicted_labels'])

    for index in permuter_metrics.index:
        permuter_metrics.at[index, 'mae'] = np.mean(permuter_metrics.loc[index]['absolute_error'])
        permuter_metrics.at[index, 'std'] = np.std(permuter_metrics.loc[index]['absolute_error'])
    return permuter_metrics


def compute_abs_error(
        input_data: pd.DataFrame,
        reference: pd.Series,
):
    """
    Compute the absolute error of the B-Point extraction algorithms present in the Datframe against the labeled reference data.

    Parameters
    ----------
    input data: pd.DataFrame
        Dataframe containing the automatically extracted B-Point locations and the labeled reference data
    reference: pd.Series
        Series containing the column against which the absolute error should be calculated

    Returns
    -------
    pd.DataFrame
        The absolute errors of the extracted B-Point locations against the labeled reference data
    """
    if reference.name in input_data.columns:
        input_data = input_data.drop(columns=reference.name)

    abs_error = input_data.subtract(reference, axis=0).abs()
    return abs_error


def compute_error(
        input_data: pd.DataFrame,
        reference: pd.Series,
):
    """
    Compute the error of the B-Point extraction algorithms present in the Datframe against the labeled reference data.

    Parameters
    ----------
    input data: pd.DataFrame
        Dataframe containing the automatically extracted B-Point locations and the labeled reference data
    reference: pd.Series
        Series containing the column against which the absolute error should be calculated

    Returns
    -------
    pd.DataFrame
        The errors of the extracted B-Point locations against the labeled reference data
    """
    if reference.name in input_data.columns:
        input_data = input_data.drop(columns=reference.name)

    error = input_data.subtract(reference, axis=0) * -1
    return error


def impute_missing_values(
        input_data: pd.DataFrame,
        mode: Optional[str] = 'median'
):
    """
    Impute missing values in the training data on sample- (row-) level.
    Parameters
    ----------
    input_data: pd.DataFrame
        Dataframe containing the training data that should be imputed.
    mode: str
        Technique that should be used to impute missing values. Can be ``'median'`` or ``'mean'``. Default: ``'median'``.
    Returns
    -------
    pd.DataFrame
        Dataframe containing the training data with missing values imputed.
    """

    if mode == 'median':
        imputation_values = np.nanmedian(input_data, axis=1)
    elif mode == 'mean':
        imputation_values = np.round(np.nanmean(input_data, axis=1))
    else:
        raise KeyError(f"Imputation mode '{mode}' is not supported. Supported modes are 'median' and 'mean'.")

    for sample in range(input_data.shape[0]):
        input_data.iloc[sample, np.isnan(input_data.iloc[sample, :].values)] = imputation_values[sample]

    return input_data

def compute_shap_values(
        pipeline_permuter: SklearnPipelinePermuter,
        training_data: pd.DataFrame,
        pipeline_elements: tuple
):
    """
    Compute the SHAP values of the machine learning estimator.
    Parameters
    ----------
    pipeline_permuter: SklearnPipelinePermuter
        Pipeline Permuter containing the trained machine learning estimators.
    training_data: pd.DataFrame
        Data that was used for training the estimators.
    pipeline: list(str)
        Pipeline combination shap values should be computed on.
    Returns
    -------
    shap_per_fold: np.array
        SHAP values of the machine learning estimator per fold.
    test_folds: np.array
        Test folds used for training of the machine learning estimator.
    pipeline_folds: list
        List containing the pipeline objects per fold.
    """
    pipeline_folds = pipeline_permuter.best_estimator_summary().loc[pipeline_elements]['best_estimator'].pipeline
    test_folds = pipeline_permuter.metric_summary().loc[pipeline_elements]['test_indices_folds']
    training_folds = pipeline_permuter.metric_summary().loc[pipeline_elements]['train_indices_folds']
    shap_per_fold = []
    for fold, pipeline in enumerate(pipeline_folds):
        X_test = training_data.iloc[list(test_folds[fold]), :]
        X_train = training_data.iloc[list(training_folds[fold]), :]
        num_samples_to_select = min(1000, X_train.shape)
        random_indices = np.random.choice(X_train.shape, num_samples_to_select, replace=False)
        X_train_subset = X_train[random_indices, :]
        estimator = pipeline[-1]
        # Create Explainer object that can calculate shap values
        explainer = shap.TreeExplainer(estimator, data=X_train_subset, feature_perturbation='interventional')
        shap_values_fold = np.array(explainer.shap_values(X_test))
        shap_per_fold.append(shap_values_fold)
    return shap_per_fold, test_folds, pipeline_folds
