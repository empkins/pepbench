from typing import Optional

import pandas as pd

from pepbench.utils._types import path_t

__all__ = ["load_preprocessed_training_data", "compute_abs_error", "compute_error"]

def load_preprocessed_training_data(
        file_path: path_t,
        algorithms: Optional[pd.MultiIndex] = None,
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
    assert file_path.is_file() , f"File '{file_path}' does not exist!"
    assert event in supported_events, f"Event '{event}' is not supported. Supported events: {supported_events}.\n"

    data = pd.read_csv(file_path, index_col=["participant", "phase", "heartbeat_id_reference"])

    if algorithms is not None:
        algos = None
        if event == 'b-point':
            if 'outlier_correction_algorithm' in data.columns:
                algos = algorithms.get_level_values('b_point_algorithm') + "_" + algorithms.get_level_values('outlier_correction_algorithm')
            else:
                algos = algorithms.get_level_values('b_point_algorithm') + "_" + "none"
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
        The errors of the extracted B-Point locations against the labeled reference data
    """
    if reference.name in input_data.columns:
        input_data = input_data.drop(columns=reference.name)
        
    error = input_data.subtract(reference, axis=0)
    return error