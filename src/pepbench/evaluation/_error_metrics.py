import numpy as np
import pandas as pd

__all__ = ["abs_error", "abs_rel_error", "error", "rel_error"]


def error(ref_data: pd.Series, est_data: pd.Series) -> pd.Series:
    """Calculate the error between the reference and estimated values.

    Parameters
    ----------
    ref_data: :class:`pandas.Series`
        The reference values.
    est_data : :class:`pandas.Series`
        The estimated values.

    Returns
    -------
    error : :class:`pandas.Series`
        The error between the detected and reference values in the form `ref_data` - `est_data`

    """
    return ref_data - est_data


def rel_error(ref_data: pd.Series, est_data: pd.Series) -> pd.Series:
    """Calculate the relative error between the reference and estimated values.

    Parameters
    ----------
    ref_data : :class:`pandas.Series`
        The reference values.
    est_data : :class:`pandas.Series`
        The estimated values.

    Returns
    -------
    error : :class:`pandas.Series`
        The relative error between the reference and estimated values in the form (`ref_data` - `est_data`) / `ref_data`

    """
    result = (ref_data - est_data) / ref_data
    result = result.replace([np.inf, -np.inf], pd.NA)
    return result


def abs_error(ref_data: pd.Series, est_data: pd.Series) -> pd.Series:
    """Calculate the absolute error between the reference and estimated values.

    Parameters
    ----------
    ref_data : :class:`pandas.Series`
        The reference values.
    est_data : :class:`pandas.Series`
        The estimated values.

    Returns
    -------
    error : :class:`pandas.Series`
        The absolute error between the reference and estimated values in the
        form `abs(ref_data - est_data)`

    """
    return np.abs(ref_data - est_data)


def abs_rel_error(ref_data: pd.Series, est_data: pd.Series) -> pd.Series:
    """Calculate the absolute relative error between the reference and estimated values.

    Parameters
    ----------
    ref_data : :class:`pandas.Series`
        The reference values.
    est_data : :class:`pandas.Series`
        The estimated values.

    Returns
    -------
    error : :class:`pandas.Series`
        The absolute relative error between the reference and estimated values in the
        form `abs((ref_data - est_data) / ref_data)`

    """
    result = np.abs((ref_data - est_data) / ref_data)
    result = result.replace([np.inf, -np.inf], pd.NA)
    return result
