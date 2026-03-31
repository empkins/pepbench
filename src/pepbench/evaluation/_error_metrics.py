r"""Error metrics for evaluating estimated values against reference values.

Provides vectorized helper functions operating on :class:`pandas.Series` objects to compute
signed error, relative error, and their absolute variants. Relative metrics
normalize division by zero (infinite results) to ``pd.NA``.

Functions
---------
error
    Signed error ``ref - est``.
rel_error
    Relative signed error ``(ref - est) / ref``.
abs_error
    Absolute error ``abs(ref - est)``.
abs_rel_error
    Absolute relative error ``abs((ref - est) / ref)``.

Notes
-----
- All functions are elementwise and expect index-aligned input series.
- Relative error is undefined for reference values equal to zero; those positions are reported as ``pd.NA``.
"""

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
        The estimated values. Must be index-aligned with ``ref_data``.

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
