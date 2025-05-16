"""Module providing custom styling functions for :class:`~pandas.DataFrame.style` objects."""

import pandas as pd

__all__ = ["highlight_min_per_group", "highlight_min_uncertainty", "highlight_outlier_improvement"]


def highlight_outlier_improvement(col: pd.Series) -> pd.Series:
    """Highlights the improvement of the outlier correction algorithm by changing the background color.

    This is a custom styling function that can be used with the :meth:`~pandas.DataFrame.style.apply` method.

    Parameters
    ----------
    col : :class:`~pandas.Series`
        A pandas Series object where each element belongs to a group identified by 'b_point_algorithm'.

    Returns
    -------
    :class:`~pandas.Series`
        A pandas Series object with the same index as the input col, where the element
        corresponding to the outlier improvement is styled with "background-color: Pink"
        and other elements are styled with "background-color: LightGreen".
    """
    if "b_point_algorithm" in col.index.names:
        idx_name = "b_point_algorithm"
    elif "B-Point Detection" in col.index.names:
        idx_name = "B-Point Detection"
    else:
        raise ValueError("Index name 'b_point_algorithm' or 'B-Point Detection' not found in the index names.")
    none_is_min = col.groupby(idx_name).transform(lambda s: any(t in s.idxmin() for t in ["none", "None"]))
    return none_is_min.map(
        {
            True: "background-color: Pink",
            False: "background-color: LightGreen",
        }
    )


def highlight_min_per_group(col: pd.Series) -> pd.Series:
    """Highlights the minimum value in each group by changing the background color.

    This is a custom styling function that can be used with the :meth:`~pandas.DataFrame.style.apply` method.

    Parameters
    ----------
    col : :class:`~pandas.Series`
        A pandas Series object where each element belongs to a group identified by 'b_point_algorithm'.

    Returns
    -------
    :class:`~pandas.Series`
        A pandas Series object with the same index as the input col, where the element
        corresponding to the minimum value in each group is styled with "background-color: LightGreen"
        and other elements are styled with an empty string.
    """
    idx_min = col.groupby("b_point_algorithm").idxmin()
    return (pd.Series(col.index.isin(idx_min), index=col.index)).map(
        {
            True: "background-color: LightGreen",
            False: "",
        }
    )


def highlight_min_uncertainty(row: pd.Series) -> pd.Series:
    """Highlights the minimum value of an uncertainty metric (mean ± std) in a row by making the font bold.

    This is a custom styling function that can be used with the :meth:`~pandas.DataFrame.style.apply` method.

    Parameters
    ----------
    row : :class:`~pandas.Series`
        A pandas Series object where each element is a string containing a numeric value
        followed by a space and additional text.

    Returns
    -------
    :class:`~pandas.Series`
        A pandas Series object with the same index as the input row, where the element
        corresponding to the minimum value is styled with "font-weight: bold;" and other
        elements are styled with an empty string.
    """
    row = row.apply(lambda s: float(s.split(" ")[0]))
    idx_min = row.index.isin([row.idxmin()])
    return (pd.Series(idx_min, index=row.index)).map(
        {
            True: "font-weight: bold;",
            False: "",
        }
    )
