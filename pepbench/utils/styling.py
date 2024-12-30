import pandas as pd

__all__ = ["highlight_outlier_improvement", "highlight_min_per_group", "highlight_min_uncertainty"]


def highlight_outlier_improvement(col: pd.Series) -> pd.Series:
    none_is_min = col.groupby("b_point_algorithm").transform(lambda s: "none" in s.idxmin())
    return none_is_min.map(
        {
            True: "background-color: Pink",
            False: "background-color: LightGreen",
        }
    )


def highlight_min_per_group(col: pd.Series) -> pd.Series:
    idx_min = col.groupby("b_point_algorithm").idxmin()
    return (pd.Series(col.index.isin(idx_min), index=col.index)).map(
        {
            True: "background-color: LightGreen",
            False: "",
        }
    )


def highlight_min_uncertainty(row: pd.Series) -> pd.Series:
    row = row.apply(lambda s: float(s.split(" ")[0]))
    idx_min = row.index.isin([row.idxmin()])
    return (pd.Series(idx_min, index=row.index)).map(
        {
            True: "font-weight: bold;",
            False: "",
        }
    )
