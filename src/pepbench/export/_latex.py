import inspect
from collections.abc import Sequence
from typing import Any

import pandas as pd
from pandas.io.formats.style import Styler

from pepbench.utils._rename_maps import (
    _algo_level_mapping,
    _algorithm_mapping,
    _metric_mapping,
    _nan_reason_mapping,
    _nan_reason_mapping_short,
)

__all__ = [
    "convert_to_latex",
    "create_algorithm_result_table",
    "create_nan_reason_table",
    "create_outlier_correction_table",
    "create_reference_pep_table",
]


def create_reference_pep_table(data: pd.DataFrame) -> pd.DataFrame:
    # Create a new DataFrame with formatted columns
    formatted_data = data.copy()
    formatted_data.index.names = [s.capitalize() for s in formatted_data.index.names]

    formatted_data[("pep_ms", "mean_std")] = formatted_data.apply(
        lambda row: f"{row[('pep_ms', 'mean')]:.2f}({row[('pep_ms', 'std')]:.2f})",
        axis=1,
    )
    formatted_data[("pep_ms", "range")] = formatted_data.apply(
        lambda row: f"[{int(row[('pep_ms', 'min')])}, {int(row[('pep_ms', 'max')])}]",
        axis=1,
    )

    # Select only the required columns for the LaTeX table
    formatted_data = formatted_data[[("pep_ms", "mean_std"), ("pep_ms", "range")]]
    formatted_data.columns = ["M ± SD [ms]", "Range [ms]"]

    return formatted_data


def create_algorithm_result_table(data: pd.DataFrame, collapse_algo_levels: bool = False) -> pd.DataFrame:
    data = data.copy()

    # Create a new DataFrame with formatted columns
    formatted_data = pd.DataFrame(index=data.index)
    algo_levels = [_algo_level_mapping[s] for s in data.index.names if s in _algo_level_mapping]
    formatted_data.index.names = [_algo_level_mapping[s] for s in formatted_data.index.names]

    export_cols_m_sd = ["Mean Absolute Error [ms]", "Mean Error [ms]", "Mean Absolute Relative Error [%]"]
    export_cols_m_sd = [s for s in export_cols_m_sd if s in data.columns]

    for key in export_cols_m_sd:
        # formatted_data[key] = data.apply(lambda row: print(row[(key, "mean")]), axis=1)
        formatted_data[key] = data.apply(
            lambda row, k=key: rf"{row[(k, 'mean')]:.1f} \pm {row[(k, 'std')]:.1f}", axis=1
        )

    if "Invalid PEPs" in data.columns:
        formatted_data[r"Invalid PEPs"] = data.apply(
            lambda row: f"{int(row[('Invalid PEPs', 'total')])} "
            rf"({(row[('Invalid PEPs', 'total')] / row[('Total PEPs', 'total')] * 100):.1f} \%)",
            axis=1,
        )

    # rename index values
    formatted_data = formatted_data.rename(index=_algorithm_mapping)

    if collapse_algo_levels:
        formatted_data = formatted_data.reset_index()
        formatted_data["algo_merged"] = formatted_data.apply(lambda row: " | ".join(row[algo_levels]), axis=1)
        formatted_data = formatted_data.set_index("algo_merged").drop(columns=algo_levels)

    return formatted_data


def create_outlier_correction_table(data: pd.DataFrame, outlier_algos: Sequence[str] | None = None) -> pd.DataFrame:
    data_index = data.index.get_level_values("b_point_algorithm").unique()
    data = data.groupby("b_point_algorithm", group_keys=False).apply(
        lambda df: df.reindex(outlier_algos, level="outlier_correction_algorithm")
    )
    data = data.reindex(data_index, level="b_point_algorithm")
    data = data.rename(index=_algorithm_mapping)
    data = data.rename(_metric_mapping, axis=1)
    data.index.names = [_algo_level_mapping[s] for s in data.index.names]
    return data


def create_nan_reason_table(
    data: pd.DataFrame, outlier_algos: Sequence[str] | None = None, use_short_names: bool = True
) -> pd.DataFrame:
    data = data[[("nan_reason", "estimated")]].dropna().droplevel(level=-1, axis=1)
    data = data.groupby(["b_point_algorithm", "outlier_correction_algorithm"]).value_counts()
    data = data.unstack().fillna(0).astype("Int64")
    if outlier_algos is not None:
        data = data.reindex(outlier_algos, level="outlier_correction_algorithm")
    data.columns.name = "Reason"
    if use_short_names:
        data = data.rename(columns=_nan_reason_mapping_short)
    else:
        data = data.rename(columns=_nan_reason_mapping)
    data = data.rename(index=_algorithm_mapping)
    data.index.names = [_algo_level_mapping[s] for s in data.index.names]

    return data


def convert_to_latex(
    data: pd.DataFrame | Styler, collapse_index_columns: bool = False, **kwargs: dict[str, Any]
) -> str:
    kwargs.setdefault("hrules", True)
    kwargs.setdefault("position", "ht")
    kwargs.setdefault("siunitx", True)
    kwargs.setdefault("clines", "skip-last;data")
    if "environment" not in kwargs:
        kwargs.setdefault("position_float", "centering")
    kwargs.setdefault("convert_css", True)
    rename_map = {
        "Q-Peak Detection": "Algorithm",
        "Mean Absolute Error [ms]": r"\ac{MAE} [ms]",
        "Mean Error [ms]": r"\ac{ME} [ms]",
        "Mean Absolute Relative Error [%]": r"\ac{MARE} [%]",
        "Invalid PEPs": r"Invalid\newline PEPs",
    }

    if isinstance(data, pd.DataFrame):
        if collapse_index_columns:
            data = data.reset_index()

        data = data.rename(columns=rename_map)
        styler = data.style
    if isinstance(data, Styler):
        styler = data

    if collapse_index_columns:
        styler = styler.hide(axis=0)

    styler = styler.format_index(lambda x: x.replace("%", r"\%"), axis=1)
    if kwargs.get("column_header_bold", False):
        styler = styler.map_index(lambda _: "font-weight: bold;", axis=1)
    if kwargs.get("row_header_bold", False):
        styler = styler.map_index(lambda _: "font-weight: bold;", axis=0)

    if kwargs.get("escape_index", False):
        styler = styler.format_index(escape="latex", axis=0)
    if kwargs.get("escape_columns", False):
        styler = styler.format_index(escape="latex", axis=1)

    kwargs_styler = {k: v for k, v in kwargs.items() if k in list(inspect.signature(styler.to_latex).parameters.keys())}

    latex_str = styler.to_latex(**kwargs_styler)

    latex_str = latex_str.replace(
        "\\centering",
        "\\centering\n\\sisetup{separate-uncertainty=true,multi-part-units=single,table-align-uncertainty=true}",
    )
    return latex_str
