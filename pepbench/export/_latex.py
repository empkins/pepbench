import pandas as pd
from pepbench.utils._rename_maps import (
    _ylabel_mapping,
    _algorithm_mapping,
    _xlabel_mapping,
    _metric_mapping,
    _algo_level_mapping,
)

__all__ = ["create_reference_pep_table", "create_algorithm_result_table", "convert_to_latex"]


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


def create_algorithm_result_table(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    # Create a new DataFrame with formatted columns
    formatted_data = pd.DataFrame(index=data.index)
    formatted_data.index.names = [_algo_level_mapping[s] for s in formatted_data.index.names]

    export_cols_m_sd = ["Mean Absolute Error [ms]", "Mean Error [ms]", "Mean Absolute Relative Error [%]"]

    for key in export_cols_m_sd:
        formatted_data[key] = data.apply(lambda row: f"{row[(key, 'mean')]:.1f} \pm {row[(key, 'std')]:.1f}", axis=1)

    formatted_data[r"Invalid PEPs"] = data.apply(
        lambda row: f"{int(row[('Invalid PEPs', 'total')])} "
        rf"({(row[('Invalid PEPs', 'total')] / row[('Total PEPs', 'total')] * 100):.1f} \%)",
        axis=1,
    )

    # rename index values
    formatted_data = formatted_data.rename(index=_algorithm_mapping)

    return formatted_data


def convert_to_latex(data: pd.DataFrame, collapse_index_columns: bool = False, **kwargs):
    kwargs.setdefault("hrules", True)
    kwargs.setdefault("position", "ht")
    kwargs.setdefault("siunitx", True)
    kwargs.setdefault("clines", "skip-last;data")
    rename_map = {
        "Q-Peak Detection": "Algorithm",
        "Mean Absolute Error [ms]": r"\ac{MAE} [ms]",
        "Mean Error [ms]": r"\ac{ME} [ms]",
        "Mean Absolute Relative Error [%]": r"\ac{MARE} [%]",
        "Invalid PEPs": r"Invalid\newline PEPs",
    }

    if collapse_index_columns:
        data = data.reset_index()

    data = data.rename(columns=rename_map)

    styler = data.style
    if collapse_index_columns:
        styler = styler.hide(axis=0)

    styler = styler.format_index(lambda x: x.replace("%", r"\%"), axis=1)
    if kwargs.get("column_header_bold", False):
        styler = styler.applymap_index(lambda x: "font-weight: bold;", axis=1)
    if kwargs.get("row_header_bold", False):
        styler = styler.applymap_index(lambda x: "font-weight: bold;", axis=0)

    latex_str = styler.to_latex(
        column_format=kwargs.get("column_format"),
        hrules=kwargs.get("hrules"),
        position=kwargs.get("position"),
        siunitx=kwargs.get("siunitx"),
        position_float="centering",
        clines=kwargs.get("clines"),
        label=kwargs.get("label"),
        caption=kwargs.get("caption"),
        convert_css=True,
    )

    latex_str = latex_str.replace(
        "\\centering",
        "\\centering\n\\sisetup{separate-uncertainty=true,multi-part-units=single,table-align-uncertainty=true}",
    )
    return latex_str
