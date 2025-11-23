"""
Utilities for producing publication-ready tables and LaTeX from PEPBench results.

This module provides helpers to convert pipeline results (pandas DataFrames)
into presentation-ready DataFrames or LaTeX strings. Functions format numeric
metrics (means, standard deviations, ranges), aggregate and reindex results
by algorithm and outlier-correction method, and map internal metric/algorithm
identifiers to human- and LaTeX-friendly labels using the package's rename
mappings.

The functions are intended to operate on DataFrames produced by the PEPBench
analysis pipeline and assume the package's internal schema (multi-indexed
rows for algorithm identifiers and multi-level columns for metrics where
applicable). Several functions accept a `pandas.io.formats.style.Styler`
or return a `pandas.DataFrame` ready for rendering or further styling.

Functions
---------
convert_to_latex(data, collapse_index_columns: bool = False, **kwargs)
    Convert a `pandas.DataFrame` or `pandas.io.formats.style.Styler` to LaTeX.
    Sets sensible defaults for tabular formatting (siunitx, hrules, clines),
    preserves existing Styler formatting, and injects `\sisetup{...}` for
    separate-uncertainty rendering. If `collapse_index_columns` is True the
    index is reset and the row index is hidden in the produced LaTeX.

create_algorithm_result_table(data: pandas.DataFrame, collapse_algo_levels: bool = False) -> pandas.DataFrame
    Format algorithm-level result tables. Combines mean and standard deviation
    into `mean \pm std` strings for commonly exported metrics, computes and
    formats an "Invalid PEPs" column (count and percent), and renames index
    values using the package algorithm mapping. If `collapse_algo_levels` is
    True, multi-level algorithm indices are collapsed into a single string
    column.

create_nan_reason_table(data: pandas.DataFrame, outlier_algos: Sequence[str] | None = None, use_short_names: bool = True) -> pandas.DataFrame
    Build a table of counts per reason for NaN/invalid PEPs grouped by
    baseline (`b_point_algorithm`) and `outlier_correction_algorithm`. Returns
    integer counts with columns renamed to short or long reason labels. Missing
    outlier algorithms can be reindexed with `outlier_algos`.

create_outlier_correction_table(data: pandas.DataFrame, outlier_algos: Sequence[str] | None = None) -> pandas.DataFrame
    Reindex and format results grouped by baseline (`b_point_algorithm`) and
    outlier-correction algorithm. Renames metric columns and algorithm levels
    for consistent presentation and ensures a predictable index order when
    `outlier_algos` is provided.

create_reference_pep_table(data: pandas.DataFrame) -> pandas.DataFrame
    Format reference PEP (`pep_ms`) statistics into two presentation columns:
    "M ± SD [ms]" (mean ± standard deviation) and "Range [ms]" (min, max).
    Index level names are capitalized for display.

Parameters
----------
data : pandas.DataFrame or pandas.io.formats.style.Styler
    Analysis results produced by the PEPBench pipeline. Specific functions
    may expect particular column layouts (e.g. multi-level columns with
    `('metric', 'mean')`, `('metric', 'std')`, `('Invalid PEPs', 'total')`,
    or `('nan_reason', 'estimated')`).
collapse_index_columns : bool, optional
    When converting to LaTeX, collapse and hide index columns (default False).
outlier_algos : Sequence[str] or None, optional
    Desired ordering / reindexing for outlier-correction algorithm rows.
use_short_names : bool, optional
    Use short labels for NaN reasons when True (default True).

Returns
-------
str or pandas.DataFrame
    `convert_to_latex` returns a LaTeX string. Other functions return a
    `pandas.DataFrame` formatted for presentation.

Notes
-----
- The module uses internal rename maps from `pepbench.utils._rename_maps` to
  produce human- and LaTeX-friendly labels for algorithms, metrics, and
  NaN reasons.
- Input DataFrames are expected to follow the package's internal schema:
  multi-index rows for algorithm identifiers (e.g. `b_point_algorithm`,
  `outlier_correction_algorithm`) and multi-level columns for metrics where
  applicable.
- `convert_to_latex` inspects the `Styler.to_latex` signature and forwards
  only supported keyword arguments; defaults such as `siunitx` and `hrules`
  are applied unless overridden.

"""

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
    """Format reference PEP statistics for presentation.

    Combines mean and standard deviation for the `pep_ms` metric into a
    single "M ± SD" style column and creates a range column with integer
    min/max values. Index level names are capitalized for display.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame produced by the analysis pipeline containing multi-level
        columns for `pep_ms` (expected keys: `('pep_ms', 'mean')`,
        `('pep_ms', 'std')`, `('pep_ms', 'min')`, `('pep_ms', 'max')`).

    Returns
    -------
    pandas.DataFrame
        A two-column DataFrame with columns ``"M ± SD [ms]"`` and
        ``"Range [ms]"`` suitable for LaTeX rendering or further styling.

    Notes
    -----
    - The function expects the input to contain the named `pep_ms` subcolumns.
    - Output column names are plain presentation labels (not LaTeX-escaped).

    Examples
    --------
    >>> tbl = create_reference_pep_table(df)
    """
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
    """Produce a presentation-ready table of algorithm-level results.

    Formats common numeric metrics by combining mean and standard deviation
    into ``mean \pm std`` strings, computes a human-readable "Invalid PEPs"
    column (count and percent), and renames index values using the package's
    algorithm mapping. Optionally collapse multi-level algorithm index
    levels into a single string column.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame indexed by algorithm identifiers and containing metric
        subcolumns (e.g. `('Mean Absolute Error [ms]', 'mean')`,
        `('Mean Absolute Error [ms]', 'std')`, etc.).
    collapse_algo_levels : bool, optional
        If True, collapse multi-level algorithm index levels into a single
        display column (default False).

    Returns
    -------
    pandas.DataFrame
        Formatted DataFrame with human-readable metric columns and renamed
        index values ready for conversion to LaTeX or other presentation.

    Notes
    -----
    - If the input contains an ``Invalid PEPs`` column, the function expects
      subcolumns `('Invalid PEPs', 'total')` and `('Total PEPs', 'total')`.
    - Index renaming uses ``_algorithm_mapping`` and level name mapping uses
      ``_algo_level_mapping`` from the module.

    Examples
    --------
    >>> table = create_algorithm_result_table(df, collapse_algo_levels=True)
    """
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
    """
    Reindex and format results grouped by baseline and outlier-correction algorithm.

    Ensures a predictable ordering of rows per baseline (``b_point_algorithm``)
    and reindexes each baseline group to include the provided ``outlier_algos``
    (if not None). Renames metric columns and algorithm index levels for
    consistent presentation.

    Parameters
    ----------
    data : pandas.DataFrame
        Grouped analysis results indexed by at least ``b_point_algorithm`` and
        ``outlier_correction_algorithm`` levels.
    outlier_algos : Sequence[str] or None, optional
        Desired ordering / reindexing for the ``outlier_correction_algorithm``
        level. If None, the original ordering is preserved.

    Returns
    -------
    pandas.DataFrame
        Reindexed and renamed DataFrame ready for display or further styling.

    Notes
    -----
    - The function will rename columns using ``_metric_mapping`` and index
      values using ``_algorithm_mapping`` from the module.
    """
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
    """
    Build a table of counts per NaN / invalid PEP reason per algorithm.

    Extracts the ``('nan_reason', 'estimated')`` column, counts occurrences
    grouped by ``b_point_algorithm`` and ``outlier_correction_algorithm``,
    and returns an integer count table. Columns are optionally renamed to
    short or long human-friendly labels.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing a ``('nan_reason', 'estimated')`` column.
    outlier_algos : Sequence[str] or None, optional
        If provided, reindex the ``outlier_correction_algorithm`` level to
        include these algorithms in the given order.
    use_short_names : bool, optional
        If True, rename reason columns using the short mapping
        (default True).

    Returns
    -------
    pandas.DataFrame
        Integer-count DataFrame (dtype ``Int64``) indexed by algorithm levels
        with columns representing NaN reasons.

    Notes
    -----
    - Missing combinations are filled with zero counts.
    - Index renaming uses ``_algorithm_mapping`` and level name mapping uses
      ``_algo_level_mapping``.
    """
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
    """
    Convert a DataFrame or Styler to a LaTeX table string with sensible defaults.

    Applies module default options for LaTeX rendering (siunitx, hrules,
    clines, centering) and preserves any existing Styler formatting. If a
    plain DataFrame is provided, it is converted to a `Styler` before rendering.
    Only keyword arguments supported by `Styler.to_latex` are forwarded.

    Parameters
    ----------
    data : pandas.DataFrame or pandas.io.formats.style.Styler
        Data to render as LaTeX. If a DataFrame is provided, it will be
        converted to a `Styler` and column names will be renamed using a
        small presentation mapping.
    collapse_index_columns : bool, optional
        When True, reset the index (so index columns become regular columns)
        and hide the row index in the produced LaTeX (default False).
    **kwargs : dict
        Additional keyword arguments passed to `Styler.to_latex`. Unrecognized
        keys are ignored.

    Returns
    -------
    str
        LaTeX string produced by `Styler.to_latex` with an injected `\sisetup`
        call to enable separate-uncertainty rendering.

    Notes
    -----
    - The function renames a small set of column headers to LaTeX-friendly
      labels prior to styling.
    - The `Styler.to_latex` signature is inspected and only supported
      parameters are forwarded.
    """
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
