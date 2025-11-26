r"""
Top-level utilities for converting PEPBench result to presentation-ready tables and LaTeX output.

This package exposes helpers that take analysis output (pandas DataFrames produced
by the PEPBench pipeline) and produce nicely formatted DataFrames or LaTeX code
suitable for inclusion in manuscripts. Functions rename indices/columns using
internal mappings and apply consistent formatting appropriate for publication.

Functions
---------
:func:`~pepbench.export._latex.convert_to_latex`(data, collapse_index_columns: bool = False, **kwargs)
    Convert a `pandas.DataFrame` or `pandas.io.formats.style.Styler` to LaTeX.
    Provides sensible defaults for tabular formatting (siunitx, hrules, clines)
    and preserves any Styler formatting when given a `Styler`.

:func:`~pepbench.export._latex.create_algorithm_result_table`(data: :class:`~pandas.DataFrame`,
    collapse_algo_levels: bool = False) -> :class:`~pandas.DataFrame`
    Format algorithm-level result tables. Combines mean and standard deviation
    into `mean \pm std` strings, computes percentage invalid PEPs, and renames
    algorithm identifiers for presentation.

:func:`~pepbench.export._latex.create_nan_reason_table`(
    data: :class:`~pandas.DataFrame`,
    outlier_algos: Sequence[str] | None = None,
    use_short_names: bool = True,
) -> pandas.DataFrame
    Build a table of counts per reason for NaN / invalid PEPs per algorithm and
    outlier-correction algorithm.

:func:`~pepbench.export._latex.create_outlier_correction_table`(
    data: :class:`~pandas.DataFrame`,
    outlier_algos: :class:`~pandas.Sequence`[str] | None = None,
) -> :class:`~pandas.DataFrame`
    Reindex and format results grouped by baseline (b_point) and
    outlier-correction algorithm. Rename metrics and levels for presentation.


:func:`~pepbench.export._latex.create_reference_pep_table`(
    data: :class:`~pandas.DataFrame`,
) -> :class:`~pandas.DataFrame`
    Format reference PEP (`pep_ms`) statistics into two presentation columns:
    mean ± SD and range in milliseconds.

Notes
-----
- All functions expect DataFrames following the package's internal schema:
  multi-indexed rows for algorithm identifiers and multi-level columns for
  metrics where applicable.
- The module uses internal rename maps to produce human- and LaTeX-friendly
  labels; these mappings live in `pepbench.utils._rename_maps`.

"""

from pepbench.export._latex import (
    convert_to_latex,
    create_algorithm_result_table,
    create_nan_reason_table,
    create_outlier_correction_table,
    create_reference_pep_table,
)

__all__ = [
    "convert_to_latex",
    "create_algorithm_result_table",
    "create_nan_reason_table",
    "create_outlier_correction_table",
    "create_reference_pep_table",
]
