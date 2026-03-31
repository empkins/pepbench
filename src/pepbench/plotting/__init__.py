"""High-level plotting interface for PEPBench.

This module re-exports plotting utilities and submodules used to create
visualizations for PEP benchmark signals, reference annotations and
algorithm results. It provides a convenient top-level import point so
users can access commonly used plotting functionality as
``pepbench.plotting`` (for example, ``from pepbench.plotting import plot_signals``).

Submodules
----------
algorithms
    Visualizations and helpers for plotting algorithm outputs and comparisons.
results
    Tools for plotting benchmark result summaries and challenge outputs.

Functions
---------
:func:`~pepbench.plotting.plot_signals`
    Plot one or more raw signals with common styling and layout options.
:func:`~pepbench.plotting.plot_signals_from_challenge_results`
    Plot signals using challenge result objects as input.
:func:`~pepbench.plotting.plot_signals_with_algorithm_results`
    Plot signals together with one or more algorithm result overlays.
:func:`~pepbench.plotting.plot_signals_with_reference_labels`
    Plot signals with reference label annotations.
:func:`~pepbench.plotting.plot_signals_with_reference_pep`
    Plot signals with reference PEP (peak/feature) annotations.

Notes
-----
The actual plotting implementations live in
``pepbench.plotting._base_plotting`` and the submodules exported here.
This module only aggregates and re-exports those symbols for convenience.

Examples
--------
>>> from pepbench.plotting import plot_signals, algorithms
>>> plot_signals(signals)
>>> algorithms.plot_algorithm_comparison(results)

"""

from pepbench.plotting import algorithms, results
from pepbench.plotting._base_plotting import (
    plot_signals,
    plot_signals_from_challenge_results,
    plot_signals_with_algorithm_results,
    plot_signals_with_reference_labels,
    plot_signals_with_reference_pep,
)

__all__ = [
    "algorithms",
    "plot_signals",
    "plot_signals_from_challenge_results",
    "plot_signals_with_algorithm_results",
    "plot_signals_with_reference_labels",
    "plot_signals_with_reference_pep",
    "results",
]
