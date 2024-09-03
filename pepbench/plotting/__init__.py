"""Module for plotting signals and results of PEP benchmarks."""

from pepbench.plotting._base_plotting import (
    plot_signals,
    plot_signals_from_challenge_results,
    plot_signals_with_reference_labels,
    plot_signals_with_reference_pep,
)

__all__ = [
    "plot_signals",
    "plot_signals_with_reference_labels",
    "plot_signals_from_challenge_results",
    "plot_signals_with_reference_pep",
]
