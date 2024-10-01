from typing import Optional, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
from fau_colors import cmaps
from matplotlib import pyplot as plt

from pepbench.data_handling import get_data_for_algo
from pepbench.data_handling._data_handling import get_performance_metric
from pepbench.plotting._utils import _get_fig_ax

_ylabel_mapping = {
    "pep_ms": "PEP [ms]",
    "rr_interval_ms": "RR-Interval [ms]",
    "error_per_sample_ms": "Error [ms]",
    "absolute_error_per_sample_ms": "Absolute Error [ms]",
    "absolute_relative_error_per_sample_percent": "Absolute Relative Error [%]",
}

_xlabel_mapping = {"phase": "Phase", "participant": "Participant", "condition": "Condition"}

_algo_level_mapping = {
    "q_wave_algorithm": "Q-Wave Detection",
    "b_point_algorithm": "B-Point Detection",
    "outlier_correction_algorithm": "Outlier Correction",
}

_algorithm_mapping = {
    "b-point-reference": "Reference B-Point",
    "q-wave-reference": "Reference Q-Wave",
    "dwt-neurokit": "DWT NeuroKit",
}
_algorithm_mapping.update(**{f"r-peak-diff-{i}-ms": f"R-Peak - {i} ms" for i in np.arange(32, 44, 2)})
_algorithm_mapping.update(
    **{
        "second-derivative": "Debski1993.",
        "third-derivative": "Arbol2017",
        "straight-line": "Drost2022",
        "multiple-conditions": "Forouzanfar2018",
    }
)
_algorithm_mapping.update(
    **{"none": "No Correction", "linear-interpolation": "Linear Interpolation", "autoregression": "Forouzanfar2018"}
)

_metric_mapping = {
    "absolute_error_per_sample_ms": "Absolute Error [ms]",
    "error_per_sample_ms": "Error [ms]",
    "relative_error_per_sample_percent": "Relative Error [%]",
}


__all__ = ["plot_reference_pep_boxplot", "plot_algorithm_performances_boxplot", "plot_error_residual"]


def plot_reference_pep_boxplot(
    data: pd.DataFrame, x: str, y: Optional[str] = "pep_ms", hue: Optional[str] = None, **kwargs: dict
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = _get_fig_ax(**kwargs)

    if hue is None:
        sns.boxplot(data=data.reset_index(), x=x, y=y, ax=ax)
    else:
        sns.boxplot(data=data.reset_index(), x=x, y=y, hue=hue, ax=ax)

    ax.set_ylabel(_ylabel_mapping[y])
    ax.set_xlabel(_xlabel_mapping[x])

    fig.tight_layout()
    return fig, ax


def plot_algorithm_performances_boxplot(
    data: pd.DataFrame, metric: str = "absolute_error_per_sample_ms", **kwargs
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = _get_fig_ax(**kwargs)

    data = get_performance_metric(data, metric)

    # create new index level which merges the algorithm levels
    data = data.reset_index()
    data = data.replace(_algorithm_mapping)
    data = data.assign(algorithm=data[_algo_level_mapping.keys()].apply(lambda x: "\n".join(x), axis=1))

    # filter kwargs for sns.boxplot
    kwargs_boxplot = {
        k: v for k, v in kwargs.items() if k in sns.boxplot.__code__.co_varnames + plt.boxplot.__code__.co_varnames
    }

    meanprops = _get_meanprops(**kwargs)

    sns.boxplot(data, x="algorithm", y=metric, ax=ax, meanprops=meanprops, **kwargs_boxplot)
    ax.set_ylabel(
        _metric_mapping[metric],
        labelpad=12,
        # fontweight="bold",
    )
    ax.set_xlabel(
        "Algorithm Combination (Q-Wave Detection | B-Point Detection | Outlier Correction)",
        labelpad=12,
        # fontweight="bold",
    )

    fig.tight_layout()
    return fig, ax


def plot_error_residual(data: pd.DataFrame, algorithm: Sequence[str], **kwargs) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = _get_fig_ax(**kwargs)

    data = get_data_for_algo(data, algorithm)
    data = data["pep_ms"]
    data = data.dropna()

    algorithm = [f"{_algorithm_mapping[algo]}" for algo in algorithm]
    algorithm = " | ".join(algorithm)

    # filter kwargs for plt.scatter
    kwargs_scatter = {k: v for k, v in kwargs.items() if k in plt.scatter.__code__.co_varnames + ("color", "alpha")}
    pg.plot_blandaltman(x=data["reference"], y=data["estimated"], xaxis="x", ax=ax, **kwargs_scatter)

    ax.set_xlabel("Reference PEP [ms]")
    ax.set_ylabel("Reference - Estimated PEP [ms]")
    ax.set_title(
        f"Algorithm Combination:\n{algorithm}",
        fontdict={"fontweight": "bold"},
    )

    fig.tight_layout()
    return fig, ax


def _get_meanprops(**kwargs):
    if "meanprops" in kwargs:
        return kwargs["meanprops"]
    return {"marker": "X", "markerfacecolor": cmaps.fau[0], "markeredgecolor": cmaps.fau[0], "markersize": "6"}
