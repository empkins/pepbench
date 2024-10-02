from typing import Optional, Sequence, Union, Callable

import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
from fau_colors import cmaps, colors_all
from matplotlib import pyplot as plt

from pepbench.data_handling import get_data_for_algo
from pepbench.data_handling._data_handling import get_performance_metric, rr_interval_to_heart_rate
from pepbench.plotting._utils import _get_fig_ax

_ylabel_mapping = {
    "pep_ms": "PEP [ms]",
    "rr_interval_ms": "RR-Interval [ms]",
    "error_per_sample_ms": "Error [ms]",
    "absolute_error_per_sample_ms": "Absolute Error [ms]",
    "absolute_relative_error_per_sample_percent": "Absolute Relative Error [%]",
}

_xlabel_mapping = {
    "phase": "Phase",
    "participant": "Participant",
    "condition": "Condition",
}

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


__all__ = [
    "boxplot_reference_pep",
    "vioplot_reference_pep",
    "boxplot_algorithm_performance",
    "violinplot_algorithm_performance",
    "residual_plot_pep",
    "residual_plot_pep_subject",
    "residual_plot_pep_phase",
    "residual_plot_pep_heart_rate",
    "regplot_error_heart_rate",
    "regplot_pep_heart_rate",
]


def boxplot_reference_pep(
    data: pd.DataFrame, x: str, y: Optional[str] = "pep_ms", hue: Optional[str] = None, **kwargs: dict
) -> tuple[plt.Figure, plt.Axes]:
    return _plot_helper_reference_pep(data, sns.boxplot, x, y, hue, **kwargs)


def vioplot_reference_pep(
    data: pd.DataFrame, x: str, y: Optional[str] = "pep_ms", hue: Optional[str] = None, **kwargs: dict
) -> tuple[plt.Figure, plt.Axes]:
    return _plot_helper_reference_pep(data, sns.violinplot, x, y, hue, **kwargs)


def _plot_helper_reference_pep(
    data: pd.DataFrame,
    plot_func: Callable,
    x: str,
    y: str = "pep_ms",
    hue: Optional[str] = None,
    **kwargs: dict,
):
    fig, ax = _get_fig_ax(**kwargs)

    if hue is None:
        plot_func(data=data.reset_index(), x=x, y=y, ax=ax)
    else:
        plot_func(data=data.reset_index(), x=x, y=y, hue=hue, ax=ax)

    ax.set_ylabel(_ylabel_mapping[y])
    ax.set_xlabel(_xlabel_mapping[x])

    fig.tight_layout()
    return fig, ax


def boxplot_algorithm_performance(
    data: pd.DataFrame, metric: str = "absolute_error_per_sample_ms", **kwargs
) -> tuple[plt.Figure, plt.Axes]:
    return _plot_helper_algorithm_performance(data, sns.boxplot, metric, **kwargs)


def violinplot_algorithm_performance(
    data: pd.DataFrame, metric: str = "absolute_error_per_sample_ms", **kwargs
) -> tuple[plt.Figure, plt.Axes]:
    return _plot_helper_algorithm_performance(data, sns.violinplot, metric, **kwargs)


def _plot_helper_algorithm_performance(
    data: pd.DataFrame, plot_func: Callable, metric: str = "absolute_error_per_sample_ms", **kwargs
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = _get_fig_ax(**kwargs)

    data = get_performance_metric(data, metric)

    # create new index level which merges the algorithm levels
    data = data.reset_index()
    data = data.replace(_algorithm_mapping)
    data = data.assign(algorithm=data[_algo_level_mapping.keys()].apply(lambda x: "\n".join(x), axis=1))

    # filter kwargs for sns.boxplot
    kwargs_boxplot = {
        k: v
        for k, v in kwargs.items()
        if k
        in sns.boxplot.__code__.co_varnames
        + plt.boxplot.__code__.co_varnames
        + plt.violinplot.__code__.co_varnames
        + sns.violinplot.__code__.co_varnames
    }

    meanprops = _get_meanprops(**kwargs)

    plot_func(data, x="algorithm", y=metric, ax=ax, meanprops=meanprops, **kwargs_boxplot)
    ax.set_ylabel(
        _metric_mapping[metric],
        labelpad=12,
        # fontweight="bold",
    )
    ax.set_xlabel(
        "PEP Pipeline (Q-Wave Detection | B-Point Detection | Outlier Correction)",
        labelpad=12,
        # fontweight="bold",
    )

    fig.tight_layout()
    return fig, ax


def residual_plot_pep(data: pd.DataFrame, algorithm: Sequence[str], **kwargs) -> tuple[plt.Figure, plt.Axes]:
    kwargs.setdefault("color", cmaps.fau[0])
    kwargs.setdefault("alpha", 0.3)

    fig, ax = _get_fig_ax(**kwargs)

    data = get_data_for_algo(data, algorithm)
    data = data["pep_ms"]
    data = data.dropna()

    # filter kwargs for plt.scatter
    kwargs_scatter = {k: v for k, v in kwargs.items() if k in plt.scatter.__code__.co_varnames + ("color", "alpha")}
    pg.plot_blandaltman(x=data["reference"], y=data["estimated"], xaxis="x", ax=ax, **kwargs_scatter)

    ax.set_xlabel("Reference PEP [ms]")
    ax.set_ylabel("Reference - Estimated PEP [ms]")
    ax.set_title(
        f"PEP Pipeline:\n{_pep_pipeline_to_str(algorithm)}",
        fontdict={"fontweight": "bold"},
    )

    fig.tight_layout()
    return fig, ax


def residual_plot_pep_subject(data: pd.DataFrame, algorithm: Sequence[str], **kwargs) -> tuple[plt.Figure, plt.Axes]:
    kwargs.setdefault("base_color", "Spectral")
    return _residual_plot_error_detailed_helper(data, algorithm, "participant", **kwargs)


def residual_plot_pep_phase(data: pd.DataFrame, algorithm: Sequence[str], **kwargs) -> tuple[plt.Figure, plt.Axes]:
    kwargs.setdefault("base_color", f"blend:{colors_all.fau},{colors_all.tech_light}")
    return _residual_plot_error_detailed_helper(data, algorithm, "phase", **kwargs)


def residual_plot_pep_heart_rate(
    data: pd.DataFrame, algorithm: Sequence[str], bins: Optional[Union[int, str, Sequence[int]]] = 10, **kwargs
) -> tuple[plt.Figure, plt.Axes]:
    data = rr_interval_to_heart_rate(data)
    histogram, bin_edges = np.histogram(data["heart_rate_bpm"].dropna(), bins=bins)

    # add category for heart rate
    data = data.assign(
        heart_rate_range=pd.cut(
            data["heart_rate_bpm"],
            bins=bin_edges,
            labels=[f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges) - 1)],
        )
    )
    data = data.set_index("heart_rate_range", append=True)
    return _residual_plot_error_detailed_helper(data, algorithm, "heart_rate_range", **kwargs)


def _residual_plot_error_detailed_helper(
    data: pd.DataFrame, algorithm: Sequence[str], grouper: str, **kwargs
) -> tuple[plt.Figure, plt.Axes]:
    kwargs.setdefault("alpha", 0.8)
    rect = kwargs.pop("rect", (0, 0, 0.85, 1))
    # create new color palette based on the base color with the length of the number of participants
    n_colors = data.index.get_level_values(grouper).nunique()
    base_color = kwargs.pop("base_color", "Spectral")
    palette = sns.color_palette(base_color, n_colors=n_colors)

    # use residual plot to only plot mean and confidence interval of all data;
    # manually plot the scatter plot using participant as hue variable afterwards
    kwargs_new = kwargs.copy()
    kwargs_new.update(**{"alpha": 0.0})
    fig, ax = residual_plot_pep(data, algorithm, **kwargs_new)

    data = get_data_for_algo(data, algorithm)

    data = data["pep_ms"].reset_index()
    data = data.assign(x=data["reference"], y=data["reference"] - data["estimated"])

    # filter kwargs for plt.scatter
    kwargs_scatter = {
        k: v for k, v in kwargs.items() if k in sns.scatterplot.__code__.co_varnames + plt.scatter.__code__.co_varnames
    }
    sns.scatterplot(data=data, x="x", y="y", hue=grouper, **kwargs_scatter, palette=palette)

    ax.legend().remove()
    fig.legend(title=" ".join([s.capitalize() for s in grouper.split("_")]), loc="upper right")
    fig.tight_layout(rect=rect)

    return fig, ax


def regplot_error_heart_rate(
    data: pd.DataFrame, algorithm: Sequence[str], error_metric: str = "error_per_sample_ms", **kwargs
) -> tuple[plt.Figure, plt.Axes]:
    kwargs.setdefault("color", cmaps.tech[0])
    kwargs.setdefault("scatter_kws", {"alpha": 0.3})
    kwargs.setdefault("line_kws", {"color": cmaps.fau[0], "alpha": 0.8})
    fig, ax = _get_fig_ax(**kwargs)

    data = rr_interval_to_heart_rate(data)
    data = get_data_for_algo(data, algorithm)
    data = data.droplevel(level=-1, axis=1)

    sns.regplot(data=data.reset_index(), x="heart_rate_bpm", y=error_metric, ax=ax, **kwargs)

    ax.set_xlabel("Heart Rate [bpm]")
    ax.set_ylabel(_ylabel_mapping[error_metric])
    ax.set_title("PEP Pipeline:\n" + _pep_pipeline_to_str(algorithm), fontweight="bold")

    return fig, ax


def regplot_pep_heart_rate(
    data: pd.DataFrame, algorithm: Sequence[str], use_reference: bool = True, **kwargs
) -> tuple[plt.Figure, plt.Axes]:
    kwargs.setdefault("color", cmaps.tech[0])
    kwargs.setdefault("scatter_kws", {"alpha": 0.3})
    kwargs.setdefault("line_kws", {"color": cmaps.fau[0], "alpha": 0.8})
    fig, ax = _get_fig_ax(**kwargs)

    data = rr_interval_to_heart_rate(data)
    data = get_data_for_algo(data, algorithm)
    pep_ms = data[("pep_ms", "reference")] if use_reference else data[("pep_ms", "estimated")]
    heart_rate_bpm = data[["heart_rate_bpm"]]
    data = pd.concat([pep_ms, heart_rate_bpm], axis=1).droplevel(level=-1, axis=1)

    sns.regplot(data=data.reset_index(), x="heart_rate_bpm", y="pep_ms", ax=ax, **kwargs)
    ax.set_xlabel("Heart Rate [bpm]")
    ax.set_ylabel("PEP [ms]")

    if use_reference:
        title = "PEP Reference"
    else:
        title = "PEP Pipeline:\n" + _pep_pipeline_to_str(algorithm)
    ax.set_title(title, fontweight="bold")

    fig.tight_layout()

    return fig, ax


def _get_meanprops(**kwargs):
    if "meanprops" in kwargs:
        return kwargs["meanprops"]
    return {"marker": "X", "markerfacecolor": cmaps.fau[0], "markeredgecolor": cmaps.fau[0], "markersize": "6"}


def _pep_pipeline_to_str(pipeline: Sequence[str]) -> str:
    return " | ".join([_algorithm_mapping[algo] for algo in pipeline])
