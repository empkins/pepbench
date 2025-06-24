"""Module for plotting PEP benchmark results."""

import inspect
from collections.abc import Callable, Sequence
from typing import Any

import biopsykit as bp
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from fau_colors.v2021 import cmaps, colors_all
from matplotlib import pyplot as plt

from pepbench.data_handling import get_data_for_algo
from pepbench.data_handling._data_handling import get_performance_metric, get_reference_data
from pepbench.plotting._base_plotting import _plot_blandaltman, _plot_paired
from pepbench.plotting._utils import _get_fig_ax, _get_fig_axs, _remove_duplicate_legend_entries
from pepbench.utils._rename_maps import (
    _algo_level_mapping,
    _algorithm_mapping,
    _metric_mapping,
    _xlabel_mapping,
    _ylabel_mapping,
)

__all__ = [
    "boxplot_algorithm_performance",
    "boxplot_reference_pep",
    "histplot_heart_rate",
    "paired_plot_error_outlier_correction",
    "paired_plot_error_pep_pipeline",
    "plot_q_wave_detection_waveform_detailed_comparison",
    "regplot_error_bmi",
    "regplot_error_heart_rate",
    "regplot_pep_heart_rate",
    "residual_plot_pep",
    "residual_plot_pep_bmi",
    "residual_plot_pep_heart_rate",
    "residual_plot_pep_participant",
    "residual_plot_pep_phase",
    "violinplot_algorithm_performance",
    "violinplot_reference_pep",
]

from pepbench.utils._types import str_t


def boxplot_reference_pep(
    data: pd.DataFrame, x: str, y: str | None = "pep_ms", hue: str | None = None, **kwargs: dict
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a boxplot of reference PEP values.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        Dataframe containing the data to plot.
    x : str
        Column name of the x-axis data to plot, typically the different phases or conditions of the experiment.
    y: str, optional
        Column name of the y-axis data to plot. Default: "pep_ms".
    hue: str, optional
        Column name to plot different hues, if desired. Default: None.
    kwargs : Any
        Additional keyword arguments to pass to the plotting function.

    Returns
    -------
    :class:`~matplotlib.figure.Figure`, :class:`~matplotlib.axes.Axes`
        Figure and axes of the plot.

    """
    return _plot_helper_reference_pep(data, bp.plotting.feature_boxplot, x, y, hue, **kwargs)


def violinplot_reference_pep(
    data: pd.DataFrame, x: str, y: str | None = "pep_ms", hue: str | None = None, **kwargs: dict
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a violin of reference PEP values.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        Dataframe containing the data to plot.
    x : str
        Column name of the x-axis data to plot, typically the different phases or conditions of the experiment.
    y: str, optional
        Column name of the y-axis data to plot. Default: "pep_ms".
    hue: str, optional
        Column name to plot different hues, if desired. Default: None.
    kwargs : Any
        Additional keyword arguments to pass to the plotting function.

    Returns
    -------
    :class:`~matplotlib.figure.Figure`, :class:`~matplotlib.axes.Axes`
        Figure and axes of the plot.

    """
    return _plot_helper_reference_pep(data, sns.violinplot, x, y, hue, **kwargs)


def _plot_helper_reference_pep(
    data: pd.DataFrame,
    plot_func: Callable,
    x: str,
    y: str = "pep_ms",
    hue: str | None = None,
    **kwargs: dict,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = _get_fig_ax(kwargs)

    rect = kwargs.pop("rect", (0, 0, 0.85, 1))
    title = kwargs.pop("title", None)

    meanprops = _get_meanprops(**kwargs)

    if hue is None:
        plot_func(data=data.reset_index(), x=x, y=y, ax=ax, meanprops=meanprops, **kwargs)
    else:
        plot_func(data=data.reset_index(), x=x, y=y, hue=hue, meanprops=meanprops, ax=ax, **kwargs)

    ax.set_ylabel(_ylabel_mapping[y])
    ax.set_xlabel(_xlabel_mapping[x])

    if title is not None:
        ax.set_title(title, fontweight="bold")

    fig.tight_layout(rect=rect)
    return fig, ax


def boxplot_algorithm_performance(
    data: pd.DataFrame, metric: str = "absolute_error_per_sample_ms", **kwargs: dict
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a boxplot of a performance metric for different algorithms.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        Dataframe containing the data to plot.
    metric : str, optional
        Algorithm performance metric to plot. Default: "absolute_error_per_sample_ms"
    kwargs : Any
        Additional keyword arguments to pass to the plotting function.

    Returns
    -------
    :class:`~matplotlib.figure.Figure`, :class:`~matplotlib.axes.Axes`
        Figure and axes of the plot.

    """
    return _plot_helper_algorithm_performance(data, sns.boxplot, metric, **kwargs)


def violinplot_algorithm_performance(
    data: pd.DataFrame, metric: str = "absolute_error_per_sample_ms", **kwargs: dict
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a violinplot of a performance metric for different algorithms.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        Dataframe containing the data to plot.
    metric : str, optional
        Algorithm performance metric to plot. Default: "absolute_error_per_sample_ms"
    kwargs : Any
        Additional keyword arguments to pass to the plotting function.

    Returns
    -------
    :class:`~matplotlib.figure.Figure`, :class:`~matplotlib.axes.Axes`
        Figure and axes of the plot.

    """
    return _plot_helper_algorithm_performance(data, sns.violinplot, metric, **kwargs)


def _plot_helper_algorithm_performance(
    data: pd.DataFrame, plot_func: Callable, metric: str = "absolute_error_per_sample_ms", **kwargs: dict
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = _get_fig_ax(kwargs)
    rect = kwargs.pop("rect", (0, 0, 1, 1))

    data = get_performance_metric(data, metric)

    # create new index level which merges the algorithm levels
    data = data.reset_index()
    data = data.dropna().astype({metric: "float"})
    data = data.replace(_algorithm_mapping)

    algo_levels = [s for s in data.columns if s in _algo_level_mapping]
    data = data.assign(algorithm=data[algo_levels].apply(lambda x: "\n".join(x), axis=1))

    # filter kwargs for sns.boxplot
    if "boxplot" in plot_func.__name__:
        kwargs_plot = {
            k: v
            for k, v in kwargs.items()
            if k
            in list(inspect.signature(sns.boxplot).parameters.keys())
            + list(inspect.signature(plt.boxplot).parameters.keys())
        }
        kwargs_plot["meanprops"] = _get_meanprops(**kwargs)
    elif "violin" in plot_func.__name__:
        kwargs_plot = {
            k: v for k, v in kwargs.items() if k in list(inspect.signature(sns.violinplot).parameters.keys())
        }
    else:
        raise ValueError(f"Unknown plot function: {plot_func.__name__}")

    plot_func(data, x="algorithm", y=metric, hue="algorithm", ax=ax, **kwargs_plot)
    ax.set_ylabel(_metric_mapping[metric])

    xlabel = _format_pep_pipeline(algo_levels)

    ax.set_xlabel(
        xlabel,
        labelpad=12,
        # fontweight="bold",
    )

    if "title" in kwargs:
        ax.set_title(kwargs.pop("title"), fontweight="bold")

    fig.tight_layout(rect=rect)
    return fig, ax


def residual_plot_pep(data: pd.DataFrame, algorithm: str_t, **kwargs: dict) -> tuple[plt.Figure, plt.Axes]:
    """Plot a residual plot of PEP values for a specific algorithm.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        Dataframe containing the data to plot.
    algorithm : str or list of str
        Name of the algorithm (or list of algorithm names, if a pipeline is used) to plot.
    kwargs : Any
        Additional keyword arguments to pass to the plotting function.

    Returns
    -------
    :class:`~matplotlib.figure.Figure`, :class:`~matplotlib.axes.Axes`
        Figure and axes of the plot

    """
    kwargs.setdefault("color", cmaps.fau[0])
    kwargs.setdefault("alpha", 0.3)

    show_upper_limit = kwargs.pop("show_upper_limit", False)

    if isinstance(algorithm, str):
        algorithm = [algorithm]

    fig, ax = _get_fig_ax(kwargs)

    algo_levels = [s for s in data.index.names if s in _algo_level_mapping]

    data = get_data_for_algo(data, algorithm)

    data = data["pep_ms"]
    data = data.dropna()

    _plot_blandaltman(x=data["reference"], y=data["estimated"], xaxis="x", ax=ax, **kwargs)

    title = _format_title(algo_levels, algorithm)

    ax.set_xlabel("Reference PEP [ms]")
    ax.set_ylabel("Reference - Estimated PEP [ms]")
    ax.set_title(title, fontdict={"fontweight": "bold"})

    if show_upper_limit:
        xvals = np.arange(0, int(ax.get_ylim()[1]), 1)
        ax.plot(xvals, xvals, color=cmaps.wiso[0], ls="--")

    fig.tight_layout()
    return fig, ax


def residual_plot_pep_participant(data: pd.DataFrame, algorithm: str_t, **kwargs: dict) -> tuple[plt.Figure, plt.Axes]:
    """Plot a residual plot of PEP values for a specific algorithm, grouped by participant.

    Each participant is represented by a different color. The name of the participant column is assumed to be
    "participant".

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        Dataframe containing the data to plot.
    algorithm : str or list of str
        Name of the algorithm (or list of algorithm names, if a pipeline is used) to plot.
    kwargs : Any
        Additional keyword arguments to pass to the plotting function.


    Returns
    -------
    :class:`~matplotlib.figure.Figure`, :class:`~matplotlib.axes.Axes`
        Figure and axes of the plot

    """
    kwargs.setdefault("base_color", "Spectral")
    kwargs.setdefault("show_legend", False)
    kwargs.setdefault("rect", (0, 0, 1, 1))
    return _residual_plot_error_detailed_helper(data, algorithm, "participant", **kwargs)


def residual_plot_pep_phase(
    data: pd.DataFrame, algorithm: Sequence[str], **kwargs: dict
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a residual plot of PEP values for a specific algorithm, grouped by experimental phase.

    Each experimental phase is represented by a different color. The name of the phase column is assumed to be "phase".

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        Dataframe containing the data to plot.
    algorithm : str or list of str
        Name of the algorithm (or list of algorithm names, if a pipeline is used) to plot.
    kwargs : Any
        Additional keyword arguments to pass to the plotting function.


    Returns
    -------
    :class:`~matplotlib.figure.Figure`, :class:`~matplotlib.axes.Axes`
        Figure and axes of the plot

    """
    kwargs.setdefault("base_color", f"blend:{colors_all.fau},{colors_all.tech_light}")
    return _residual_plot_error_detailed_helper(data, algorithm, "phase", **kwargs)


def residual_plot_pep_heart_rate(
    data: pd.DataFrame, algorithm: Sequence[str], bins: int | str | Sequence[int] | None = 10, **kwargs: dict
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a residual plot of PEP values for a specific algorithm, grouped by heart rate bins.

    Each heart rate bin is represented by a different color. The name of the heart rate column is assumed to be
    "heart_rate_bpm".

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        Dataframe containing the data to plot.
    algorithm : str or list of str
        Name of the algorithm (or list of algorithm names, if a pipeline is used) to plot.
    bins : int or str
        Number of bins to use for the heart rate histogram using :func:`~numpy.histogram`. If `bins` is an int,
        it defines the number of equal-width bins in the given range (10, by default). If `bins` is a sequence, it
        defines a monotonically increasing array of bin edges, including the rightmost edge, allowing for
        non-uniform bin widths. If `bins` is a string, it defines the method used to calculate the
        optimal bin width, as defined by :func:`~numpy.histogram_bin_edges`.
        See also :func:`~numpy.histogram` for more information.
        Default: 10.
    kwargs : Any
        Additional keyword arguments to pass to the plotting function.

    Returns
    -------
    :class:`~matplotlib.figure.Figure`, :class:`~matplotlib.axes.Axes`
        Figure and axes of the plot

    """
    kwargs.setdefault("rect", (0, 0, 0.85, 1))
    kwargs.setdefault("base_color", "Spectral_r")

    histogram, bin_edges = np.histogram(data["heart_rate_bpm"].dropna(), bins=bins)

    kwargs["num_groups"] = len(bin_edges) - 1

    # add category for heart rate
    data = data.assign(
        heart_rate_range=pd.cut(
            data[("heart_rate_bpm", "estimated")],
            bins=bin_edges,
            labels=[f"{int(bin_edges[i])}-{int(bin_edges[i + 1])}" for i in range(len(bin_edges) - 1)],
        )
    )
    data = data.set_index("heart_rate_range", append=True)
    return _residual_plot_error_detailed_helper(data, algorithm, "heart_rate_range", **kwargs)


def residual_plot_pep_bmi(
    data: pd.DataFrame, algorithm: Sequence[str], bins: int | str | Sequence[int] | None = 4, **kwargs: dict
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a residual plot of PEP values for a specific algorithm, grouped by BMI bins.

    Each BMI bin is represented by a different color. The name of the BMI column is assumed to be "BMI".

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        Dataframe containing the data to plot.
    algorithm : str or list of str
        Name of the algorithm (or list of algorithm names, if a pipeline is used) to plot.
    bins : int or str
        Number of bins to use for the heart rate histogram using :func:`~numpy.histogram`. If `bins` is an int,
        it defines the number of equal-width bins in the given range (4, by default). If `bins` is a sequence, it
        defines a monotonically increasing array of bin edges, including the rightmost edge, allowing for
        non-uniform bin widths. If `bins` is a string, it defines the method used to calculate the
        optimal bin width, as defined by :func:`~numpy.histogram_bin_edges`.
        See also :func:`~numpy.histogram` for more information.
        Default: 4.
    kwargs : Any
        Additional keyword arguments to pass to the plotting function.

    Returns
    -------
    :class:`~matplotlib.figure.Figure`, :class:`~matplotlib.axes.Axes`
        Figure and axes of the plot

    """
    kwargs.setdefault("rect", (0, 0, 0.85, 1))
    kwargs.setdefault("base_color", "Spectral_r")

    histogram, bin_edges = np.histogram(data["BMI"].dropna(), bins=bins)
    kwargs["num_groups"] = len(bin_edges) - 1

    # add category for heart rate
    data = data.assign(
        bmi_range=pd.cut(
            data[("BMI", "estimated")],
            bins=bin_edges,
            labels=[f"{int(bin_edges[i])}-{int(bin_edges[i + 1])}" for i in range(len(bin_edges) - 1)],
        )
    )
    data = data.set_index("bmi_range", append=True)
    return _residual_plot_error_detailed_helper(data, algorithm, "bmi_range", **kwargs)


def residual_plot_pep_age(
    data: pd.DataFrame, algorithm: Sequence[str], bins: int | str | Sequence[int] | None = 5, **kwargs: dict
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a residual plot of PEP values for a specific algorithm, grouped by age bins.

    Each age bin is represented by a different color. The name of the age column is assumed to be "Age".

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        Dataframe containing the data to plot.
    algorithm : str or list of str
        Name of the algorithm (or list of algorithm names, if a pipeline is used) to plot.
    bins : int or str
        Number of bins to use for the age histogram using :func:`~numpy.histogram`. If `bins` is an int,
        it defines the number of equal-width bins in the given range (5, by default). If `bins` is a sequence, it
        defines a monotonically increasing array of bin edges, including the rightmost edge, allowing for
        non-uniform bin widths. If `bins` is a string, it defines the method used to calculate the
        optimal bin width, as defined by :func:`~numpy.histogram_bin_edges`.
        See also :func:`~numpy.histogram` for more information.
        Default: 5.
    kwargs : Any
        Additional keyword arguments to pass to the plotting function.

    Returns
    -------
    :class:`~matplotlib.figure.Figure`, :class:`~matplotlib.axes.Axes`
        Figure and axes of the plot

    """
    kwargs.setdefault("rect", (0, 0, 0.85, 1))
    kwargs.setdefault("base_color", "Spectral_r")

    histogram, bin_edges = np.histogram(data["Age"].dropna(), bins=bins)
    kwargs["num_groups"] = len(bin_edges) - 1

    # add category for age
    data = data.assign(
        age_range=pd.cut(
            data[("Age", "estimated")],
            bins=bin_edges,
            labels=[f"{int(bin_edges[i])}-{int(bin_edges[i + 1])}" for i in range(len(bin_edges) - 1)],
        )
    )
    data = data.set_index("age_range", append=True)
    return _residual_plot_error_detailed_helper(data, algorithm, "age_range", **kwargs)


def _residual_plot_error_detailed_helper(
    data: pd.DataFrame, algorithm: str_t, grouper: str, **kwargs: dict
) -> tuple[plt.Figure, plt.Axes]:
    kwargs.setdefault("alpha", 0.8)
    rect = kwargs.pop("rect", (0, 0, 0.90, 1))
    # create a new color palette based on the base color with the length of the number of groups
    n_colors = kwargs.pop("num_groups", data.index.get_level_values(grouper).nunique())
    base_color = kwargs.pop("base_color", "Spectral")
    palette = sns.color_palette(base_color, n_colors=n_colors)
    show_upper_limit = kwargs.pop("show_upper_limit", False)
    show_legend = kwargs.pop("show_legend", True)

    fig, ax = _get_fig_ax(kwargs)

    # use residual plot to only plot mean and confidence interval of all data;
    # manually plot the scatter plot using participant as hue variable afterwards
    kwargs_new = kwargs.copy()
    kwargs_new.update(alpha=0.0)

    data_scatter = get_data_for_algo(data, algorithm)

    data_scatter = data_scatter["pep_ms"].reset_index()
    data_scatter = data_scatter.assign(
        x=data_scatter["reference"], y=data_scatter["reference"] - data_scatter["estimated"]
    )

    # filter kwargs for plt.scatter
    kwargs_scatter = {
        k: v
        for k, v in kwargs.items()
        if k
        in list(inspect.signature(sns.scatterplot).parameters.keys())
        + list(inspect.signature(plt.scatter).parameters.keys())
    }

    sns.scatterplot(data=data_scatter, x="x", y="y", hue=grouper, ax=ax, **kwargs_scatter, palette=palette)

    fig, ax = residual_plot_pep(data, algorithm, ax=ax, **kwargs_new)

    if show_upper_limit:
        xvals = np.arange(0, int(ax.get_ylim()[1]), 1)
        ax.plot(xvals, xvals, color=cmaps.wiso[0], ls="--")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend().remove()
    handles, labels = _remove_duplicate_legend_entries(handles, labels)
    if show_legend:
        fig.legend(
            handles=handles,
            labels=labels,
            title=" ".join([s.capitalize() for s in grouper.split("_")]),
            loc="upper right",
        )

    fig.tight_layout(rect=rect)

    return fig, ax


def _add_corr_coeff(data: pd.DataFrame, x: str, y: str, ax: plt.Axes, **kwargs: dict) -> None:
    kwargs.setdefault("x_coord", 0.95)
    kwargs.setdefault("y_coord", 0.95)

    corr = pg.corr(data[x], data[y])
    s = f"r = {corr['r'].iloc[0]:.2f}, p {_format_p_value(corr['p-val'].iloc[0])}"
    prefix = kwargs.get("prefix")
    if prefix:
        s = f"{prefix}: " + s

    ax.text(
        x=kwargs["x_coord"],
        y=kwargs["y_coord"],
        s=s,
        transform=ax.transAxes,
        fontsize="medium",
        verticalalignment="top",
        horizontalalignment="right",
    )


def histplot_heart_rate(data: pd.DataFrame, hue: str | None = None, **kwargs: dict) -> tuple[plt.Figure, plt.Axes]:
    """Plot a histogram of heart rate values.

    The heart rate is assumed to be in the column "heart_rate_bpm".

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        Dataframe containing the data to plot.
    hue : str, optional
        Column name to plot different hues, if desired. Default: None.
    kwargs : Any
        Additional keyword arguments to pass to the plotting function. See :func:`~seaborn.histplot` for more
        information.

    Returns
    -------
    :class:`~matplotlib.figure.Figure`, :class:`~matplotlib.axes.Axes`
        Figure and axes of the plot

    """
    kwargs.setdefault("stat", "percent")
    kwargs.setdefault("kde", True)
    show_legend = kwargs.pop("show_legend", True)
    # legend_loc = kwargs.pop("legend_loc", "upper right")

    rect_default = (0, 0, 0.85, 1) if show_legend else (0, 0, 1, 1)

    rect = kwargs.pop("rect", rect_default)

    fig, ax = _get_fig_ax(kwargs)
    ax = sns.histplot(data=data.reset_index(), x="heart_rate_bpm", hue=hue, ax=ax, **kwargs)

    ax.set_xlabel("Heart Rate [bpm]")

    fig.tight_layout(rect=rect)
    # if show_legend and hue is not None:
    #    fig.legend(title=hue.capitalize(), handles=handles, labels=labels, loc=legend_loc)

    return fig, ax


def regplot_pep_heart_rate(
    data: pd.DataFrame,
    algorithm: str_t | None = None,
    use_reference: bool = False,
    add_corr_coeff: bool = False,
    groupby: str | None = None,
    **kwargs: dict,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a regression plot of PEP values against heart rate.

    The PEP values are assumed to be in the column "pep_ms" and the heart rate values are assumed to be in the column
    "heart_rate_bpm".

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        Dataframe containing the data to plot.
    algorithm : str or list of str, optional
        Name of the algorithm (or list of algorithm names, if a pipeline is used) to plot.
        If ``use_reference`` is True, this parameter is ignored. If ``use_reference`` is False,
        this parameter is required.
    use_reference : bool, optional
        ``True`` to use the reference data, ``False`` to use the results of the algorithm (pipeline)
        specified in ``algorithm``. Default: ``False``.
    add_corr_coeff : bool, optional
        ``True`` to add the correlation coefficient to the plot, ``False`` otherwise. Default: ``False``.
    groupby : str, optional
        Column name to group the data by. If specified, the data will be grouped by this column and a repeated-measures
        regression plot will be created using :func:`~pingouin.plot_rm_corr`. ``None`` by default, which means that no
        grouping is applied and a standard regression plot is created using :func:`~seaborn.regplot`.

    kwargs : Any
        Additional keyword arguments to pass to the plotting function. See :func:`~seaborn.regplot` for more
        information.

    Raises
    ------
    ValueError
        If ``use_reference`` is False and ``algorithm`` is not specified.

    Returns
    -------
    :class:`~matplotlib.figure.Figure`, :class:`~matplotlib.axes.Axes`
        Figure and axes of the plot

    """
    if groupby is None:
        kwargs.setdefault("color", cmaps.tech[0])
        kwargs.setdefault("line_kws", {"color": cmaps.tech_dark[0], "alpha": 0.8})

    kwargs.setdefault("scatter_kws", {"alpha": 0.4})
    fig, ax = _get_fig_ax(kwargs)

    if use_reference:
        data = get_reference_data(data)
    else:
        if algorithm is None:
            raise ValueError("If `use_reference` is False, `algorithm` must be specified.")
        data = get_data_for_algo(data, algorithm)

    if data.columns.nlevels > 1:
        data = data.xs("estimated", level=-1, axis=1)

    # filter kwargs for sns.regplot
    kwargs_regplot = {
        k: v
        for k, v in kwargs.items()
        if k
        in list(inspect.signature(sns.regplot).parameters.keys())
        + list(inspect.signature(plt.scatter).parameters.keys())
        + list(inspect.signature(plt.plot).parameters.keys())
    }

    if groupby is not None:
        palette = iter(kwargs.get("palette", cmaps.faculties_light))
        palette_line = iter(kwargs.get("palette_line", cmaps.faculties_dark))
        data.groupby(groupby, group_keys=False).apply(
            lambda df: sns.regplot(
                data=df.reset_index(),
                x="heart_rate_bpm",
                y="pep_ms",
                ax=ax,
                color=next(palette),
                line_kws={"color": next(palette_line), "alpha": 0.8},
                **kwargs_regplot,
                label=df.name,
            )
        )

        if add_corr_coeff:
            handles, labels = ax.get_legend_handles_labels()
            corr_vals = {}
            for name, group in data.groupby(groupby):
                corr = pg.corr(group["heart_rate_bpm"], group["pep_ms"])
                corr_vals[name] = f"r = {corr['r'].iloc[0]:.2f}, p {_format_p_value(corr['p-val'].iloc[0])}"

            labels = [f"{name}: {corr_vals[name]}" for name in labels]
            fig.legend(
                handles=handles,
                labels=labels,
                title=kwargs.get("legend_title", groupby.capitalize()),
                loc=kwargs.get("legend_loc", "upper right"),
            )
    else:
        sns.regplot(data=data.reset_index(), x="heart_rate_bpm", y="pep_ms", ax=ax, **kwargs_regplot)
        if add_corr_coeff:
            _add_corr_coeff(data, x="heart_rate_bpm", y="pep_ms", ax=ax)

    ax.set_xlabel("Heart Rate [bpm]")
    ax.set_ylabel("PEP [ms]")

    title = "PEP Reference" if use_reference else "PEP Pipeline:\n" + _pep_pipeline_to_str(algorithm)
    ax.set_title(title, fontweight="bold")

    fig.tight_layout(rect=kwargs.get("rect", (0, 0, 1, 1)))

    return fig, ax


def regplot_error_heart_rate(
    data: pd.DataFrame,
    algorithm: str_t,
    error_metric: str = "error_per_sample_ms",
    add_corr_coeff: bool = False,
    **kwargs: dict,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a regression plot of a PEP estimation *error* metric against heart rate.

    The error metric is assumed to be in the column specified by the parameter ``error_metric`` and the heart rate
    values are assumed to be in the column "heart_rate_bpm".

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        Dataframe containing the data to plot.
    algorithm : str or list of str
        Name of the algorithm (or list of algorithm names, if a pipeline is used) to plot.
    error_metric : str, optional
        Error metric to plot. Default: "error_per_sample_ms".
    add_corr_coeff : bool, optional
        ``True`` to add the correlation coefficient to the plot, ``False`` otherwise. Default: ``False``.
    kwargs : Any
        Additional keyword arguments to pass to the plotting function. See :func:`~seaborn.regplot` for more
        information.

    Returns
    -------
    :class:`~matplotlib.figure.Figure`, :class:`~matplotlib.axes.Axes`
        Figure and axes of the plot

    """
    kwargs.setdefault("color", cmaps.tech[0])
    kwargs.setdefault("scatter_kws", {"alpha": 0.3})
    kwargs.setdefault("line_kws", {"color": cmaps.fau[0], "alpha": 0.8})
    fig, ax = _get_fig_ax(kwargs)

    if isinstance(algorithm, str):
        algorithm = [algorithm]
    algo_levels = [s for s in data.index.names if s in _algo_level_mapping]

    data = get_data_for_algo(data, algorithm)

    data = data.reindex(["estimated", "metric"], level=-1, axis=1)
    data = data.droplevel(level=-1, axis=1)

    sns.regplot(data=data.reset_index(), x="heart_rate_bpm", y=error_metric, ax=ax, **kwargs)

    if add_corr_coeff:
        _add_corr_coeff(data, x="heart_rate_bpm", y=error_metric, ax=ax)

    ax.set_xlabel("Heart Rate [bpm]")
    ax.set_ylabel(_ylabel_mapping[error_metric])

    if len(algo_levels) == 1:
        title = f"{_algo_level_mapping[algo_levels[0]]}: {_pep_pipeline_to_str(algorithm)}"
    else:
        title = f"PEP Pipeline:\n{_pep_pipeline_to_str(algorithm)}"
    ax.set_title(title, fontweight="bold")

    return fig, ax


def regplot_error_bmi(
    data: pd.DataFrame,
    algorithm: str_t,
    error_metric: str = "error_per_sample_ms",
    add_corr_coeff: bool = False,
    **kwargs: dict,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a regression plot of a PEP estimation *error* metric against BMI.

    The error metric is assumed to be in the column specified by the parameter ``error_metric`` and the BMI values
    are assumed to be in the column "BMI".

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        Dataframe containing the data to plot.
    algorithm : str or list of str
        Name of the algorithm (or list of algorithm names, if a pipeline is used) to plot.
    error_metric : str, optional
        Error metric to plot. Default: "error_per_sample_ms".
    add_corr_coeff : bool, optional
        ``True`` to add the correlation coefficient to the plot, ``False`` otherwise. Default: ``False``.
    kwargs : Any
        Additional keyword arguments to pass to the plotting function. See :func:`~seaborn.regplot` for more
        information.

    Returns
    -------
    :class:`~matplotlib.figure.Figure`, :class:`~matplotlib.axes.Axes`
        Figure and axes of the plot

    """
    kwargs.setdefault("color", cmaps.tech[0])
    kwargs.setdefault("scatter_kws", {"alpha": 0.3})
    kwargs.setdefault("line_kws", {"color": cmaps.fau[0], "alpha": 0.8})
    fig, ax = _get_fig_ax(kwargs)

    if isinstance(algorithm, str):
        algorithm = [algorithm]
    algo_levels = [s for s in data.index.names if s in _algo_level_mapping]

    data = get_data_for_algo(data, algorithm)

    data = data.reindex(["estimated", "metric"], level=-1, axis=1)
    data = data.droplevel(level=-1, axis=1)

    sns.regplot(data=data.reset_index(), x="BMI", y=error_metric, ax=ax, **kwargs)

    if add_corr_coeff:
        _add_corr_coeff(data, x="BMI", y=error_metric, ax=ax)

    ax.set_xlabel("BMI [kg/m²]")
    ax.set_ylabel(_ylabel_mapping[error_metric])

    if len(algo_levels) == 1:
        title = f"{_algo_level_mapping[algo_levels[0]]}: {_pep_pipeline_to_str(algorithm)}"
    else:
        title = f"PEP Pipeline:\n{_pep_pipeline_to_str(algorithm)}"
    ax.set_title(title, fontweight="bold")

    return fig, ax


def regplot_error_age(
    data: pd.DataFrame,
    algorithm: str_t,
    error_metric: str = "error_per_sample_ms",
    add_corr_coeff: bool = False,
    **kwargs: dict,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a regression plot of a PEP estimation *error* metric against age.

    The error metric is assumed to be in the column specified by the parameter ``error_metric`` and the age values
    are assumed to be in the column "age".

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        Dataframe containing the data to plot.
    algorithm : str or list of str
        Name of the algorithm (or list of algorithm names, if a pipeline is used) to plot.
    error_metric : str, optional
        Error metric to plot. Default: "error_per_sample_ms".
    add_corr_coeff : bool, optional
        ``True`` to add the correlation coefficient to the plot, ``False`` otherwise. Default: ``False``.
    kwargs : Any
        Additional keyword arguments to pass to the plotting function. See :func:`~seaborn.regplot` for more
        information.

    Returns
    -------
    :class:`~matplotlib.figure.Figure`, :class:`~matplotlib.axes.Axes`
        Figure and axes of the plot

    """
    kwargs.setdefault("color", cmaps.tech[0])
    kwargs.setdefault("scatter_kws", {"alpha": 0.3})
    kwargs.setdefault("line_kws", {"color": cmaps.fau[0], "alpha": 0.8})
    fig, ax = _get_fig_ax(kwargs)

    if isinstance(algorithm, str):
        algorithm = [algorithm]
    algo_levels = [s for s in data.index.names if s in _algo_level_mapping]

    data = get_data_for_algo(data, algorithm)

    data = data.reindex(["estimated", "metric"], level=-1, axis=1)
    data = data.droplevel(level=-1, axis=1)

    sns.regplot(data=data.reset_index(), x="Age", y=error_metric, ax=ax, **kwargs)

    if add_corr_coeff:
        _add_corr_coeff(data, x="Age", y=error_metric, ax=ax)

    ax.set_xlabel("Age [years]")
    ax.set_ylabel(_ylabel_mapping[error_metric])

    if len(algo_levels) == 1:
        title = f"{_algo_level_mapping[algo_levels[0]]}: {_pep_pipeline_to_str(algorithm)}"
    else:
        title = f"PEP Pipeline:\n{_pep_pipeline_to_str(algorithm)}"
    ax.set_title(title, fontweight="bold")

    return fig, ax


def paired_plot_error_outlier_correction(
    data: pd.DataFrame, outlier_algo_combis: Sequence[Sequence[str]], dv: str, **kwargs: dict[str, Any]
) -> tuple[plt.Figure, Sequence[plt.Axes]]:
    """Plot a paired plot of an error metric for different outlier correction algorithms.

    A paired plot is a repeated-measures boxplot that additionally shows the paired data points and highlights the
    changes between the paired data points. This plot is useful to visualize the effect of different outlier correction
    algorithms on the error metric.

    See :func:``pingouin.plot_paired`` for more information.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        Dataframe containing the data to plot.
    outlier_algo_combis : list of string tuples
        List of tuples containing algorithm pipeline steps to compare against
    dv : str
        Dependent variable to plot. Can either be an error metric or a performance metric.
    kwargs : Any
        Additional keyword arguments to pass to the plotting function.

    Returns
    -------
    :class:`~matplotlib.figure.Figure`, list of :class:`~matplotlib.axes.Axes`
        Figure and axes of the plot

    """
    kwargs.setdefault("ncols", len(outlier_algo_combis))
    fig, axs = _get_fig_axs(kwargs)

    if "error" in dv:
        colors = kwargs.pop("colors", ["indianred", "grey", "green"])
    else:
        colors = kwargs.pop("colors", ["grey", "green", "indianred"])
    pointplot_kwargs = kwargs.pop("pointplot_kwargs", {"scale": 0.6, "marker": ".", "alpha": 0.5})

    for i, (ax, outlier_combi) in enumerate(zip(axs, outlier_algo_combis, strict=False)):
        data_plot = data.reindex(outlier_combi, level="outlier_correction_algorithm")
        data_plot = data_plot.unstack("outlier_correction_algorithm")
        eq_mask = ~(data_plot.diff(axis=1) == 0).any(axis=1)
        data_plot = data_plot.loc[eq_mask].stack(future_stack=True)

        # rename outlier correction algorithms
        data_plot = data_plot.rename(index=_algorithm_mapping)
        outlier_combi_format = [_algorithm_mapping[algo] for algo in outlier_combi]
        _plot_paired(
            data=data_plot,
            dv=dv,
            within="outlier_correction_algorithm",
            subject="id_concat",
            order=outlier_combi_format,
            colors=colors,
            pointplot_kwargs=pointplot_kwargs,
            ax=ax,
        )

        if i == 0:
            ax.set_ylabel(_ylabel_mapping[dv])
        ax.set_xlabel(_algo_level_mapping["outlier_correction_algorithm"])

        # xmax = len(outlier_combi) - 1 + 0.15
        ax.set_xlim([-0.15, 1.15])
    if "title" in kwargs:
        fig.suptitle(f"B-Point Algorithm: {_algorithm_mapping[kwargs.pop('title')]}", fontweight="bold")

    fig.tight_layout()

    return fig, axs


def paired_plot_error_pep_pipeline(
    data: pd.DataFrame, pep_pipelines: Sequence[Sequence[str]], dv: str, **kwargs: dict[str, Any]
) -> tuple[plt.Figure, Sequence[plt.Axes]]:
    """Plot a paired plot of an error metric for different PEP extraction pipelines.

    A paired plot is a repeated-measures boxplot that additionally shows the paired data points and highlights the
    changes between the paired data points. This plot is useful to visualize the differences between different PEP
    extraction pipelines.

    See :func:``pingouin.plot_paired`` for more information.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        Dataframe containing the data to plot.
    pep_pipelines : list of string tuples
        List of tuples with pipeline configurations to compare against
    dv : str
        Dependent variable to plot. Can either be an error metric or a performance metric.
    kwargs : Any
        Additional keyword arguments to pass to the plotting function.

    Returns
    -------
    :class:`~matplotlib.figure.Figure`, list of :class:`~matplotlib.axes.Axes`
        Figure and axes of the plot

    """
    kwargs.setdefault("ncols", len(pep_pipelines))
    fig, axs = _get_fig_axs(kwargs)
    if "error" in dv:
        colors = kwargs.pop("colors", ["#C50F3C", "#8C9FB1", "#7BB725"])
    else:
        colors = kwargs.pop("colors", ["#8C9FB1", "#7BB725", "#C50F3C"])
    pointplot_kwargs = kwargs.pop("pointplot_kwargs", {"scale": 0.6, "marker": ".", "alpha": 0.2})

    for i, (ax, pep_pipeline) in enumerate(zip(axs, pep_pipelines, strict=False)):
        pep_pipeline_format = ["_".join(p) for p in pep_pipeline]
        data_plot = data.reindex(pep_pipeline_format, level="pipeline")
        data_plot = data_plot.unstack("pipeline")
        eq_mask = ~(data_plot.diff(axis=1) == 0).any(axis=1)
        data_plot = data_plot.loc[eq_mask].stack(future_stack=True)

        # rename outlier correction algorithms
        _plot_paired(
            data=data_plot,
            dv=dv,
            within="pipeline",
            subject="id_concat",
            order=pep_pipeline_format,
            colors=colors,
            pointplot_kwargs=pointplot_kwargs,
            ax=ax,
        )

        if i == 0:
            ax.set_ylabel(_ylabel_mapping[dv])
        # ax.set_xlabel(_algo_level_mapping["outlier_correction_algorithm"])
        xmax = len(pep_pipeline_format) - 1 + 0.15
        ax.set_xlim([-0.15, xmax])
    if "title" in kwargs:
        fig.suptitle(f"B-Point Algorithm: {_algorithm_mapping[kwargs.pop('title')]}", fontweight="bold")

    fig.tight_layout()

    return fig, axs


def _get_meanprops(**kwargs: dict) -> dict:
    if "meanprops" in kwargs:
        return kwargs["meanprops"]
    return {"marker": "X", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "6"}


def _format_title(algo_levels: Sequence[str], algorithm: Sequence[str]) -> str:
    if len(algo_levels) == 1:
        title = f"{_algo_level_mapping[algo_levels[0]]}: {_pep_pipeline_to_str(algorithm)}"
    else:
        title = f"PEP Pipeline:\n{_pep_pipeline_to_str(algorithm)}"
    return title


def _pep_pipeline_to_str(pipeline: Sequence[str]) -> str:
    return " | ".join([_algorithm_mapping[algo] for algo in pipeline])


def _format_p_value(p_val: float) -> str:
    # sanitize p-value to ensure it's a float, then format it to 3 decimal places
    p_val = f"{float(p_val):.3f}"
    p_val = "< 0.001" if p_val == "0.000" else "= " + p_val
    return p_val


def _format_pep_pipeline(algo_levels: Sequence[str]) -> str:
    if algo_levels == 3:
        pipeline_str = f"PEP Pipeline ({' | '.join([_algo_level_mapping[algo_level] for algo_level in algo_levels])})"
    elif algo_levels == 1:
        pipeline_str = _algo_level_mapping[algo_levels[0]]
    else:
        pipeline_str = " | ".join([_algo_level_mapping[algo_level] for algo_level in algo_levels])

    return pipeline_str


def plot_q_wave_detection_waveform_detailed_comparison(
    datapoint_01: pd.DataFrame,
    datapoint_02: pd.DataFrame,
    base_plot_func: Callable,
    plot_func_01_params: dict,
    plot_func_02_params: dict,
    ax_inset_01_params: dict,
    ax_inset_02_params: dict,
    datapoint_01_name: str,
    datapoint_02_name: str,
) -> plt.Figure | Sequence[plt.Axes]:
    """Plot a detailed comparison of two Q-wave detection waveforms with insets.

    This function creates a figure with two subplots, each containing a waveform plot of a Q-wave detection result.
    Additionally, each subplot contains an inset with a zoomed-in view of the waveform. This function can be helpful
    to compare the effect of different ECG waveforms on the Q-wave detection result.

    Parameters
    ----------
    datapoint_01 : :class:`~pandas.DataFrame`
        Dataframe containing the first data point to plot.
    datapoint_02 : :class:`~pandas.DataFrame`
        Dataframe containing the second data point to plot.
    base_plot_func : Callable
        Base plotting function to use for the waveform plot. Should be one of the Q-wave plotting functions from
        ``pepbench.plotting.algorithms``.
    plot_func_01_params : dict
        Additional keyword arguments to pass to the plotting function for the first datapoint. Can include parameters
        such as ``normalize_time`` or ``use_tight``.
    plot_func_02_params : dict
        Additional keyword arguments to pass to the plotting function for the second datapoint. Can include parameters
        such as ``normalize_time`` or ``use_tight``.
    ax_inset_01_params : dict
        Parameters to pass for configuring the inset for the first datapoint.
        See :func:`~matplotlib.axes.Axes.inset_axes` for more information.
    ax_inset_02_params : dict
        Parameters to pass for configuring the inset for the second datapoint.
        See :func:`~matplotlib.axes.Axes.inset_axes` for more information.
    datapoint_01_name : str
        Name of the first datapoint to display in the plot.
    datapoint_02_name : str
        Name of the second datapoint to display in the plot.


    Returns
    -------
    :class:`~matplotlib.figure.Figure`, list of :class:`~matplotlib.axes.Axes`
        Figure and axes of the plot

    """
    fig, axs = plt.subplots(
        nrows=2,
        gridspec_kw={"left": 0.075, "bottom": 0.1, "top": 0.85, "right": 0.75, "hspace": 0.25},
        sharex=True,
        sharey=True,
    )

    plot_func_01_params.setdefault("normalize_time", True)
    plot_func_02_params.setdefault("normalize_time", True)
    plot_func_01_params.setdefault("use_tight", False)
    plot_func_02_params.setdefault("use_tight", False)

    base_plot_func(datapoint=datapoint_01, ax=axs[0], **plot_func_01_params)
    base_plot_func(datapoint=datapoint_02, ax=axs[1], **plot_func_02_params)

    xlim = axs[0].get_xlim()
    ylim = axs[0].get_ylim()

    xmin_01 = ax_inset_01_params.pop("xmin", 0)
    xmax_01 = ax_inset_01_params.pop("xmax", 1)

    ax_inset1 = axs[0].inset_axes(
        **ax_inset_01_params, yticks=[], yticklabels=[], xlim=(xmin_01, xmax_01), ylim=(ylim[0] + 0.1, ylim[1] - 0.1)
    )
    base_plot_func(datapoint=datapoint_01, ax=ax_inset1, **plot_func_01_params)

    ax_inset1.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
    ax_inset1.set_xlim(xmin_01, xmax_01)
    ax_inset1.set_xlabel(None)
    ax_inset1.set_ylabel(None)

    xmin_02 = ax_inset_02_params.pop("xmin", 0)
    xmax_02 = ax_inset_02_params.pop("xmax", 1)

    ax_inset2 = axs[1].inset_axes(
        **ax_inset_02_params, yticks=[], yticklabels=[], xlim=(xmin_02, xmax_02), ylim=(ylim[0] + 0.1, ylim[1] - 0.1)
    )
    base_plot_func(datapoint=datapoint_02, ax=ax_inset2, **plot_func_02_params)

    ax_inset2.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
    ax_inset2.set_xlim(xmin_02, xmax_02)
    ax_inset2.set_xlabel(None)
    ax_inset2.set_ylabel(None)

    axs[0].set_xlim(0.1, xlim[0] + 0.33 * (xlim[1] - xlim[0]))

    axs[0].indicate_inset_zoom(ax_inset1, edgecolor="black")
    axs[1].indicate_inset_zoom(ax_inset2, edgecolor="black")

    axs[0].text(x=0.5, y=1.05, s=datapoint_01_name, transform=axs[0].transAxes, fontdict={"fontweight": "bold"})
    axs[1].text(x=0.5, y=1.05, s=datapoint_02_name, transform=axs[1].transAxes, fontdict={"fontweight": "bold"})

    return fig, axs
