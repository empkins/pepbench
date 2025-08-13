"""Module for visualizing Q-peak detection and B-point detection algorithms."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from biopsykit.signals.ecg.event_extraction import (
    QPeakExtractionForouzanfar2018,
    QPeakExtractionMartinez2004Neurokit,
    QPeakExtractionSciPyFindPeaksNeurokit,
    QPeakExtractionVanLien2013,
)
from biopsykit.signals.icg.event_extraction import (
    BPointExtractionLozano2007LinearRegression,
    BPointExtractionLozano2007QuadraticRegression,
    BPointExtractionMiljkovic2022,
    BPointExtractionPale2021,
    BPointExtractionStern1985,
)
from fau_colors.v2021 import cmaps
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

from pepbench.algorithms.icg import (
    BPointExtractionArbol2017IsoelectricCrossings,
    BPointExtractionArbol2017SecondDerivative,
    BPointExtractionArbol2017ThirdDerivative,
    BPointExtractionDebski1993SecondDerivative,
    BPointExtractionDrost2022,
    BPointExtractionForouzanfar2018,
    BPointExtractionSherwood1990,
    CPointExtractionScipyFindPeaks,
)
from pepbench.datasets import BasePepDatasetWithAnnotations
from pepbench.plotting._base_plotting import _plot_signals_one_axis, plot_signals
from pepbench.plotting._utils import (
    _add_ecg_q_peaks,
    _add_ecg_r_peaks,
    _add_heartbeat_borders,
    _add_icg_b_points,
    _add_icg_c_points,
    _get_annotation_bbox_no_edge,
    _get_data,
    _get_fig_ax,
    _get_heartbeat_borders,
    _get_heartbeats,
    _get_legend_loc,
    _get_rect,
    _get_reference_labels,
    _handle_legend_one_axis,
    _handle_legend_two_axes,
    _sanitize_heartbeat_subset,
)

__all__ = [
    "plot_b_point_extraction_arbol2017_isoelectric_crossings",
    "plot_b_point_extraction_arbol2017_second_derivative",
    "plot_b_point_extraction_arbol2017_third_derivative",
    "plot_b_point_extraction_debski1993_second_derivative",
    "plot_b_point_extraction_drost2022",
    "plot_b_point_extraction_forouzanfar2018",
    "plot_b_point_extraction_lozano2007_linear_regression",
    "plot_b_point_extraction_lozano2007_quadratic_regression",
    "plot_b_point_extraction_miljkovic2022",
    "plot_b_point_extraction_pale2021",
    "plot_b_point_extraction_sherwood1990",
    "plot_b_point_extraction_stern1985",
    "plot_q_peak_extraction_forounzafar2018",
    "plot_q_peak_extraction_martinez2004_neurokit",
    "plot_q_peak_extraction_vanlien2013",
]


def plot_q_peak_extraction_martinez2004_neurokit(
    datapoint: BasePepDatasetWithAnnotations,
    *,
    heartbeat_subset: Sequence[int] | None = None,
    normalize_time: bool = False,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot example of Q-peak extraction using the Martinez et al. (2004) algorithm [1].

    The algorithm is implemented as :class:``QPeakExtractionMartinez2004Neurokit``.

    Parameters
    ----------
    datapoint : BasePepDatasetWithAnnotations
        Datapoint to plot.
    heartbeat_subset : list of int, optional
        List of heartbeat_ids to plot. If None, all heartbeats are plotted.
    normalize_time : bool, optional
        Whether to normalize the time axis to seconds, starting at 0, or not. Default: False
    kwargs : dict
        Additional keyword arguments to pass to the plotting functions. Examples are:
        --- General ---
        * ``fig``, ``ax``: :class:`matplotlib.figure.Figure`, :class:`matplotlib.axes.Axes`
            Existing Figure and Axes objects to plot on; If not provided, a new figure and axes are created.
        * ``figsize``: tuple
            Size of the figure.
        * ``legend_loc``: str
            Location of the legend
        * ``legend_outside``: bool
            Whether to place the legend outside the plot or not.
        * ``legend_orientation``: str
            Orientation of the legend, either "horizontal" or "vertical".
        * ``legend_max_cols``: int
            Maximum number of columns for the legend if ``legend_orientation`` is "horizontal".
        * ``rect``: tuple
            Rectangle coordinates for tight layout, i.e, the bounding box (x0, y0, x1, y1) that the subplots will fit
            into.
        * ``use_tight``: bool
            Whether to use tight layout or not. Default: True
        --- Heartbeat Borders ---
        * ``heartbeat_border_color``: str
            Color of the heartbeat borders.
        --- Q-Peaks ---
        * ``q_peak_marker``: str
            Marker style for Q-peaks.
        * ``q_peak_linestyle``: str
            Line style for Q-peaks.
        * ``q_peak_linewidth``: float
            Line width for Q-peaks.
        * ``q_peak_alpha``: float
            Alpha value for Q-peak vertical lines.


    Return
    ------
    fig : :class:`matplotlib.figure.Figure`
        Figure object.
    ax : :class:`matplotlib.axes.Axes`
        Axes object.

    Raises
    ------
    ValueError
        If the ECG data is too short for Q-peak detection (i.e., less than 4 seconds).


    See Also
    --------
    :class:``pepbench.algorithms.ecg.QPeakExtractionMartinez2004Neurokit``
        Algorithm implementation.


    References
    ----------
    .. [1] Martinez, J. P., Almeida, R., Olmos, S., Rocha, A. P., & Laguna, P. (2004). A wavelet-based ECG delineator:
        evaluation on standard databases. IEEE Transactions on Biomedical Engineering, 51(4), 570-581.
        https://doi.org/10.1109/TBME.2003.821031

    """
    kwargs.setdefault("legend_outside", True)
    kwargs.setdefault("legend_orientation", "horizontal")
    kwargs.setdefault("legend_loc", _get_legend_loc(kwargs))

    fig, ax = _get_fig_ax(kwargs)

    rect = _get_rect(kwargs)

    heartbeat_subset = _sanitize_heartbeat_subset(heartbeat_subset)
    ecg_data, _ = _get_data(datapoint, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset)

    if len(ecg_data) < 4 * datapoint.sampling_rate_ecg:
        raise ValueError("ECG data is too short for Q-peak detection. Please provide more heartbeats.")

    heartbeats = _get_heartbeats(datapoint, heartbeat_subset)
    heartbeat_borders = _get_heartbeat_borders(ecg_data, heartbeats)

    q_peak_algo = QPeakExtractionMartinez2004Neurokit()
    q_peak_algo.extract(ecg=ecg_data, heartbeats=heartbeats, sampling_rate_hz=datapoint.sampling_rate_ecg)

    q_peak_samples = q_peak_algo.points_["q_peak_sample"].dropna()
    q_peak_samples_reference = _get_reference_labels(datapoint, heartbeat_subset)["q_peaks"]

    _plot_signals_one_axis(
        datapoint=datapoint,
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        plot_icg=False,
        ax=ax,
        **kwargs,
    )

    _add_ecg_q_peaks(
        ecg_data,
        q_peak_samples_reference,
        ax=ax,
        q_peak_label="Reference Q-Peaks",
        q_peak_color=cmaps.med_dark[0],
        **kwargs,
    )
    _add_ecg_q_peaks(
        ecg_data,
        q_peak_samples,
        ax=ax,
        q_peak_label="Detected Q-Peaks",
        q_peak_color=cmaps.med[0],
        **kwargs,
    )

    _add_heartbeat_borders(heartbeat_borders, ax=ax, **kwargs)

    _handle_legend_one_axis(fig=fig, ax=ax, **kwargs)

    # check if figure is a figure (and not a SubFigure) and not already constrained layout
    if kwargs.get("use_tight", True):
        fig.tight_layout(rect=rect)

    return fig, ax


def plot_q_peak_extraction_scipy_findpeaks_neurokit(
    datapoint: BasePepDatasetWithAnnotations,
    *,
    heartbeat_subset: Sequence[int] | None = None,
    normalize_time: bool = False,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot example of Q-peak extraction using the SciPy FindPeaks algorithm provided by NeuroKit.

    The algorithm is implemented as :class:``QPeakExtractionSciPyFindPeaksNeurokit``.

    Parameters
    ----------
    datapoint : BasePepDatasetWithAnnotations
        Datapoint to plot.
    heartbeat_subset : list of int, optional
        List of heartbeat_ids to plot. If None, all heartbeats are plotted.
    normalize_time : bool, optional
        Whether to normalize the time axis to seconds, starting at 0, or not. Default: False
    kwargs : dict
        Additional keyword arguments to pass to the plotting functions. Examples are:
        --- General ---
        * ``fig``, ``ax``: :class:`matplotlib.figure.Figure`, :class:`matplotlib.axes.Axes`
            Existing Figure and Axes objects to plot on; If not provided, a new figure and axes are created.
        * ``figsize``: tuple
            Size of the figure.
        * ``legend_loc``: str
            Location of the legend
        * ``legend_outside``: bool
            Whether to place the legend outside the plot or not.
        * ``legend_orientation``: str
            Orientation of the legend, either "horizontal" or "vertical".
        * ``legend_max_cols``: int
            Maximum number of columns for the legend if ``legend_orientation`` is "horizontal".
        * ``rect``: tuple
            Rectangle coordinates for tight layout, i.e, the bounding box (x0, y0, x1, y1) that the subplots will fit
            into.
        * ``use_tight``: bool
            Whether to use tight layout or not. Default: True
        --- Heartbeat Borders ---
        * ``heartbeat_border_color``: str
            Color of the heartbeat borders.
        --- Q-Peaks ---
        * ``q_peak_marker``: str
            Marker style for Q-peaks.
        * ``q_peak_linestyle``: str
            Line style for Q-peaks.
        * ``q_peak_linewidth``: float
            Line width for Q-peaks.
        * ``q_peak_alpha``: float
            Alpha value for Q-peak vertical lines.

    Return
    ------
    fig : :class:`matplotlib.figure.Figure`
        Figure object.
    ax : :class:`matplotlib.axes.Axes`
        Axes object.

    See Also
    --------
    :class:``pepbench.algorithms.ecg.QPeakExtractionSciPyFindPeaksNeurokit``
        Algorithm implementation.


    Raises
    ------
    ValueError
        If the ECG data is too short for Q-peak detection (i.e., less than 4 seconds).

    """
    kwargs.setdefault("legend_outside", True)
    kwargs.setdefault("legend_orientation", "horizontal")
    kwargs.setdefault("legend_loc", _get_legend_loc(**kwargs))

    rect = _get_rect(**kwargs)

    heartbeat_subset = _sanitize_heartbeat_subset(heartbeat_subset)
    ecg_data, _ = _get_data(datapoint, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset)

    if len(ecg_data) < 4 * datapoint.sampling_rate_ecg:
        raise ValueError("ECG data is too short for Q-peak detection. Please provide more heartbeats.")

    heartbeats = _get_heartbeats(datapoint, heartbeat_subset)
    heartbeat_borders = _get_heartbeat_borders(ecg_data, heartbeats)

    q_peak_algo = QPeakExtractionSciPyFindPeaksNeurokit()
    q_peak_algo.extract(ecg=ecg_data, heartbeats=heartbeats, sampling_rate_hz=datapoint.sampling_rate_ecg)

    q_peak_samples = q_peak_algo.points_["q_peak_sample"].dropna()
    q_peak_samples_reference = _get_reference_labels(datapoint, heartbeat_subset)["q_peaks"]

    fig, ax = _plot_signals_one_axis(
        datapoint=datapoint,
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        plot_icg=False,
        **kwargs,
    )

    _add_ecg_q_peaks(
        ecg_data,
        q_peak_samples_reference,
        ax=ax,
        q_peak_label="Reference Q-Peaks",
        q_peak_color=cmaps.med_dark[0],
        **kwargs,
    )
    _add_ecg_q_peaks(
        ecg_data,
        q_peak_samples,
        ax=ax,
        q_peak_label="Detected Q-Peaks",
        q_peak_color=cmaps.med[0],
        **kwargs,
    )

    _add_heartbeat_borders(heartbeat_borders, ax=ax, **kwargs)

    _handle_legend_one_axis(fig=fig, ax=ax, **kwargs)
    fig.tight_layout(rect=rect)

    return fig, ax


def plot_q_peak_extraction_vanlien2013(
    datapoint: BasePepDatasetWithAnnotations,
    *,
    heartbeat_subset: Sequence[int] | None = None,
    normalize_time: bool = False,
    algo_params: dict | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot example of Q-peak extraction using the van Lien et al. (2013) algorithm [1].

    The algorithm is implemented as :class:``QPeakExtractionVanLien2013``.

    Parameters
    ----------
    datapoint : BasePepDatasetWithAnnotations
        Datapoint to plot.
    heartbeat_subset : list of int, optional
        List of heartbeat_ids to plot. If None, all heartbeats are plotted.
    normalize_time : bool, optional
        Whether to normalize the time axis to seconds, starting at 0, or not. Default: False
    algo_params : dict, optional
        Parameters passed to the algorithm instance.
        See :class:``pepbench.algorithms.ecg.QPeakExtractionVanLien2013`` for available parameters.
        Default: None (i.e., the default parameters of the algorithm are used).
    kwargs : dict
        Additional keyword arguments to pass to the plotting functions. Examples are:
        --- General ---
        * ``fig``, ``ax``: :class:`matplotlib.figure.Figure`, :class:`matplotlib.axes.Axes`
            Existing Figure and Axes objects to plot on; If not provided, a new figure and axes are created.
        * ``figsize``: tuple
            Size of the figure.
        * ``legend_loc``: str
            Location of the legend
        * ``legend_outside``: bool
            Whether to place the legend outside the plot or not.
        * ``legend_orientation``: str
            Orientation of the legend, either "horizontal" or "vertical".
        * ``legend_max_cols``: int
            Maximum number of columns for the legend if ``legend_orientation`` is "horizontal".
        * ``rect``: tuple
            Rectangle coordinates for tight layout, i.e, the bounding box (x0, y0, x1, y1) that the subplots will fit
            into.
        * ``use_tight``: bool
            Whether to use tight layout or not. Default: True
        --- Heartbeat Borders ---
        * ``heartbeat_border_color``: str
            Color of the heartbeat borders.
        --- R-Peaks ---
        * ``r_peak_marker``: str
            Marker style for R-peaks.
        * ``r_peak_linestyle``: str
            Line style for R-peaks.
        * ``r_peak_linewidth``: float
            Line width for R-peaks.
        * ``r_peak_alpha``: float
            Alpha value for the R-peak vertical lines
        * ``r_peak_plot_marker``: bool
            Whether to plot markers at the R-peaks or not.
        * ``r_peak_plot_vline``: bool
            Whether to plot vertical lines at the R-peaks or not.
        --- Q-Peaks ---
        * ``q_peak_marker``: str
            Marker style for Q-peaks.
        * ``q_peak_linestyle``: str
            Line style for Q-peaks.
        * ``q_peak_linewidth``: float
            Line width for Q-peaks.
        * ``q_peak_alpha``: float
            Alpha value for Q-peak vertical lines.


    Return
    ------
    fig : :class:`matplotlib.figure.Figure`
        Figure object.
    ax : :class:`matplotlib.axes.Axes`
        Axes object.

    Raises
    ------
    ValueError
        If the ECG data is too short for Q-peak detection (i.e., less than 4 heartbeats).

    See Also
    --------
    :class:``pepbench.algorithms.ecg.QPeakExtractionVanLien2013``
        Algorithm implementation.

    References
    ----------
    .. [1] Van Lien, R., Schutte, N. M., Meijer, J. H., & De Geus, E. J. C. (2013). Estimated preejection period (PEP)
        based on the detection of the R-peak and dZ/dt-min peaks does not adequately reflect the actual PEP across a
        wide range of laboratory and ambulatory conditions. International Journal of Psychophysiology, 87(1), 60-69.
        https://doi.org/10.1016/j.ijpsycho.2012.11.001


    """
    kwargs.setdefault("legend_outside", True)
    kwargs.setdefault("legend_orientation", "horizontal")
    kwargs.setdefault("legend_loc", _get_legend_loc(kwargs))
    rect = _get_rect(kwargs)

    if algo_params is None:
        algo_params = {}

    heartbeat_subset = _sanitize_heartbeat_subset(heartbeat_subset)
    ecg_data, _ = _get_data(datapoint, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset)

    heartbeats = _get_heartbeats(datapoint, heartbeat_subset)
    heartbeat_borders = _get_heartbeat_borders(ecg_data, heartbeats)

    q_peak_algo = QPeakExtractionVanLien2013(**algo_params)
    q_peak_algo.extract(ecg=ecg_data, heartbeats=heartbeats, sampling_rate_hz=datapoint.sampling_rate_ecg)

    r_peak_samples = heartbeats["r_peak_sample"].astype(int)
    q_peak_samples = q_peak_algo.points_["q_peak_sample"].astype(int)
    q_peak_samples_reference = _get_reference_labels(datapoint, heartbeat_subset)["q_peaks"]

    time_interval_ms = q_peak_algo.get_params()["time_interval_ms"]

    kwargs.setdefault("r_peak_linewidth", 2)
    kwargs.setdefault("r_peak_linestyle", "--")
    kwargs.setdefault("r_peak_marker", "X")
    kwargs.setdefault("q_peak_linewidth", 2)

    fig, ax = _plot_signals_one_axis(
        datapoint=datapoint,
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        plot_icg=False,
        **kwargs,
    )
    _add_heartbeat_borders(heartbeat_borders, ax=ax, **kwargs)
    _add_ecg_r_peaks(ecg_data, r_peak_samples, ax=ax, **kwargs)
    _add_ecg_q_peaks(
        ecg_data,
        q_peak_samples_reference,
        q_peak_label="Reference Q-Peaks",
        q_peak_color=cmaps.med_dark[0],
        ax=ax,
        **kwargs,
    )
    _add_ecg_q_peaks(
        ecg_data,
        q_peak_samples,
        q_peak_label="Detected Q-Peaks",
        ax=ax,
        **kwargs,
    )

    # get the maximum R-peak amplitude
    r_peak_amplitude = ecg_data.iloc[r_peak_samples].max()

    # draw arrow from R-peak to Q-peaks
    for r_peak, q_peak in zip(r_peak_samples, q_peak_samples, strict=False):
        x_q_peak = ecg_data.index[q_peak]
        x_r_peak = ecg_data.index[r_peak]
        y = r_peak_amplitude.iloc[0]
        middle_x = x_q_peak + (x_r_peak - x_q_peak) / 2
        # align text to the center of the array
        ax.annotate(
            "",
            xy=(x_q_peak, y),
            xytext=(x_r_peak, y),
            # align text to the center of the array
            arrowprops={"arrowstyle": "->", "color": cmaps.tech_dark[0], "lw": 2, "shrinkA": 0.0, "shrinkB": 0.0},
            ha="center",
            zorder=2,
        )
        ax.annotate(
            rf"$- {time_interval_ms}\,ms$",
            xy=(middle_x, y),
            xytext=(0, 12),
            textcoords="offset points",
            bbox=_get_annotation_bbox_no_edge(),
            ha="center",
        )

    _handle_legend_one_axis(fig=fig, ax=ax, **kwargs)

    old_ylims = ax.get_ylim()
    ax.set_ylim(old_ylims[0], 1.15 * old_ylims[1])

    fig.tight_layout(rect=rect)

    return fig, ax


def plot_q_peak_extraction_forounzafar2018(
    datapoint: BasePepDatasetWithAnnotations,
    *,
    heartbeat_subset: Sequence[int] | None = None,
    normalize_time: bool = False,
    algo_params: dict | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot example of Q-peak extraction using the Forouzanfar et al. (2018) algorithm [1].

    The algorithm is implemented as :class:``QPeakExtractionForouzanfar2018``.

    Parameters
    ----------
    datapoint : BasePepDatasetWithAnnotations
        Datapoint to plot.
    heartbeat_subset : list of int, optional
        List of heartbeat_ids to plot. If None, all heartbeats are plotted.
    normalize_time : bool, optional
        Whether to normalize the time axis to seconds, starting at 0, or not. Default: False
    algo_params : dict, optional
        Parameters passed to the algorithm instance.
        See :class:``pepbench.algorithms.ecg.QPeakExtractionForouzanfar2018`` for available parameters.
        Default: None (i.e., the default parameters of the algorithm are used).
    kwargs : dict
        Additional keyword arguments to pass to the plotting functions. Examples are:
        --- General ---
        * ``fig``, ``ax``: :class:`matplotlib.figure.Figure`, :class:`matplotlib.axes.Axes`
            Existing Figure and Axes objects to plot on; If not provided, a new figure and axes are created.
        * ``figsize``: tuple
            Size of the figure.
        * ``legend_loc``: str
            Location of the legend
        * ``legend_outside``: bool
            Whether to place the legend outside the plot or not.
        * ``legend_orientation``: str
            Orientation of the legend, either "horizontal" or "vertical".
        * ``legend_max_cols``: int
            Maximum number of columns for the legend if ``legend_orientation`` is "horizontal".
        * ``rect``: tuple
            Rectangle coordinates for tight layout, i.e, the bounding box (x0, y0, x1, y1) that the subplots will fit
            into.
        * ``use_tight``: bool
            Whether to use tight layout or not. Default: True
        --- Heartbeat Borders ---
        * ``heartbeat_border_color``: str
            Color of the heartbeat borders.
        --- R-Peaks ---
        * ``r_peak_marker``: str
            Marker style for R-peaks.
        * ``r_peak_linestyle``: str
            Line style for R-peaks.
        * ``r_peak_linewidth``: float
            Line width for R-peaks.
        * ``r_peak_alpha``: float
            Alpha value for the R-peak vertical lines
        * ``r_peak_plot_marker``: bool
            Whether to plot markers at the R-peaks or not.
        * ``r_peak_plot_vline``: bool
            Whether to plot vertical lines at the R-peaks or not.
        --- Q-Peaks ---
        * ``q_peak_marker``: str
            Marker style for Q-peaks.
        * ``q_peak_linestyle``: str
            Line style for Q-peaks.
        * ``q_peak_linewidth``: float
            Line width for Q-peaks.
        * ``q_peak_alpha``: float
            Alpha value for Q-peak vertical lines.


    Return
    ------
    fig : :class:`matplotlib.figure.Figure`
        Figure object.
    ax : :class:`matplotlib.axes.Axes`
        Axes object.

    See Also
    --------
    :class:``pepbench.algorithms.ecg.QPeakExtractionVanLien2013``
        Algorithm implementation.

    References
    ----------
    .. [1] Forouzanfar, M., Baker, F. C., De Zambotti, M., McCall, C., Giovangrandi, L., & Kovacs, G. T. A. (2018).
        Toward a better noninvasive assessment of preejection period: A novel automatic algorithm for B-point detection
        and correction on thoracic impedance cardiogram. Psychophysiology, 55(8), e13072.
        https://doi.org/10.1111/psyp.13072

    """
    fig, ax = plt.subplots(**kwargs)
    kwargs.setdefault("legend_outside", True)
    kwargs.setdefault("legend_orientation", "horizontal")
    kwargs.setdefault("legend_max_cols", 4)
    kwargs.setdefault("legend_loc", _get_legend_loc(kwargs))
    rect = _get_rect(kwargs)

    if algo_params is None:
        algo_params = {}

    heartbeat_subset = _sanitize_heartbeat_subset(heartbeat_subset)
    ecg_data, icg_data = _get_data(datapoint, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset)
    heartbeats = _get_heartbeats(datapoint, heartbeat_subset)
    heartbeat_borders = _get_heartbeat_borders(icg_data, heartbeats)

    q_peak_algo = QPeakExtractionForouzanfar2018(**algo_params)
    q_peak_algo.extract(ecg=ecg_data, heartbeats=heartbeats, sampling_rate_hz=datapoint.sampling_rate_ecg)

    q_peak_samples = q_peak_algo.points_["q_peak_sample"].dropna().astype(int)
    q_peak_samples_reference = _get_reference_labels(datapoint, heartbeat_subset)["q_peaks"]

    ecg_data = ecg_data.squeeze()
    _plot_signals_one_axis(
        df=ecg_data,
        ax=ax,
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        color=cmaps.fau[0],
        **kwargs,
    )

    _add_heartbeat_borders(heartbeats=heartbeat_borders, ax=ax, **kwargs)

    _add_ecg_q_peaks(
        ecg_data,
        q_peak_samples_reference,
        q_peak_label="Reference Q-Peaks",
        q_peak_color=cmaps.med_dark[0],
        ax=ax,
        **kwargs,
    )
    _add_ecg_q_peaks(
        ecg_data,
        q_peak_samples,
        q_peak_label="Detected Q-Peaks",
        ax=ax,
        **kwargs,
    )

    for _idx, row in heartbeats.iterrows():
        start = row["start_sample"]
        end = row["end_sample"]
        r_peak = row["r_peak_sample"]

        ecg_heartbeat = ecg_data.iloc[start:end]

        threshold = -1.2 * ecg_data.iloc[r_peak] / datapoint.sampling_rate_ecg

        # plot threshold per heartbeat
        ax.hlines(
            threshold,
            xmin=ecg_heartbeat.index[0],
            xmax=ecg_heartbeat.index[-1],
            color=cmaps.fau_dark[2],
            linestyle="--",
            linewidth=2,
            zorder=0,
            label=r"Threshold $(-1.2 \cdot \text{R-Peak} / f_{s})$",
        )

    _handle_legend_one_axis(fig=fig, ax=ax, **kwargs)

    fig.tight_layout(rect=rect)

    return fig, ax


def plot_b_point_extraction_stern1985(
    datapoint: BasePepDatasetWithAnnotations,
    *,
    heartbeat_subset: Sequence[int] | None = None,
    normalize_time: bool = False,
    algo_params: dict | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, Sequence[plt.Axes]]:
    """Plot example of B-point extraction using the Stern et al. (1985) algorithm [1].

    The algorithm is implemented as :class:``BPointExtractionStern1985``.

    Parameters
    ----------
    datapoint : BasePepDatasetWithAnnotations
        Datapoint to plot.
    heartbeat_subset : list of int, optional
        List of heartbeat_ids to plot. If None, all heartbeats are plotted.
    normalize_time : bool, optional
        Whether to normalize the time axis to seconds, starting at 0, or not. Default: False
    algo_params : dict, optional
        Parameters passed to the algorithm instance for C-point and B-point extraction.
        See :class:``pepbench.algorithms.icg.CPointExtractionScipyFindPeaks`` and
            :class:``pepbench.algorithms.icg.BPointExtractionStern1985`` for available parameters.
        Default: None (i.e., the default parameters of the algorithms are used).
    kwargs : dict
        Additional keyword arguments to pass to the plotting functions. Examples are:
        --- General ---
        * ``figsize``: tuple
            Size of the figure.
        * ``legend_loc``: str
            Location of the legend
        * ``legend_outside``: bool
            Whether to place the legend outside the plot or not.
        * ``legend_orientation``: str
            Orientation of the legend, either "horizontal" or "vertical".
        * ``legend_max_cols``: int
            Maximum number of columns for the legend if ``legend_orientation`` is "horizontal".
        * ``rect``: tuple
            Rectangle coordinates for tight layout, i.e, the bounding box (x0, y0, x1, y1) that the subplots will fit
            into.
        * ``use_tight``: bool
            Whether to use tight layout or not. Default: True
        --- Heartbeat Borders ---
        * ``heartbeat_border_color``: str
            Color of the heartbeat borders.
        --- R-Peaks ---
        * ``r_peak_marker``: str
            Marker style for R-peaks.
        * ``r_peak_linestyle``: str
            Line style for R-peaks.
        * ``r_peak_linewidth``: float
            Line width for R-peaks.
        * ``r_peak_alpha``: float
            Alpha value for the R-peak vertical lines
        * ``r_peak_plot_marker``: bool
            Whether to plot markers at the R-peaks or not.
        * ``r_peak_plot_vline``: bool
            Whether to plot vertical lines at the R-peaks or not.
        --- B-Points ---
        * ``b_point_marker``: str
            Marker style for B-points.
        * ``b_point_linestyle``: str
            Line style for B-points.
        * ``b_point_linewidth``: float
            Line width for B-points.
        * ``b_point_alpha``: float
            Alpha value for B-point vertical lines.


    Return
    ------
    fig : :class:`matplotlib.figure.Figure`
        Figure object.
    axs : list of :class:`matplotlib.axes.Axes`
        list of Axes objects, one for each subplot.

    See Also
    --------
    :class:``pepbench.algorithms.icg.BPointExtractionStern1985``
        Algorithm implementation.

    References
    ----------
    .. [1] Stern, H. C., Wolf, G. K., & Belz, G. G. (1985). Comparative measurements of left ventricular ejection time
        by mechano-, echo- and electrical impedance cardiography. Arzneimittel-Forschung, 35(10), 1582-1586.

    """
    fig, axs = plt.subplots(nrows=2, sharex=True, **kwargs)
    kwargs.setdefault("legend_outside", True)
    kwargs.setdefault("legend_orientation", "horizontal")
    kwargs.setdefault("legend_loc", _get_legend_loc(kwargs))
    kwargs.setdefault("legend_max_cols", 4)
    rect = _get_rect(kwargs)

    if algo_params is None:
        algo_params = {}

    heartbeat_subset = _sanitize_heartbeat_subset(heartbeat_subset)
    ecg_data, icg_data = _get_data(datapoint, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset)
    heartbeats = _get_heartbeats(datapoint, heartbeat_subset)
    heartbeat_borders = _get_heartbeat_borders(icg_data, heartbeats)

    algo_params_c_point = {key: val for key, val in algo_params.items() if key in ["window_c_correction"]}
    algo_params_b_point = {key: val for key, val in algo_params.items() if key not in algo_params_c_point}
    c_point_algo = CPointExtractionScipyFindPeaks(**algo_params_c_point)
    c_point_algo.extract(icg=icg_data, heartbeats=heartbeats, sampling_rate_hz=datapoint.sampling_rate_icg)

    b_point_algo = BPointExtractionStern1985(**algo_params_b_point)
    b_point_algo.extract(
        icg=icg_data, heartbeats=heartbeats, c_points=c_point_algo.points_, sampling_rate_hz=datapoint.sampling_rate_icg
    )

    icg_data = icg_data.squeeze()

    icg_2nd_der = np.gradient(icg_data)
    icg_2nd_der = pd.DataFrame(icg_2nd_der, index=icg_data.index, columns=["ICG 2nd Deriv. $(d^2Z/dt^2)$"])

    # compute zero crossings of the second derivative
    icg_2nd_der_zero_crossings = np.where(np.diff(np.signbit(icg_2nd_der.squeeze())))[0]

    c_point_samples = c_point_algo.points_["c_point_sample"].dropna().astype(int)
    b_point_samples_reference = _get_reference_labels(datapoint, heartbeat_subset)["b_points"]
    b_point_samples = b_point_algo.points_["b_point_sample"].dropna().astype(int)

    # get only the zero crossings between heartbeat start and c_point_sample
    zero_crossings_filtered = []
    for idx, row in heartbeats.iterrows():
        zero_crossings_filtered.append(
            icg_2nd_der_zero_crossings[
                (icg_2nd_der_zero_crossings > row["start_sample"])
                & (icg_2nd_der_zero_crossings < c_point_samples.loc[idx] - 1)
            ]
        )
    zero_crossings_filtered = np.concatenate(zero_crossings_filtered)

    _plot_signals_one_axis(
        df=icg_data,
        ax=axs[0],
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        color=cmaps.tech[0],
        **kwargs,
    )
    _plot_signals_one_axis(
        df=icg_2nd_der,
        ax=axs[1],
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        color=cmaps.tech_dark[0],
        **kwargs,
    )

    _add_heartbeat_borders(heartbeats=heartbeat_borders, ax=axs[0], **kwargs)
    _add_heartbeat_borders(heartbeats=heartbeat_borders, ax=axs[1], **kwargs)

    _add_icg_c_points(icg_data, c_point_samples, ax=axs[0], **kwargs)
    _add_icg_c_points(icg_data, c_point_samples, ax=axs[1], c_point_plot_marker=False, **kwargs)

    _add_icg_b_points(
        icg_data,
        b_point_samples_reference,
        ax=axs[0],
        b_point_label="Reference B-Points",
        b_point_color=cmaps.phil_dark[0],
        **kwargs,
    )

    _add_icg_b_points(
        icg_data,
        b_point_samples,
        ax=axs[0],
        b_point_label="Detected B-Points",
        **kwargs,
    )

    # plot the zero crossings of the second derivative
    _add_icg_b_points(
        icg_2nd_der,
        zero_crossings_filtered,
        ax=axs[1],
        b_point_label="Zero Crossings of $d^2Z/dt^2$ before C-Point",
        b_point_color=cmaps.phil[2],
        **kwargs,
    )

    # add zero line to second derivative plot
    axs[1].axhline(0, color="black", linestyle="--", linewidth=1, zorder=0)

    _handle_legend_two_axes(fig=fig, axs=axs, **kwargs)

    fig.align_ylabels()
    fig.tight_layout(rect=rect)

    return fig, axs


def plot_b_point_extraction_sherwood1990(
    datapoint: BasePepDatasetWithAnnotations,
    *,
    heartbeat_subset: Sequence[int] | None = None,
    normalize_time: bool = False,
    algo_params: dict | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot example of B-point extraction using the Sherwood et al. (1990) algorithm [1].

    The algorithm is implemented as :class:``BPointExtractionSherwood1990``.

    Parameters
    ----------
    datapoint : BasePepDatasetWithAnnotations
        Datapoint to plot.
    heartbeat_subset : list of int, optional
        List of heartbeat_ids to plot. If None, all heartbeats are plotted.
    normalize_time : bool, optional
        Whether to normalize the time axis to seconds, starting at 0, or not. Default: False
    algo_params : dict, optional
        Parameters passed to the algorithm instance for C-point and B-point extraction.
        See :class:``pepbench.algorithms.icg.CPointExtractionScipyFindPeaks`` and
            :class:``pepbench.algorithms.icg.BPointExtractionSherwood1990`` for available parameters.
        Default: None (i.e., the default parameters of the algorithms are used).
    kwargs : dict
        Additional keyword arguments to pass to the plotting functions. Examples are:
        --- General ---
        * ``fig``, ``ax``: :class:`matplotlib.figure.Figure`, :class:`matplotlib.axes.Axes`
            Existing Figure and Axes objects to plot on; If not provided, a new figure and axes are created.
        * ``figsize``: tuple
            Size of the figure.
        * ``legend_loc``: str
            Location of the legend
        * ``legend_outside``: bool
            Whether to place the legend outside the plot or not.
        * ``legend_orientation``: str
            Orientation of the legend, either "horizontal" or "vertical".
        * ``legend_max_cols``: int
            Maximum number of columns for the legend if ``legend_orientation`` is "horizontal".
        * ``rect``: tuple
            Rectangle coordinates for tight layout, i.e, the bounding box (x0, y0, x1, y1) that the subplots will fit
            into.
        * ``use_tight``: bool
            Whether to use tight layout or not. Default: True
        --- Heartbeat Borders ---
        * ``heartbeat_border_color``: str
            Color of the heartbeat borders.
        --- R-Peaks ---
        * ``r_peak_marker``: str
            Marker style for R-peaks.
        * ``r_peak_linestyle``: str
            Line style for R-peaks.
        * ``r_peak_linewidth``: float
            Line width for R-peaks.
        * ``r_peak_alpha``: float
            Alpha value for the R-peak vertical lines
        * ``r_peak_plot_marker``: bool
            Whether to plot markers at the R-peaks or not.
        * ``r_peak_plot_vline``: bool
            Whether to plot vertical lines at the R-peaks or not.
        --- B-Points ---
        * ``b_point_marker``: str
            Marker style for B-points.
        * ``b_point_linestyle``: str
            Line style for B-points.
        * ``b_point_linewidth``: float
            Line width for B-points.
        * ``b_point_alpha``: float
            Alpha value for B-point vertical lines.


    Return
    ------
    fig : :class:`matplotlib.figure.Figure`
        Figure object.
    ax : :class:`matplotlib.axes.Axes`
        Axes object.

    See Also
    --------
    :class:``pepbench.algorithms.icg.BPointExtractionSherwood1990``
        Algorithm implementation.

    References
    ----------
    .. [1] Sherwood, A., Allen, M. T., Fahrenberg, J., Kelsey, R. M., Lovallo, W. R., & Doornen, L. J. P. (1990).
        Methodological Guidelines for Impedance Cardiography. Psychophysiology, 27(1), 1-23.
        https://doi.org/10.1111/j.1469-8986.1990.tb02171.x

    """
    fig, ax = plt.subplots(**kwargs)
    kwargs.setdefault("legend_outside", True)
    kwargs.setdefault("legend_orientation", "horizontal")
    kwargs.setdefault("legend_loc", _get_legend_loc(kwargs))
    kwargs.setdefault("legend_max_cols", 4)
    rect = _get_rect(kwargs)

    if algo_params is None:
        algo_params = {}

    heartbeat_subset = _sanitize_heartbeat_subset(heartbeat_subset)
    ecg_data, icg_data = _get_data(datapoint, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset)
    heartbeats = _get_heartbeats(datapoint, heartbeat_subset)
    heartbeat_borders = _get_heartbeat_borders(icg_data, heartbeats)

    algo_params_c_point = {key: val for key, val in algo_params.items() if key in ["window_c_correction"]}
    algo_params_b_point = {key: val for key, val in algo_params.items() if key not in algo_params_c_point}
    c_point_algo = CPointExtractionScipyFindPeaks(**algo_params_c_point)
    c_point_algo.extract(icg=icg_data, heartbeats=heartbeats, sampling_rate_hz=datapoint.sampling_rate_icg)

    b_point_algo = BPointExtractionSherwood1990(**algo_params_b_point)
    b_point_algo.extract(
        icg=icg_data, heartbeats=heartbeats, c_points=c_point_algo.points_, sampling_rate_hz=datapoint.sampling_rate_icg
    )

    b_point_samples_reference = _get_reference_labels(datapoint, heartbeat_subset)["b_points"]
    b_point_samples = b_point_algo.points_["b_point_sample"].dropna().astype(int)

    icg_data = icg_data.squeeze()

    zero_crossings = np.where(np.diff(np.sign(icg_data)))[0]
    c_point_samples = c_point_algo.points_["c_point_sample"].dropna().astype(int)
    # get only the zero crossings between heartbeat start and c_point_sample
    zero_crossings_filtered = []
    for idx, row in heartbeats.iterrows():
        zero_crossings_filtered.append(
            zero_crossings[(zero_crossings > row["start_sample"]) & (zero_crossings < c_point_samples[idx])]
        )

    zero_crossings_filtered = np.concatenate(zero_crossings_filtered)
    zero_crossings_filtered = pd.Series(zero_crossings_filtered, name="zero_crossing_sample")

    _plot_signals_one_axis(
        df=icg_data,
        ax=ax,
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        color=cmaps.tech[0],
        **kwargs,
    )

    _add_heartbeat_borders(heartbeats=heartbeat_borders, ax=ax, **kwargs)

    _add_icg_c_points(icg_data, c_point_samples, ax=ax, **kwargs)

    _add_icg_b_points(
        icg_data,
        b_point_samples_reference,
        ax=ax,
        b_point_label="Reference B-Points",
        b_point_color=cmaps.phil_dark[0],
        **kwargs,
    )

    _add_icg_b_points(
        icg_data,
        b_point_samples,
        ax=ax,
        b_point_label="Detected B-Points",
        **kwargs,
    )

    _add_icg_b_points(
        icg_data,
        zero_crossings_filtered,
        ax=ax,
        b_point_label="Zero Crossings before C-Point",
        b_point_marker="X",
        b_point_color=cmaps.phil[2],
        **kwargs,
    )

    ax.axhline(
        0,
        color=cmaps.tech_dark[1],
        linestyle="--",
        linewidth=2,
        label="Zero Line",
    )

    _handle_legend_one_axis(fig=fig, ax=ax, **kwargs)

    fig.tight_layout(rect=rect)
    return fig, ax


def plot_b_point_extraction_debski1993_second_derivative(
    datapoint: BasePepDatasetWithAnnotations,
    *,
    heartbeat_subset: Sequence[int] | None = None,
    normalize_time: bool = False,
    algo_params: dict | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, Sequence[plt.Axes]]:
    """Plot example of B-point extraction using the second derivative method by Debski et al. (1993) [1].

    The algorithm is implemented as :class:``BPointExtractionDebski1993SecondDerivative``.

    Parameters
    ----------
    datapoint : BasePepDatasetWithAnnotations
        Datapoint to plot.
    heartbeat_subset : list of int, optional
        List of heartbeat_ids to plot. If None, all heartbeats are plotted.
    normalize_time : bool, optional
        Whether to normalize the time axis to seconds, starting at 0, or not. Default: False
    algo_params : dict, optional
        Parameters passed to the algorithm instance for C-point and B-point extraction.
        See :class:``pepbench.algorithms.icg.CPointExtractionScipyFindPeaks`` and
            :class:``pepbench.algorithms.icg.BPointExtractionDebski1993SecondDerivative`` for available parameters.
        Default: None (i.e., the default parameters of the algorithms are used).
    kwargs : dict
        Additional keyword arguments to pass to the plotting functions. Examples are:
        --- General ---
        * ``figsize``: tuple
            Size of the figure.
        * ``legend_loc``: str
            Location of the legend
        * ``legend_outside``: bool
            Whether to place the legend outside the plot or not.
        * ``legend_orientation``: str
            Orientation of the legend, either "horizontal" or "vertical".
        * ``legend_max_cols``: int
            Maximum number of columns for the legend if ``legend_orientation`` is "horizontal".
        * ``rect``: tuple
            Rectangle coordinates for tight layout, i.e, the bounding box (x0, y0, x1, y1) that the subplots will fit
            into.
        * ``use_tight``: bool
            Whether to use tight layout or not. Default: True
        --- Heartbeat Borders ---
        * ``heartbeat_border_color``: str
            Color of the heartbeat borders.
        --- R-Peaks ---
        * ``r_peak_marker``: str
            Marker style for R-peaks.
        * ``r_peak_linestyle``: str
            Line style for R-peaks.
        * ``r_peak_linewidth``: float
            Line width for R-peaks.
        * ``r_peak_alpha``: float
            Alpha value for the R-peak vertical lines
        * ``r_peak_plot_marker``: bool
            Whether to plot markers at the R-peaks or not.
        * ``r_peak_plot_vline``: bool
            Whether to plot vertical lines at the R-peaks or not.
        --- B-Points ---
        * ``b_point_marker``: str
            Marker style for B-points.
        * ``b_point_linestyle``: str
            Line style for B-points.
        * ``b_point_linewidth``: float
            Line width for B-points.
        * ``b_point_alpha``: float
            Alpha value for B-point vertical lines.


    Return
    ------
    fig : :class:`matplotlib.figure.Figure`
        Figure object.
    axs : list of :class:`matplotlib.axes.Axes`
        list of Axes objects, one for each subplot.

    See Also
    --------
    :class:``pepbench.algorithms.icg.BPointExtractionDebski1993SecondDerivative``
        Algorithm implementation.

    References
    ----------
    .. [1] Debski, T. T., Zhang, Y., Jennings, J. R., & Kamarck, T. W. (1993). Stability of cardiac impedance
        measures: Aortic opening (B-point) detection and scoring. Biological Psychology, 36(1-2), 63-74.
        https://doi.org/10.1016/0301-0511(93)90081-I

    """
    fig, axs = plt.subplots(nrows=2, sharex=True, **kwargs)
    kwargs.setdefault("legend_outside", True)
    kwargs.setdefault("legend_orientation", "horizontal")
    kwargs.setdefault("legend_loc", _get_legend_loc(kwargs))
    rect = _get_rect(kwargs)

    if algo_params is None:
        algo_params = {}

    heartbeat_subset = _sanitize_heartbeat_subset(heartbeat_subset)
    ecg_data, icg_data = _get_data(datapoint, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset)
    heartbeats = _get_heartbeats(datapoint, heartbeat_subset)
    heartbeat_borders = _get_heartbeat_borders(icg_data, heartbeats)

    icg_data = icg_data.squeeze()
    # compute ICG derivation
    icg_2nd_der = np.gradient(icg_data)
    icg_2nd_der = pd.DataFrame(icg_2nd_der, index=icg_data.index, columns=["ICG Deriv. $(d^2Z/dt^2)$"])

    algo_params_c_point = {key: val for key, val in algo_params.items() if key in ["window_c_correction"]}
    algo_params_b_point = {key: val for key, val in algo_params.items() if key not in algo_params_c_point}
    c_point_algo = CPointExtractionScipyFindPeaks(**algo_params_c_point)
    c_point_algo.extract(icg=icg_data, heartbeats=heartbeats, sampling_rate_hz=datapoint.sampling_rate_icg)

    b_point_algo = BPointExtractionDebski1993SecondDerivative(**algo_params_b_point)
    b_point_algo.extract(
        icg=icg_data, heartbeats=heartbeats, c_points=c_point_algo.points_, sampling_rate_hz=datapoint.sampling_rate_icg
    )

    r_peak_samples = heartbeats["r_peak_sample"].dropna().astype(int)
    c_point_samples = c_point_algo.points_["c_point_sample"].dropna().astype(int)
    b_point_samples = b_point_algo.points_["b_point_sample"].dropna().astype(int)
    search_window = pd.concat([r_peak_samples, c_point_samples], axis=1)

    b_point_samples_reference = _get_reference_labels(datapoint, heartbeat_subset)["b_points"]

    _plot_signals_one_axis(
        datapoint=datapoint,
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        plot_ecg=True,
        ax=axs[0],
        **kwargs,
    )
    _plot_signals_one_axis(
        df=icg_2nd_der,
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        plot_ecg=False,
        ax=axs[1],
        color=cmaps.fau_light[0],
        **kwargs,
    )
    _add_heartbeat_borders(heartbeats=heartbeat_borders, ax=axs[0], **kwargs)
    _add_heartbeat_borders(heartbeats=heartbeat_borders, ax=axs[1], **kwargs)

    _add_ecg_r_peaks(ecg_data=ecg_data, r_peaks=r_peak_samples, ax=axs[0], r_peak_linestyle="--", **kwargs)
    _add_ecg_r_peaks(
        ecg_data=ecg_data, r_peaks=r_peak_samples, ax=axs[1], r_peak_linestyle="--", r_peak_plot_marker=False, **kwargs
    )
    _add_icg_c_points(icg_data, c_point_samples, ax=axs[0], **kwargs)
    _add_icg_c_points(icg_data, c_point_samples, ax=axs[1], c_point_plot_marker=False, **kwargs)

    _add_icg_b_points(
        icg_2nd_der,
        b_point_samples,
        ax=axs[1],
        b_point_label="$d^2Z/dt^2$ Local Min.",
        b_point_marker="X",
        b_point_color=cmaps.phil_light[0],
        **kwargs,
    )
    _add_icg_b_points(
        icg_data,
        b_point_samples_reference,
        ax=axs[0],
        b_point_label="Reference B-Points",
        b_point_color=cmaps.phil_dark[0],
        **kwargs,
    )
    _add_icg_b_points(
        icg_data,
        b_point_samples,
        ax=axs[0],
        b_point_label="Detected B-Points",
        **kwargs,
    )

    for _idx, row in search_window.iterrows():
        start = icg_2nd_der.index[row["r_peak_sample"]]
        end = icg_2nd_der.index[row["c_point_sample"]]
        axs[0].axvspan(start, end, color=cmaps.fau_light[1], alpha=0.3, zorder=0, label="B-Point Search Windows")
        axs[1].axvspan(start, end, color=cmaps.fau_light[1], alpha=0.3, zorder=0, label="B-Point Search Windows")

    _handle_legend_two_axes(
        fig=fig,
        axs=axs,
        **kwargs,
    )

    fig.align_ylabels()
    fig.tight_layout(rect=rect)

    return fig, axs


def plot_b_point_extraction_arbol2017_isoelectric_crossings(
    datapoint: BasePepDatasetWithAnnotations,
    *,
    heartbeat_subset: Sequence[int] | None = None,
    normalize_time: bool = False,
    algo_params: dict | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot example of B-point extraction using the isoelectric crossings method by Arbol et al. (2017) [1].

    The algorithm is implemented as :class:``BPointExtractionArbol2017IsoelectricCrossings``.

    Parameters
    ----------
    datapoint : BasePepDatasetWithAnnotations
        Datapoint to plot.
    heartbeat_subset : list of int, optional
        List of heartbeat_ids to plot. If None, all heartbeats are plotted.
    normalize_time : bool, optional
        Whether to normalize the time axis to seconds, starting at 0, or not. Default: False
    algo_params : dict, optional
        Parameters passed to the algorithm instance for C-point and B-point extraction.
        See :class:``pepbench.algorithms.icg.CPointExtractionScipyFindPeaks`` and
            :class:``pepbench.algorithms.icg.BPointExtractionArbol2017IsoelectricCrossings`` for available parameters.
        Default: None (i.e., the default parameters of the algorithms are used).
    kwargs : dict
        Additional keyword arguments to pass to the plotting functions. Examples are:
        --- General ---
        * ``fig``, ``ax``: :class:`matplotlib.figure.Figure`, :class:`matplotlib.axes.Axes`
            Existing Figure and Axes objects to plot on; If not provided, a new figure and axes are created.
        * ``figsize``: tuple
            Size of the figure.
        * ``legend_loc``: str
            Location of the legend
        * ``legend_outside``: bool
            Whether to place the legend outside the plot or not.
        * ``legend_orientation``: str
            Orientation of the legend, either "horizontal" or "vertical".
        * ``legend_max_cols``: int
            Maximum number of columns for the legend if ``legend_orientation`` is "horizontal".
        * ``rect``: tuple
            Rectangle coordinates for tight layout, i.e, the bounding box (x0, y0, x1, y1) that the subplots will fit
            into.
        * ``use_tight``: bool
            Whether to use tight layout or not. Default: True
        --- Heartbeat Borders ---
        * ``heartbeat_border_color``: str
            Color of the heartbeat borders.
        --- R-Peaks ---
        * ``r_peak_marker``: str
            Marker style for R-peaks.
        * ``r_peak_linestyle``: str
            Line style for R-peaks.
        * ``r_peak_linewidth``: float
            Line width for R-peaks.
        * ``r_peak_alpha``: float
            Alpha value for the R-peak vertical lines
        * ``r_peak_plot_marker``: bool
            Whether to plot markers at the R-peaks or not.
        * ``r_peak_plot_vline``: bool
            Whether to plot vertical lines at the R-peaks or not.
        --- B-Points ---
        * ``b_point_marker``: str
            Marker style for B-points.
        * ``b_point_linestyle``: str
            Line style for B-points.
        * ``b_point_linewidth``: float
            Line width for B-points.
        * ``b_point_alpha``: float
            Alpha value for B-point vertical lines.


    Return
    ------
    fig : :class:`matplotlib.figure.Figure`
        Figure object.
    ax : :class:`matplotlib.axes.Axes`
        Axes object.

    See Also
    --------
    :class:``pepbench.algorithms.icg.BPointExtractionArbol2017IsoelectricCrossings``
        Algorithm implementation.

    References
    ----------
    .. [1] Árbol, J. R., Perakakis, P., Garrido, A., Mata, J. L., Fernández-Santaella, M. C., & Vila, J. (2017).
        Mathematical detection of aortic valve opening (B point) in impedance cardiography: A comparison of three
        popular algorithms. Psychophysiology, 54(3), 350-357. https://doi.org/10.1111/psyp.12799

    """
    fig, ax = plt.subplots(**kwargs)
    kwargs.setdefault("legend_max_cols", 4)
    kwargs.setdefault("legend_outside", True)
    kwargs.setdefault("legend_orientation", "horizontal")
    kwargs.setdefault("legend_loc", _get_legend_loc(kwargs))
    rect = _get_rect(kwargs)

    if algo_params is None:
        algo_params = {}

    heartbeat_subset = _sanitize_heartbeat_subset(heartbeat_subset)
    ecg_data, icg_data = _get_data(datapoint, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset)
    heartbeats = _get_heartbeats(datapoint, heartbeat_subset)
    heartbeat_borders = _get_heartbeat_borders(icg_data, heartbeats)

    algo_params_c_point = {key: val for key, val in algo_params.items() if key in ["window_c_correction"]}
    algo_params_b_point = {key: val for key, val in algo_params.items() if key not in algo_params_c_point}
    c_point_algo = CPointExtractionScipyFindPeaks(**algo_params_c_point)
    c_point_algo.extract(icg=icg_data, heartbeats=heartbeats, sampling_rate_hz=datapoint.sampling_rate_icg)

    b_point_algo = BPointExtractionArbol2017IsoelectricCrossings(**algo_params_b_point)
    b_point_algo.extract(
        icg=icg_data, heartbeats=heartbeats, c_points=c_point_algo.points_, sampling_rate_hz=datapoint.sampling_rate_icg
    )

    c_point_samples = c_point_algo.points_["c_point_sample"].dropna().astype(int)
    b_point_samples = b_point_algo.points_["b_point_sample"].dropna().astype(int)
    b_point_samples_reference = _get_reference_labels(datapoint, heartbeat_subset)["b_points"]

    icg_data = icg_data.squeeze()

    _plot_signals_one_axis(
        df=icg_data,
        ax=ax,
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        color=cmaps.tech[0],
        **kwargs,
    )

    for idx, row in heartbeats.iterrows():
        start = row["start_sample"]
        end = row["end_sample"]

        # get isoelectric line per heartbeat
        icg_heartbeat = icg_data.iloc[start:end]
        iso_line = icg_heartbeat.mean()

        # compute isoelectric line crossings
        iso_crossings = np.where(np.diff(np.sign(icg_heartbeat - iso_line)))[0]
        # filter only isoelectric crossings that are *before* the C-point
        c_point = c_point_samples[idx] - start
        iso_crossings = iso_crossings[iso_crossings < c_point]
        iso_crossings = iso_crossings + start
        _add_icg_b_points(
            icg_data,
            iso_crossings,
            ax=ax,
            b_point_label="Isoelectric Crossings before C-Point",
            b_point_color=cmaps.phil[2],
            **kwargs,
        )

        # plot isoelectric line per heartbeat
        ax.hlines(
            iso_line,
            xmin=icg_heartbeat.index[0],
            xmax=icg_heartbeat.index[-1],
            color=cmaps.tech_dark[0],
            linestyle="--",
            linewidth=2,
            zorder=0,
            label="Isoelectric Line per Heartbeat",
        )

    _add_heartbeat_borders(heartbeats=heartbeat_borders, ax=ax, **kwargs)
    _add_icg_b_points(
        icg_data,
        b_point_samples_reference,
        ax=ax,
        b_point_label="Reference B-Points",
        b_point_color=cmaps.phil_dark[0],
        **kwargs,
    )

    _add_icg_b_points(
        icg_data,
        b_point_samples,
        ax=ax,
        b_point_label="Detected B-Points",
        **kwargs,
    )

    _add_icg_c_points(icg_data, c_point_samples, ax=ax, **kwargs)

    _handle_legend_one_axis(
        fig=fig,
        ax=ax,
        **kwargs,
    )

    fig.tight_layout(rect=rect)
    fig.align_ylabels()

    return fig, ax


def plot_b_point_extraction_arbol2017_second_derivative(
    datapoint: BasePepDatasetWithAnnotations,
    *,
    heartbeat_subset: Sequence[int] | None = None,
    normalize_time: bool = False,
    algo_params: dict | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, Sequence[plt.Axes]]:
    """Plot example of B-point extraction using the second derivative method by Arbol et al. (2017) [1].

    The algorithm is implemented as :class:``BPointExtractionArbol2017SecondDerivative``.

    Parameters
    ----------
    datapoint : BasePepDatasetWithAnnotations
        Datapoint to plot.
    heartbeat_subset : list of int, optional
        List of heartbeat_ids to plot. If None, all heartbeats are plotted.
    normalize_time : bool, optional
        Whether to normalize the time axis to seconds, starting at 0, or not. Default: False
    algo_params : dict, optional
        Parameters passed to the algorithm instance for C-point and B-point extraction.
        See :class:``pepbench.algorithms.icg.CPointExtractionScipyFindPeaks`` and
            :class:``pepbench.algorithms.icg.BPointExtractionArbol2017SecondDerivative`` for available parameters.
        Default: None (i.e., the default parameters of the algorithms are used).
    kwargs : dict
        Additional keyword arguments to pass to the plotting functions. Examples are:
        --- General ---
        * ``figsize``: tuple
            Size of the figure.
        * ``legend_loc``: str
            Location of the legend
        * ``legend_outside``: bool
            Whether to place the legend outside the plot or not.
        * ``legend_orientation``: str
            Orientation of the legend, either "horizontal" or "vertical".
        * ``legend_max_cols``: int
            Maximum number of columns for the legend if ``legend_orientation`` is "horizontal".
        * ``rect``: tuple
            Rectangle coordinates for tight layout, i.e, the bounding box (x0, y0, x1, y1) that the subplots will fit
            into.
        * ``use_tight``: bool
            Whether to use tight layout or not. Default: True
        --- Heartbeat Borders ---
        * ``heartbeat_border_color``: str
            Color of the heartbeat borders.
        --- R-Peaks ---
        * ``r_peak_marker``: str
            Marker style for R-peaks.
        * ``r_peak_linestyle``: str
            Line style for R-peaks.
        * ``r_peak_linewidth``: float
            Line width for R-peaks.
        * ``r_peak_alpha``: float
            Alpha value for the R-peak vertical lines
        * ``r_peak_plot_marker``: bool
            Whether to plot markers at the R-peaks or not.
        * ``r_peak_plot_vline``: bool
            Whether to plot vertical lines at the R-peaks or not.
        --- B-Points ---
        * ``b_point_marker``: str
            Marker style for B-points.
        * ``b_point_linestyle``: str
            Line style for B-points.
        * ``b_point_linewidth``: float
            Line width for B-points.
        * ``b_point_alpha``: float
            Alpha value for B-point vertical lines.


    Return
    ------
    fig : :class:`matplotlib.figure.Figure`
        Figure object.
    axs : list of :class:`matplotlib.axes.Axes`
        list of Axes objects, one for each subplot.

    See Also
    --------
    :class:``pepbench.algorithms.icg.BPointExtractionArbol2017SecondDerivative``
        Algorithm implementation.

    References
    ----------
    .. [1] Árbol, J. R., Perakakis, P., Garrido, A., Mata, J. L., Fernández-Santaella, M. C., & Vila, J. (2017).
        Mathematical detection of aortic valve opening (B point) in impedance cardiography: A comparison of three
        popular algorithms. Psychophysiology, 54(3), 350-357. https://doi.org/10.1111/psyp.12799

    """
    fig, axs = plt.subplots(nrows=2, sharex=True, **kwargs)
    kwargs.setdefault("legend_max_cols", 4)
    kwargs.setdefault("legend_outside", True)
    kwargs.setdefault("legend_orientation", "horizontal")
    kwargs.setdefault("legend_loc", _get_legend_loc(kwargs))
    kwargs.setdefault("rect", (0, 0, 1, 0.8))
    rect = _get_rect(kwargs)

    if algo_params is None:
        algo_params = {}

    heartbeat_subset = _sanitize_heartbeat_subset(heartbeat_subset)
    ecg_data, icg_data = _get_data(datapoint, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset)
    heartbeats = _get_heartbeats(datapoint, heartbeat_subset)
    heartbeat_borders = _get_heartbeat_borders(icg_data, heartbeats)

    algo_params_c_point = {key: val for key, val in algo_params.items() if key in ["window_c_correction"]}
    algo_params_b_point = {key: val for key, val in algo_params.items() if key not in algo_params_c_point}
    c_point_algo = CPointExtractionScipyFindPeaks(**algo_params_c_point)
    c_point_algo.extract(icg=icg_data, heartbeats=heartbeats, sampling_rate_hz=datapoint.sampling_rate_icg)

    b_point_algo = BPointExtractionArbol2017SecondDerivative(**algo_params_b_point)
    b_point_algo.extract(
        icg=icg_data, heartbeats=heartbeats, c_points=c_point_algo.points_, sampling_rate_hz=datapoint.sampling_rate_icg
    )

    icg_data = icg_data.squeeze()
    icg_2nd_der = np.gradient(icg_data)
    icg_2nd_der = pd.DataFrame(icg_2nd_der, index=icg_data.index, columns=["ICG 2nd Deriv. $(d^2Z/dt^2)$"])

    c_point_samples = c_point_algo.points_["c_point_sample"].dropna().astype(int)
    search_window_start = c_point_samples - int(150 / 1000 * datapoint.sampling_rate_icg)
    search_window_start.name = "search_window_start"
    search_window_end = search_window_start + int(50 / 1000 * datapoint.sampling_rate_icg)
    search_window_end.name = "search_window_end"
    search_window = pd.concat([search_window_start, search_window_end], axis=1)

    b_point_samples = b_point_algo.points_["b_point_sample"].dropna().astype(int)
    b_point_samples_reference = _get_reference_labels(datapoint, heartbeat_subset)["b_points"]

    _plot_signals_one_axis(
        df=icg_data,
        ax=axs[0],
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        color=cmaps.tech[0],
        **kwargs,
    )
    _plot_signals_one_axis(
        df=icg_2nd_der,
        ax=axs[1],
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        color=cmaps.tech_dark[0],
        **kwargs,
    )
    _add_heartbeat_borders(heartbeats=heartbeat_borders, ax=axs[0], **kwargs)
    _add_heartbeat_borders(heartbeats=heartbeat_borders, ax=axs[1], **kwargs)

    _add_icg_b_points(
        icg_data,
        b_point_samples_reference,
        ax=axs[0],
        b_point_label="Reference B-Points",
        b_point_color=cmaps.phil_dark[0],
        **kwargs,
    )
    _add_icg_b_points(
        icg_data,
        b_point_samples,
        ax=axs[0],
        b_point_label="Detected B-Points",
        **kwargs,
    )
    _add_icg_b_points(
        icg_2nd_der,
        b_point_samples,
        b_point_label="$d^2Z/dt^2$ Local Max.",
        b_point_marker="X",
        b_point_color=cmaps.phil_light[0],
        ax=axs[1],
        **kwargs,
    )

    _add_icg_c_points(icg_data, c_point_samples, ax=axs[0], **kwargs)
    _add_icg_c_points(
        icg_data,
        search_window_start,
        ax=axs[0],
        c_point_color=cmaps.wiso_light[1],
        c_point_label="C-Points - 150 ms",
        **kwargs,
    )
    _add_icg_c_points(icg_data, c_point_samples, ax=axs[1], c_point_plot_marker=False, **kwargs)
    _add_icg_c_points(
        icg_data,
        search_window_start,
        ax=axs[1],
        c_point_color=cmaps.wiso_light[1],
        c_point_plot_marker=False,
        **kwargs,
    )
    for _idx, row in search_window.iterrows():
        start = icg_data.index[row["search_window_start"]]
        end = icg_data.index[row["search_window_end"]]
        axs[0].axvspan(
            start, end, color=cmaps.tech_light[0], alpha=0.3, zorder=0, label="B-Point Search Windows: 50 ms"
        )
        axs[1].axvspan(
            start, end, color=cmaps.tech_light[0], alpha=0.3, zorder=0, label="B-Point Search Windows: 50 ms"
        )

    _handle_legend_two_axes(
        fig=fig,
        axs=axs,
        **kwargs,
    )

    fig.tight_layout(rect=rect)
    fig.align_ylabels()

    return fig, axs


def plot_b_point_extraction_arbol2017_third_derivative(
    datapoint: BasePepDatasetWithAnnotations,
    *,
    heartbeat_subset: Sequence[int] | None = None,
    normalize_time: bool = False,
    algo_params: dict | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, Sequence[plt.Axes]]:
    """Plot example of B-point extraction using the third derivative method by Arbol et al. (2017) [1].

    The algorithm is implemented as :class:``BPointExtractionArbol2017ThirdDerivative``.

    Parameters
    ----------
    datapoint : BasePepDatasetWithAnnotations
        Datapoint to plot.
    heartbeat_subset : list of int, optional
        List of heartbeat_ids to plot. If None, all heartbeats are plotted.
    normalize_time : bool, optional
        Whether to normalize the time axis to seconds, starting at 0, or not. Default: False
    algo_params : dict, optional
        Parameters passed to the algorithm instance for C-point and B-point extraction.
        See :class:``pepbench.algorithms.icg.CPointExtractionScipyFindPeaks`` and
            :class:``pepbench.algorithms.icg.BPointExtractionArbol2017ThirdDerivative`` for available parameters.
        Default: None (i.e., the default parameters of the algorithms are used).
    kwargs : dict
        Additional keyword arguments to pass to the plotting functions. Examples are:
        --- General ---
        * ``figsize``: tuple
            Size of the figure.
        * ``legend_loc``: str
            Location of the legend
        * ``legend_outside``: bool
            Whether to place the legend outside the plot or not.
        * ``legend_orientation``: str
            Orientation of the legend, either "horizontal" or "vertical".
        * ``legend_max_cols``: int
            Maximum number of columns for the legend if ``legend_orientation`` is "horizontal".
        * ``rect``: tuple
            Rectangle coordinates for tight layout, i.e, the bounding box (x0, y0, x1, y1) that the subplots will fit
            into.
        * ``use_tight``: bool
            Whether to use tight layout or not. Default: True
        --- Heartbeat Borders ---
        * ``heartbeat_border_color``: str
            Color of the heartbeat borders.
        --- R-Peaks ---
        * ``r_peak_marker``: str
            Marker style for R-peaks.
        * ``r_peak_linestyle``: str
            Line style for R-peaks.
        * ``r_peak_linewidth``: float
            Line width for R-peaks.
        * ``r_peak_alpha``: float
            Alpha value for the R-peak vertical lines
        * ``r_peak_plot_marker``: bool
            Whether to plot markers at the R-peaks or not.
        * ``r_peak_plot_vline``: bool
            Whether to plot vertical lines at the R-peaks or not.
        --- B-Points ---
        * ``b_point_marker``: str
            Marker style for B-points.
        * ``b_point_linestyle``: str
            Line style for B-points.
        * ``b_point_linewidth``: float
            Line width for B-points.
        * ``b_point_alpha``: float
            Alpha value for B-point vertical lines.


    Return
    ------
    fig : :class:`matplotlib.figure.Figure`
        Figure object.
    axs : list of :class:`matplotlib.axes.Axes`
        list of Axes objects, one for each subplot.

    See Also
    --------
    :class:``pepbench.algorithms.icg.BPointExtractionArbol2017ThirdDerivative``
        Algorithm implementation.

    References
    ----------
    .. [1] Árbol, J. R., Perakakis, P., Garrido, A., Mata, J. L., Fernández-Santaella, M. C., & Vila, J. (2017).
        Mathematical detection of aortic valve opening (B point) in impedance cardiography: A comparison of three
        popular algorithms. Psychophysiology, 54(3), 350-357. https://doi.org/10.1111/psyp.12799

    """
    fig, axs = plt.subplots(nrows=2, sharex=True, **kwargs)
    kwargs.setdefault("legend_max_cols", 4)
    kwargs.setdefault("legend_outside", True)
    kwargs.setdefault("legend_orientation", "horizontal")
    kwargs.setdefault("legend_loc", _get_legend_loc(kwargs))
    kwargs.setdefault("rect", (0, 0, 1, 0.8))
    rect = _get_rect(kwargs)

    if algo_params is None:
        algo_params = {}

    heartbeat_subset = _sanitize_heartbeat_subset(heartbeat_subset)
    ecg_data, icg_data = _get_data(datapoint, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset)
    heartbeats = _get_heartbeats(datapoint, heartbeat_subset)
    heartbeat_borders = _get_heartbeat_borders(icg_data, heartbeats)

    algo_params_c_point = {key: val for key, val in algo_params.items() if key in ["window_c_correction"]}
    algo_params_b_point = {key: val for key, val in algo_params.items() if key not in algo_params_c_point}
    c_point_algo = CPointExtractionScipyFindPeaks(**algo_params_c_point)
    c_point_algo.extract(icg=icg_data, heartbeats=heartbeats, sampling_rate_hz=datapoint.sampling_rate_icg)

    b_point_algo = BPointExtractionArbol2017ThirdDerivative(**algo_params_b_point)
    b_point_algo.extract(
        icg=icg_data, heartbeats=heartbeats, c_points=c_point_algo.points_, sampling_rate_hz=datapoint.sampling_rate_icg
    )

    icg_data = icg_data.squeeze()
    icg_3rd_der = np.gradient(np.gradient(icg_data))
    icg_3rd_der = pd.DataFrame(icg_3rd_der, index=icg_data.index, columns=["ICG 3rd Deriv. $(d^3Z/dt^3)$"])

    c_point_samples = c_point_algo.points_["c_point_sample"].dropna().astype(int)
    c_point_minus_300_samples = c_point_samples - int(300 / 1000 * datapoint.sampling_rate_icg)
    c_point_minus_300_samples.name = "c_point_sample_minus_300"
    search_window = pd.concat([c_point_minus_300_samples, c_point_samples], axis=1)

    b_point_samples = b_point_algo.points_["b_point_sample"].dropna().astype(int)
    b_point_samples_reference = _get_reference_labels(datapoint, heartbeat_subset)["b_points"]

    _plot_signals_one_axis(
        df=icg_data,
        ax=axs[0],
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        color=cmaps.tech[0],
        **kwargs,
    )
    _plot_signals_one_axis(
        df=icg_3rd_der,
        ax=axs[1],
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        color=cmaps.tech_dark[0],
        **kwargs,
    )
    _add_heartbeat_borders(heartbeats=heartbeat_borders, ax=axs[0], **kwargs)
    _add_heartbeat_borders(heartbeats=heartbeat_borders, ax=axs[1], **kwargs)

    _add_icg_b_points(
        icg_data,
        b_point_samples_reference,
        ax=axs[0],
        b_point_label="Reference B-Points",
        b_point_color=cmaps.phil_dark[0],
        **kwargs,
    )
    _add_icg_b_points(
        icg_data,
        b_point_samples,
        ax=axs[0],
        b_point_label="Detected B-Points",
        **kwargs,
    )
    _add_icg_b_points(
        icg_3rd_der,
        b_point_samples,
        b_point_label="$d^3Z/dt^3$ Local Max.",
        b_point_marker="X",
        b_point_color=cmaps.phil_light[0],
        ax=axs[1],
        **kwargs,
    )

    _add_icg_c_points(icg_data, c_point_samples, ax=axs[0], **kwargs)
    _add_icg_c_points(
        icg_data,
        c_point_minus_300_samples,
        ax=axs[0],
        c_point_color=cmaps.wiso_light[1],
        c_point_label="C-Points - 300 ms",
        **kwargs,
    )
    _add_icg_c_points(icg_data, c_point_samples, ax=axs[1], c_point_plot_marker=False, **kwargs)
    _add_icg_c_points(
        icg_data,
        c_point_minus_300_samples,
        ax=axs[1],
        c_point_color=cmaps.wiso_light[1],
        c_point_plot_marker=False,
        **kwargs,
    )
    for _idx, row in search_window.iterrows():
        start = icg_data.index[row["c_point_sample_minus_300"]]
        end = icg_data.index[row["c_point_sample"]]
        axs[0].axvspan(start, end, color=cmaps.tech_light[0], alpha=0.3, zorder=0, label="B-Point Search Windows")
        axs[1].axvspan(start, end, color=cmaps.tech_light[0], alpha=0.3, zorder=0, label="B-Point Search Windows")

    _handle_legend_two_axes(
        fig=fig,
        axs=axs,
        **kwargs,
    )

    fig.tight_layout(rect=rect)
    fig.align_ylabels()

    return fig, axs


def plot_b_point_extraction_lozano2007_linear_regression(
    datapoint: BasePepDatasetWithAnnotations,
    *,
    heartbeat_subset: Sequence[int] | None = None,
    normalize_time: bool | None = False,
    algo_params: dict | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, Sequence[plt.Axes]]:
    """Plot example of B-point extraction using the linear regression method by Lozano et al. (2007) [1].

    The algorithm is implemented as :class:``BPointExtractionLozano2007LinearRegression``.

    Parameters
    ----------
    datapoint : BasePepDatasetWithAnnotations
        Datapoint to plot.
    heartbeat_subset : list of int, optional
        List of heartbeat_ids to plot. If None, all heartbeats are plotted.
    normalize_time : bool, optional
        Whether to normalize the time axis to seconds, starting at 0, or not. Default: False
    algo_params : dict, optional
        Parameters passed to the algorithm instance for C-point and B-point extraction.
        See :class:``pepbench.algorithms.icg.CPointExtractionScipyFindPeaks`` and
            :class:``pepbench.algorithms.icg.BPointExtractionLozano2007LinearRegression`` for available parameters.
        Default: None (i.e., the default parameters of the algorithms are used).
    kwargs : dict
        Additional keyword arguments to pass to the plotting functions. Examples are:
        --- General ---
        * ``fig``, ``ax``: :class:`matplotlib.figure.Figure`, :class:`matplotlib.axes.Axes`
            Figure and Axes objects to plot on; If not provided, a new figure and axes are created.
        * ``figsize``: tuple
            Size of the figure.
        * ``legend_loc``: str
            Location of the legend
        * ``legend_outside``: bool
            Whether to place the legend outside the plot or not.
        * ``legend_orientation``: str
            Orientation of the legend, either "horizontal" or "vertical".
        * ``legend_max_cols``: int
            Maximum number of columns for the legend if ``legend_orientation`` is "horizontal".
        * ``rect``: tuple
            Rectangle coordinates for tight layout, i.e, the bounding box (x0, y0, x1, y1) that the subplots will fit
            into.
        * ``use_tight``: bool
            Whether to use tight layout or not. Default: True
        --- Heartbeat Borders ---
        * ``heartbeat_border_color``: str
            Color of the heartbeat borders.
        --- R-Peaks ---
        * ``r_peak_marker``: str
            Marker style for R-peaks.
        * ``r_peak_linestyle``: str
            Line style for R-peaks.
        * ``r_peak_linewidth``: float
            Line width for R-peaks.
        * ``r_peak_alpha``: float
            Alpha value for the R-peak vertical lines
        * ``r_peak_plot_marker``: bool
            Whether to plot markers at the R-peaks or not.
        * ``r_peak_plot_vline``: bool
            Whether to plot vertical lines at the R-peaks or not.
        --- B-Points ---
        * ``b_point_marker``: str
            Marker style for B-points.
        * ``b_point_linestyle``: str
            Line style for B-points.
        * ``b_point_linewidth``: float
            Line width for B-points.
        * ``b_point_alpha``: float
            Alpha value for B-point vertical lines.


    Return
    ------
    fig : :class:`matplotlib.figure.Figure`
        Figure object.
    axs : list of :class:`matplotlib.axes.Axes`
        list of Axes objects, one for each subplot.

    See Also
    --------
    :class:``pepbench.algorithms.icg.BPointExtractionLozano2007LinearRegression``
        Algorithm implementation.

    References
    ----------
    .. [1] Lozano, D. L., Norman, G., Knox, D., Wood, B. L., Miller, B. D., Emery, C. F., & Berntson, G. G. (2007).
        Where to B in dZ/dt. Psychophysiology, 44(1), 113-119. https://doi.org/10.1111/j.1469-8986.2006.00468.x

    """
    kwargs.setdefault("legend_outside", True)
    kwargs.setdefault("legend_orientation", "horizontal")
    kwargs.setdefault("legend_loc", _get_legend_loc(kwargs))
    rect = _get_rect(kwargs)

    if algo_params is None:
        algo_params = {}

    heartbeat_subset = _sanitize_heartbeat_subset(heartbeat_subset)
    ecg_data, icg_data = _get_data(datapoint, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset)
    heartbeats = _get_heartbeats(datapoint, heartbeat_subset)
    heartbeat_borders = _get_heartbeat_borders(icg_data, heartbeats)

    algo_params_c_point = {
        key: val for key, val in algo_params.items() if key in ["window_c_correction", "save_candidates"]
    }
    algo_params_b_point = {key: val for key, val in algo_params.items() if key not in algo_params_c_point}
    c_point_algo = CPointExtractionScipyFindPeaks(**algo_params_c_point)
    c_point_algo.extract(icg=icg_data, heartbeats=heartbeats, sampling_rate_hz=datapoint.sampling_rate_icg)

    b_point_algo = BPointExtractionLozano2007LinearRegression(**algo_params_b_point)
    b_point_algo.extract(
        icg=icg_data, heartbeats=heartbeats, c_points=c_point_algo.points_, sampling_rate_hz=datapoint.sampling_rate_icg
    )

    b_point_samples = b_point_algo.points_["b_point_sample"].dropna().astype(int)
    c_point_samples = c_point_algo.points_["c_point_sample"].dropna().astype(int)
    r_peak_samples = heartbeats["r_peak_sample"].astype(int)

    fig, ax = plot_signals(
        datapoint=datapoint,
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        collapse=True,
        **kwargs,
    )
    _add_heartbeat_borders(heartbeats=heartbeat_borders, ax=ax, **kwargs)
    _add_ecg_r_peaks(ecg_data, r_peak_samples, ax=ax, **kwargs)
    _add_icg_c_points(icg_data, c_point_samples, ax=ax, **kwargs)
    _add_icg_b_points(icg_data, b_point_samples, ax=ax, **kwargs)

    y_c_point_max = np.max(icg_data.iloc[c_point_samples], axis=0).squeeze()
    y_r_peak_max = np.max(ecg_data.iloc[r_peak_samples], axis=0).squeeze()

    # draw arrow from R-peak to Q-peak
    for r_peak, c_point, b_point in zip(r_peak_samples, c_point_samples, b_point_samples, strict=False):
        x_r_peak = ecg_data.index[r_peak]
        x_c_point = icg_data.index[c_point]
        x_b_point = icg_data.index[b_point]

        middle_x_rc = x_r_peak + (x_c_point - x_r_peak) / 2
        middle_x_rb = x_r_peak + (x_b_point - x_r_peak) / 2
        # align text to the center of the array
        ax.annotate(
            "",
            xy=(x_c_point, y_c_point_max),
            xytext=(x_r_peak, y_c_point_max),
            # align text to the center of the array
            arrowprops={"arrowstyle": "<->", "color": cmaps.tech_dark[0], "lw": 2, "shrinkA": 0.0, "shrinkB": 0.0},
            ha="center",
            zorder=3,
        )
        ax.annotate(
            r"R-C Interval",
            xy=(middle_x_rc, y_c_point_max),
            xytext=(0, 12),
            textcoords="offset points",
            fontsize="small",
            bbox=_get_annotation_bbox_no_edge(),
            ha="center",
        )

        # align text to the center of the array
        ax.annotate(
            "",
            xy=(x_b_point, y_r_peak_max),
            xytext=(x_r_peak, y_r_peak_max),
            # align text to the center of the array
            arrowprops={"arrowstyle": "->", "color": cmaps.tech_dark[1], "lw": 2, "shrinkA": 0.0, "shrinkB": 0.0},
            ha="center",
            zorder=3,
        )
        ax.annotate(
            r"$0.55 \cdot RC + 4.45$",
            xy=(middle_x_rb, y_r_peak_max),
            xytext=(0, 12),
            textcoords="offset points",
            fontsize="small",
            bbox=_get_annotation_bbox_no_edge(),
            ha="center",
        )

    _handle_legend_one_axis(
        fig=fig,
        ax=ax,
        **kwargs,
    )

    fig.tight_layout(rect=rect)
    ylims = ax.get_ylim()
    ax.set_ylim(ylims[0], 1.25 * y_c_point_max)

    return fig, ax


def plot_b_point_extraction_lozano2007_quadratic_regression(
    datapoint: BasePepDatasetWithAnnotations,
    *,
    heartbeat_subset: Sequence[int] | None = None,
    normalize_time: bool | None = False,
    algo_params: dict | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, Sequence[plt.Axes]]:
    """Plot example of B-point extraction using the quadratic regression method by Lozano et al. (2007) [1].

    The algorithm is implemented as :class:``BPointExtractionLozano2007QuadraticRegression``.

    Parameters
    ----------
    datapoint : BasePepDatasetWithAnnotations
        Datapoint to plot.
    heartbeat_subset : list of int, optional
        List of heartbeat_ids to plot. If None, all heartbeats are plotted.
    normalize_time : bool, optional
        Whether to normalize the time axis to seconds, starting at 0, or not. Default: False
    algo_params : dict, optional
        Parameters passed to the algorithm instance for C-point and B-point extraction.
        See :class:``pepbench.algorithms.icg.CPointExtractionScipyFindPeaks`` and
            :class:``pepbench.algorithms.icg.BPointExtractionLozano2007QuadraticRegression`` for available parameters.
        Default: None (i.e., the default parameters of the algorithms are used).
    kwargs : dict
        Additional keyword arguments to pass to the plotting functions. Examples are:
        --- General ---
        * ``fig``, ``ax``: :class:`matplotlib.figure.Figure`, :class:`matplotlib.axes.Axes`
            Figure and Axes objects to plot on; If not provided, a new figure and axes are created.
        * ``figsize``: tuple
            Size of the figure.
        * ``legend_loc``: str
            Location of the legend
        * ``legend_outside``: bool
            Whether to place the legend outside the plot or not.
        * ``legend_orientation``: str
            Orientation of the legend, either "horizontal" or "vertical".
        * ``legend_max_cols``: int
            Maximum number of columns for the legend if ``legend_orientation`` is "horizontal".
        * ``rect``: tuple
            Rectangle coordinates for tight layout, i.e, the bounding box (x0, y0, x1, y1) that the subplots will fit
            into.
        * ``use_tight``: bool
            Whether to use tight layout or not. Default: True
        --- Heartbeat Borders ---
        * ``heartbeat_border_color``: str
            Color of the heartbeat borders.
        --- R-Peaks ---
        * ``r_peak_marker``: str
            Marker style for R-peaks.
        * ``r_peak_linestyle``: str
            Line style for R-peaks.
        * ``r_peak_linewidth``: float
            Line width for R-peaks.
        * ``r_peak_alpha``: float
            Alpha value for the R-peak vertical lines
        * ``r_peak_plot_marker``: bool
            Whether to plot markers at the R-peaks or not.
        * ``r_peak_plot_vline``: bool
            Whether to plot vertical lines at the R-peaks or not.
        --- B-Points ---
        * ``b_point_marker``: str
            Marker style for B-points.
        * ``b_point_linestyle``: str
            Line style for B-points.
        * ``b_point_linewidth``: float
            Line width for B-points.
        * ``b_point_alpha``: float
            Alpha value for B-point vertical lines.


    Return
    ------
    fig : :class:`matplotlib.figure.Figure`
        Figure object.
    ax : :class:`matplotlib.axes.Axes`
        Axes object.

    See Also
    --------
    :class:``pepbench.algorithms.icg.BPointExtractionLozano2007QuadraticRegression``
        Algorithm implementation.

    References
    ----------
    .. [1] Lozano, D. L., Norman, G., Knox, D., Wood, B. L., Miller, B. D., Emery, C. F., & Berntson, G. G. (2007).
        Where to B in dZ/dt. Psychophysiology, 44(1), 113-119. https://doi.org/10.1111/j.1469-8986.2006.00468.x

    """
    kwargs.setdefault("legend_outside", True)
    kwargs.setdefault("legend_orientation", "horizontal")
    kwargs.setdefault("legend_loc", _get_legend_loc(kwargs))
    rect = _get_rect(kwargs)

    if algo_params is None:
        algo_params = {}

    heartbeat_subset = _sanitize_heartbeat_subset(heartbeat_subset)
    ecg_data, icg_data = _get_data(datapoint, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset)
    heartbeats = _get_heartbeats(datapoint, heartbeat_subset)
    heartbeat_borders = _get_heartbeat_borders(icg_data, heartbeats)

    algo_params_c_point = {key: val for key, val in algo_params.items() if key in ["window_c_correction"]}
    algo_params_b_point = {key: val for key, val in algo_params.items() if key not in algo_params_c_point}
    c_point_algo = CPointExtractionScipyFindPeaks(**algo_params_c_point)
    c_point_algo.extract(icg=icg_data, heartbeats=heartbeats, sampling_rate_hz=datapoint.sampling_rate_icg)

    b_point_algo = BPointExtractionLozano2007QuadraticRegression(**algo_params_b_point)
    b_point_algo.extract(
        icg=icg_data, heartbeats=heartbeats, c_points=c_point_algo.points_, sampling_rate_hz=datapoint.sampling_rate_icg
    )

    b_point_samples = b_point_algo.points_["b_point_sample"].dropna().astype(int)
    c_point_samples = c_point_algo.points_["c_point_sample"].dropna().astype(int)
    r_peak_samples = heartbeats["r_peak_sample"].astype(int)

    fig, ax = plot_signals(
        datapoint=datapoint,
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        collapse=True,
        **kwargs,
    )
    _add_heartbeat_borders(heartbeats=heartbeat_borders, ax=ax, **kwargs)
    _add_ecg_r_peaks(ecg_data, r_peak_samples, ax=ax, **kwargs)
    _add_icg_c_points(icg_data, c_point_samples, ax=ax, **kwargs)
    _add_icg_b_points(icg_data, b_point_samples, ax=ax, **kwargs)

    y_c_point_max = np.max(icg_data.iloc[c_point_samples], axis=0).squeeze()
    y_r_peak_max = np.max(ecg_data.iloc[r_peak_samples], axis=0).squeeze()

    # draw arrow from R-peak to Q-peak
    for r_peak, c_point, b_point in zip(r_peak_samples, c_point_samples, b_point_samples, strict=False):
        x_r_peak = ecg_data.index[r_peak]
        x_c_point = icg_data.index[c_point]
        x_b_point = icg_data.index[b_point]

        middle_x_rc = x_r_peak + (x_c_point - x_r_peak) / 2
        middle_x_rb = x_r_peak + (x_b_point - x_r_peak) / 2
        # align text to the center of the array
        ax.annotate(
            "",
            xy=(x_c_point, y_c_point_max),
            xytext=(x_r_peak, y_c_point_max),
            # align text to the center of the array
            arrowprops={"arrowstyle": "<->", "color": cmaps.tech_dark[0], "lw": 2, "shrinkA": 0.0, "shrinkB": 0.0},
            ha="center",
            zorder=3,
        )
        ax.annotate(
            r"R-C Interval",
            xy=(middle_x_rc, y_c_point_max),
            xytext=(0, 12),
            textcoords="offset points",
            fontsize="small",
            bbox=_get_annotation_bbox_no_edge(),
            ha="center",
        )

        # align text to the center of the array
        ax.annotate(
            "",
            xy=(x_b_point, y_r_peak_max),
            xytext=(x_r_peak, y_r_peak_max),
            # align text to the center of the array
            arrowprops={"arrowstyle": "->", "color": cmaps.tech_dark[1], "lw": 2, "shrinkA": 0.0, "shrinkB": 0.0},
            ha="center",
            zorder=3,
        )
        ax.annotate(
            "$-3.2e^{-3} \\cdot RC^2$\n $+1.233 \\cdot RC$\n $-31.59$",
            xy=(middle_x_rb, y_r_peak_max),
            xytext=(0, 12),
            textcoords="offset points",
            fontsize="small",
            bbox=_get_annotation_bbox_no_edge(),
            ha="center",
        )

    _handle_legend_one_axis(
        fig=fig,
        ax=ax,
        **kwargs,
    )

    fig.tight_layout(rect=rect)
    ylims = ax.get_ylim()
    ax.set_ylim(ylims[0], 1.25 * y_c_point_max)

    return fig, ax


def plot_b_point_extraction_drost2022(  # noqa: PLR0915
    datapoint: BasePepDatasetWithAnnotations,
    *,
    heartbeat_subset: Sequence[int] | None = None,
    normalize_time: bool = False,
    algo_params: dict | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot example of B-point extraction using the method by Drost et al. (2022) [1].

    The algorithm is implemented as :class:``BPointExtractionDrost2022``.

    Parameters
    ----------
    datapoint : BasePepDatasetWithAnnotations
        Datapoint to plot.
    heartbeat_subset : list of int, optional
        List of heartbeat_ids to plot. If None, all heartbeats are plotted.
    normalize_time : bool, optional
        Whether to normalize the time axis to seconds, starting at 0, or not. Default: False
    algo_params : dict, optional
        Parameters passed to the algorithm instance for C-point and B-point extraction.
        See :class:``pepbench.algorithms.icg.CPointExtractionScipyFindPeaks`` and
            :class:``pepbench.algorithms.icg.BPointExtractionDrost2022`` for available parameters.
        Default: None (i.e., the default parameters of the algorithms are used).
    kwargs : dict
        Additional keyword arguments to pass to the plotting functions. Examples are:
        --- General ---
        * ``fig``, ``ax``: :class:`matplotlib.figure.Figure`, :class:`matplotlib.axes.Axes`
            Figure and Axes objects to plot on; If not provided, a new figure and axes are created.
        * ``figsize``: tuple
            Size of the figure.
        * ``legend_loc``: str
            Location of the legend
        * ``legend_outside``: bool
            Whether to place the legend outside the plot or not.
        * ``legend_orientation``: str
            Orientation of the legend, either "horizontal" or "vertical".
        * ``legend_max_cols``: int
            Maximum number of columns for the legend if ``legend_orientation`` is "horizontal".
        * ``rect``: tuple
            Rectangle coordinates for tight layout, i.e, the bounding box (x0, y0, x1, y1) that the subplots will fit
            into.
        * ``use_tight``: bool
            Whether to use tight layout or not. Default: True
        --- Heartbeat Borders ---
        * ``heartbeat_border_color``: str
            Color of the heartbeat borders.
        --- R-Peaks ---
        * ``r_peak_marker``: str
            Marker style for R-peaks.
        * ``r_peak_linestyle``: str
            Line style for R-peaks.
        * ``r_peak_linewidth``: float
            Line width for R-peaks.
        * ``r_peak_alpha``: float
            Alpha value for the R-peak vertical lines
        * ``r_peak_plot_marker``: bool
            Whether to plot markers at the R-peaks or not.
        * ``r_peak_plot_vline``: bool
            Whether to plot vertical lines at the R-peaks or not.
        --- B-Points ---
        * ``b_point_marker``: str
            Marker style for B-points.
        * ``b_point_linestyle``: str
            Line style for B-points.
        * ``b_point_linewidth``: float
            Line width for B-points.
        * ``b_point_alpha``: float
            Alpha value for B-point vertical lines.


    Return
    ------
    fig : :class:`matplotlib.figure.Figure`
        Figure object.
    axs : list of :class:`matplotlib.axes.Axes`
        list of Axes objects, one for each subplot.

    See Also
    --------
    :class:``pepbench.algorithms.icg.BPointExtractionDrost2022``
        Algorithm implementation.

    References
    ----------
    .. [1] Drost, L., Finke, J. B., Port, J., & Schächinger, H. (2022). Comparison of TWA and PEP as indices of a2- and
        ß-adrenergic activation. Psychopharmacology. https://doi.org/10.1007/s00213-022-06114-8

    """
    kwargs.setdefault("legend_max_cols", 4)
    kwargs.setdefault("legend_outside", True)
    kwargs.setdefault("legend_orientation", "horizontal")
    kwargs.setdefault("legend_loc", _get_legend_loc(kwargs))
    rect = _get_rect(kwargs)

    if algo_params is None:
        algo_params = {}

    heartbeat_subset = _sanitize_heartbeat_subset(heartbeat_subset)
    ecg_data, icg_data = _get_data(datapoint, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset)
    heartbeats = _get_heartbeats(datapoint, heartbeat_subset)
    heartbeat_borders = _get_heartbeat_borders(icg_data, heartbeats)
    algo_params_c_point = {key: val for key, val in algo_params.items() if key in ["window_c_correction"]}
    algo_params_b_point = {key: val for key, val in algo_params.items() if key not in algo_params_c_point}
    c_point_algo = CPointExtractionScipyFindPeaks(**algo_params_c_point)
    c_point_algo.extract(icg=icg_data, heartbeats=heartbeats, sampling_rate_hz=datapoint.sampling_rate_icg)

    b_point_algo = BPointExtractionDrost2022(**algo_params_b_point)
    b_point_algo.extract(
        icg=icg_data, heartbeats=heartbeats, c_points=c_point_algo.points_, sampling_rate_hz=datapoint.sampling_rate_icg
    )

    b_point_samples = b_point_algo.points_["b_point_sample"].dropna().astype(int)
    c_point_samples = c_point_algo.points_["c_point_sample"].dropna().astype(int)
    c_point_minus_150_samples = c_point_samples - int(150 / 1000 * datapoint.sampling_rate_icg)
    c_point_minus_150_samples.name = "c_point_sample_minus_150"
    search_window = pd.concat([c_point_minus_150_samples, c_point_samples], axis=1)

    b_point_samples_reference = _get_reference_labels(datapoint, heartbeat_subset)["b_points"]

    fig, ax = _plot_signals_one_axis(
        df=icg_data,
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        color=cmaps.tech[0],
        **kwargs,
    )

    _add_heartbeat_borders(heartbeats=heartbeat_borders, ax=ax, **kwargs)
    _add_icg_c_points(icg_data, c_point_algo.points_["c_point_sample"].dropna().astype(int), ax=ax, **kwargs)
    _add_icg_c_points(
        icg_data,
        c_point_minus_150_samples,
        ax=ax,
        c_point_color=cmaps.wiso_light[1],
        c_point_label="C-Points - 150 ms",
        **kwargs,
    )
    _add_icg_b_points(
        icg_data,
        b_point_samples_reference,
        ax=ax,
        b_point_label="Reference B-Points",
        b_point_color=cmaps.phil_dark[0],
        **kwargs,
    )
    _add_icg_b_points(icg_data, b_point_samples, ax=ax, b_point_label="Detected B-Points", **kwargs)

    icg_data = icg_data.squeeze()
    for idx, row in search_window.iterrows():
        start_sample = row["c_point_sample_minus_150"]
        end_sample = row["c_point_sample"]
        start = icg_data.index[row["c_point_sample_minus_150"]]
        end = icg_data.index[row["c_point_sample"]]
        c_point_sample = c_point_samples.loc[idx].astype(int)
        start_x = row["c_point_sample_minus_150"]
        start_y = float(icg_data.loc[start])
        c_point_y = float(icg_data.iloc[c_point_sample])

        line_vals = b_point_algo._get_straight_line(start_x, start_y, c_point_sample, c_point_y)
        line_vals.index = icg_data.index[start_x:c_point_sample]
        line_vals.columns = ["Straight Line Connection"]

        icg_slice = icg_data.iloc[start_sample:end_sample]
        distance = line_vals.squeeze().to_numpy() - icg_slice.squeeze().to_numpy()
        b_point_sample = start_sample + np.argmax(distance)

        line_vals.plot(ax=ax, color=cmaps.wiso_dark[1], linestyle="--", linewidth=2)

        ax.annotate(
            "",
            xy=(icg_data.index[b_point_sample], icg_data.iloc[b_point_sample]),
            xytext=(icg_data.index[b_point_sample], line_vals.iloc[np.argmax(distance)].iloc[0]),
            textcoords="data",
            arrowprops={"arrowstyle": "-", "color": cmaps.fau[0], "lw": 2},
            zorder=10,
        )

        ax.annotate(
            r"$d_{max}$",
            xy=(icg_data.index[b_point_sample], line_vals.iloc[np.argmax(distance)].iloc[0]),
            xytext=(-10, -5),
            textcoords="offset points",
            bbox=_get_annotation_bbox_no_edge(),
            ha="right",
            zorder=10,
        )

        ax.axvspan(start, end, color=cmaps.tech_light[0], alpha=0.3, zorder=0, label="B-Point Search Windows")

    _handle_legend_one_axis(fig=fig, ax=ax, **kwargs)

    fig.tight_layout(rect=rect)
    return fig, ax


def plot_b_point_extraction_pale2021(  # noqa: PLR0915
    datapoint: BasePepDatasetWithAnnotations,
    *,
    heartbeat_subset: Sequence[int] | None = None,
    normalize_time: bool = False,
    algo_params: dict | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot example of B-point extraction using the method by Pale et al. (2021) [1].

    The algorithm is implemented as :class:``BPointExtractionPale2021``.

    Parameters
    ----------
    datapoint : BasePepDatasetWithAnnotations
        Datapoint to plot.
    heartbeat_subset : list of int, optional
        List of heartbeat_ids to plot. If None, all heartbeats are plotted.
    normalize_time : bool, optional
        Whether to normalize the time axis to seconds, starting at 0, or not. Default: False
    algo_params : dict, optional
        Parameters passed to the algorithm instance for C-point and B-point extraction.
        See :class:``pepbench.algorithms.icg.CPointExtractionScipyFindPeaks`` and
            :class:``pepbench.algorithms.icg.BPointExtractionPale2021`` for available parameters.
        Default: None (i.e., the default parameters of the algorithms are used).
    kwargs : dict
        Additional keyword arguments to pass to the plotting functions. Examples are:
        --- General ---
        * ``fig``, ``ax``: :class:`matplotlib.figure.Figure`, :class:`matplotlib.axes.Axes`
            Figure and Axes objects to plot on; If not provided, a new figure and axes are created.
        * ``figsize``: tuple
            Size of the figure.
        * ``legend_loc``: str
            Location of the legend
        * ``legend_outside``: bool
            Whether to place the legend outside the plot or not.
        * ``legend_orientation``: str
            Orientation of the legend, either "horizontal" or "vertical".
        * ``legend_max_cols``: int
            Maximum number of columns for the legend if ``legend_orientation`` is "horizontal".
        * ``rect``: tuple
            Rectangle coordinates for tight layout, i.e, the bounding box (x0, y0, x1, y1) that the subplots will fit
            into.
        * ``use_tight``: bool
            Whether to use tight layout or not. Default: True
        --- Heartbeat Borders ---
        * ``heartbeat_border_color``: str
            Color of the heartbeat borders.
        --- R-Peaks ---
        * ``r_peak_marker``: str
            Marker style for R-peaks.
        * ``r_peak_linestyle``: str
            Line style for R-peaks.
        * ``r_peak_linewidth``: float
            Line width for R-peaks.
        * ``r_peak_alpha``: float
            Alpha value for the R-peak vertical lines
        * ``r_peak_plot_marker``: bool
            Whether to plot markers at the R-peaks or not.
        * ``r_peak_plot_vline``: bool
            Whether to plot vertical lines at the R-peaks or not.
        --- B-Points ---
        * ``b_point_marker``: str
            Marker style for B-points.
        * ``b_point_linestyle``: str
            Line style for B-points.
        * ``b_point_linewidth``: float
            Line width for B-points.
        * ``b_point_alpha``: float
            Alpha value for B-point vertical lines.


    Return
    ------
    fig : :class:`matplotlib.figure.Figure`
        Figure object.
    axs : list of :class:`matplotlib.axes.Axes`
        list of Axes objects, one for each subplot.

    See Also
    --------
    :class:``pepbench.algorithms.icg.BPointExtractionMiljkovic2022``
        Algorithm implementation.

    References
    ----------
    .. [1] Pale, U., Muller, N., Arza, A., & Atienza, D. (2021). ReBeatICG: Real-time Low-Complexity Beat-to-beat
        Impedance Cardiogram Delineation Algorithm. 2021 43rd Annual International Conference of the IEEE Engineering
        in Medicine & Biology Society (EMBC), 5618-5624. https://doi.org/10.1109/EMBC46164.2021.9630170

    """
    fig, axs = plt.subplots(nrows=2, sharex=True, **kwargs)
    kwargs.setdefault("legend_max_cols", 4)
    kwargs.setdefault("legend_outside", True)
    kwargs.setdefault("legend_orientation", "horizontal")
    kwargs.setdefault("legend_loc", _get_legend_loc(kwargs))
    kwargs.setdefault("rect", (0, 0, 1.0, 0.85))
    rect = _get_rect(kwargs)

    if algo_params is None:
        algo_params = {}

    heartbeat_subset = _sanitize_heartbeat_subset(heartbeat_subset)
    ecg_data, icg_data = _get_data(datapoint, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset)
    heartbeats = _get_heartbeats(datapoint, heartbeat_subset)

    heartbeat_borders = _get_heartbeat_borders(icg_data, heartbeats)

    b_point_samples_reference = _get_reference_labels(datapoint, heartbeat_subset)["b_points"]

    algo_params_c_point = {key: val for key, val in algo_params.items() if key in ["window_c_correction"]}
    algo_params_b_point = {key: val for key, val in algo_params.items() if key not in algo_params_c_point}
    c_point_algo = CPointExtractionScipyFindPeaks(**algo_params_c_point)
    c_point_algo.extract(icg=icg_data, heartbeats=heartbeats, sampling_rate_hz=datapoint.sampling_rate_icg)

    b_point_algo = BPointExtractionPale2021(**algo_params_b_point)
    b_point_algo.extract(
        icg=icg_data, heartbeats=heartbeats, c_points=c_point_algo.points_, sampling_rate_hz=datapoint.sampling_rate_icg
    )

    b_point_samples = b_point_algo.points_["b_point_sample"].dropna().astype(int)
    c_point_samples = c_point_algo.points_["c_point_sample"].dropna().astype(int)

    search_window_start_samples = c_point_samples - int(80 / 1000 * datapoint.sampling_rate_icg)
    search_window_start_samples.name = "search_window_start"

    c_point_amplitude_fraction = algo_params_b_point.get("c_point_amplitude_fraction", 0.5)
    b_point_slope_threshold_01 = algo_params_b_point.get("b_point_slope_threshold_01", 0.11)
    b_point_slope_threshold_02 = algo_params_b_point.get("b_point_slope_threshold_02", 0.08)

    b_point_candidates = []

    icg_der = pd.DataFrame(
        np.gradient(icg_data.squeeze()), index=icg_data.index, columns=["ICG 2nd Deriv. ($d^2Z/dt^2$)"]
    )

    for idx, data in heartbeats.iterrows():
        # Get the C-Point location at the current heartbeat id
        c_point = c_point_samples[idx]
        search_window_start = search_window_start_samples[idx]

        # end of the search window is the closest point before the C-point with amplitude less than
        # c_point_amplitude_fraction * c_point
        c_point_amplitude_threshold = icg_data.iloc[c_point] * c_point_amplitude_fraction

        search_window_end = np.where(icg_data.iloc[search_window_start:c_point] < c_point_amplitude_threshold)[0]
        # If no point is found, use the C-point as the end of the search window; otherwise, use the last point
        # before the C-point that meets the condition
        search_window_end = c_point if search_window_end.size == 0 else search_window_end[-1] + search_window_start

        icg_slice = icg_data.iloc[search_window_start:search_window_end]

        b_point_min = data["start_sample"] + np.argmin(icg_slice)

        # candidate 1: search for the local minimum closest to the C-point
        icg_der_slice = icg_der.iloc[search_window_start:search_window_end]
        # find zero crossings in the derivative
        zero_crossings = np.where(np.diff(np.signbit(icg_der_slice)))[0]
        # candidate 2: search for the first point at which the slope exceeds the threshold; the slope is already
        # calculated in the derivative
        slope_exceeds_threshold = np.where(icg_der_slice > b_point_slope_threshold_01)[0]
        # if no slope exceeds the threshold, use the second threshold
        if slope_exceeds_threshold.size == 0:
            slope_exceeds_threshold = np.where(icg_der_slice > b_point_slope_threshold_02)[0]
        # concatenate and sort the candidates
        candidates = np.sort(np.concatenate((zero_crossings, slope_exceeds_threshold)))

        if candidates.size == 0:
            b_point_candidates.append([b_point_min])
        else:
            b_point_candidates.append(search_window_start + candidates)

        start = icg_data.index[search_window_start]
        end = icg_data.index[search_window_end]
        axs[0].axvspan(
            start,
            end,
            color=cmaps.tech_light[0],
            alpha=0.3,
            zorder=0,
            label="B-Point Search Windows",
        )
        axs[1].axvspan(
            start,
            end,
            color=cmaps.tech_light[0],
            alpha=0.3,
            zorder=0,
            label="B-Point Search Windows",
        )

        axs[0].hlines(
            y=np.squeeze(c_point_amplitude_threshold),
            xmin=start,
            xmax=end,
            color=cmaps.nat[0],
            zorder=5,
            ls="--",
            label="C-Point Amplitude Threshold",
        )

        axs[1].hlines(
            y=b_point_slope_threshold_01,
            xmin=start,
            xmax=end,
            color=cmaps.nat_dark[0],
            zorder=5,
            ls="--",
            label="ICG 2nd Der. ($d^2Z/dt^2$) Thres. 01",
        )

        axs[1].hlines(
            y=b_point_slope_threshold_02,
            xmin=start,
            xmax=end,
            color=cmaps.nat_dark[1],
            zorder=5,
            ls="--",
            label="ICG 2nd Der. ($d^2Z/dt^2$) Thres. 02",
        )

    b_point_candidates = np.concatenate(b_point_candidates)

    _plot_signals_one_axis(
        df=icg_data,
        ax=axs[0],
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        color=cmaps.tech[0],
        columns=["ICG ($dZ/dt$)"],
        **kwargs,
    )

    _plot_signals_one_axis(
        df=icg_der,
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        plot_ecg=False,
        ax=axs[1],
        color=cmaps.fau_light[0],
        **kwargs,
    )

    _add_heartbeat_borders(heartbeats=heartbeat_borders, ax=axs[0], **kwargs)
    _add_heartbeat_borders(heartbeats=heartbeat_borders, ax=axs[1], **kwargs)

    _add_icg_c_points(
        icg_data,
        c_point_samples,
        ax=axs[0],
        **kwargs,
    )

    _add_icg_b_points(
        icg_data,
        b_point_samples_reference,
        ax=axs[0],
        b_point_label="Reference B-Points",
        b_point_color=cmaps.phil_dark[0],
        **kwargs,
    )
    _add_icg_b_points(
        icg_data,
        b_point_samples,
        ax=axs[0],
        b_point_label="Detected B-Points",
        **kwargs,
    )
    _add_icg_b_points(
        icg_der,
        b_point_candidates,
        b_point_label="B-Point Candidates",
        b_point_color=cmaps.phil[2],
        ax=axs[1],
        **kwargs,
    )

    _handle_legend_two_axes(fig=fig, axs=axs, **kwargs)

    if normalize_time or not isinstance(icg_data.index, pd.DatetimeIndex):
        axs[1].set_xlabel("Time [s]")

    fig.tight_layout(rect=rect)
    return fig, axs


def plot_b_point_extraction_miljkovic2022(  # noqa: PLR0915
    datapoint: BasePepDatasetWithAnnotations,
    *,
    heartbeat_subset: Sequence[int] | None = None,
    normalize_time: bool = False,
    algo_params: dict | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot example of B-point extraction using the method by Miljkovic and Sekara (2022) [1].

    The algorithm is implemented as :class:``BPointExtractionMiljkovic2022``.

    Parameters
    ----------
    datapoint : BasePepDatasetWithAnnotations
        Datapoint to plot.
    heartbeat_subset : list of int, optional
        List of heartbeat_ids to plot. If None, all heartbeats are plotted.
    normalize_time : bool, optional
        Whether to normalize the time axis to seconds, starting at 0, or not. Default: False
    algo_params : dict, optional
        Parameters passed to the algorithm instance for C-point and B-point extraction.
        See :class:``pepbench.algorithms.icg.CPointExtractionScipyFindPeaks`` and
            :class:``pepbench.algorithms.icg.BPointExtractionMiljkovic2022`` for available parameters.
        Default: None (i.e., the default parameters of the algorithms are used).
    kwargs : dict
        Additional keyword arguments to pass to the plotting functions. Examples are:
        --- General ---
        * ``fig``, ``ax``: :class:`matplotlib.figure.Figure`, :class:`matplotlib.axes.Axes`
            Figure and Axes objects to plot on; If not provided, a new figure and axes are created.
        * ``figsize``: tuple
            Size of the figure.
        * ``legend_loc``: str
            Location of the legend
        * ``legend_outside``: bool
            Whether to place the legend outside the plot or not.
        * ``legend_orientation``: str
            Orientation of the legend, either "horizontal" or "vertical".
        * ``legend_max_cols``: int
            Maximum number of columns for the legend if ``legend_orientation`` is "horizontal".
        * ``rect``: tuple
            Rectangle coordinates for tight layout, i.e, the bounding box (x0, y0, x1, y1) that the subplots will fit
            into.
        * ``use_tight``: bool
            Whether to use tight layout or not. Default: True
        --- Heartbeat Borders ---
        * ``heartbeat_border_color``: str
            Color of the heartbeat borders.
        --- R-Peaks ---
        * ``r_peak_marker``: str
            Marker style for R-peaks.
        * ``r_peak_linestyle``: str
            Line style for R-peaks.
        * ``r_peak_linewidth``: float
            Line width for R-peaks.
        * ``r_peak_alpha``: float
            Alpha value for the R-peak vertical lines
        * ``r_peak_plot_marker``: bool
            Whether to plot markers at the R-peaks or not.
        * ``r_peak_plot_vline``: bool
            Whether to plot vertical lines at the R-peaks or not.
        --- B-Points ---
        * ``b_point_marker``: str
            Marker style for B-points.
        * ``b_point_linestyle``: str
            Line style for B-points.
        * ``b_point_linewidth``: float
            Line width for B-points.
        * ``b_point_alpha``: float
            Alpha value for B-point vertical lines.


    Return
    ------
    fig : :class:`matplotlib.figure.Figure`
        Figure object.
    axs : list of :class:`matplotlib.axes.Axes`
        list of Axes objects, one for each subplot.

    See Also
    --------
    :class:``pepbench.algorithms.icg.BPointExtractionMiljkovic2022``
        Algorithm implementation.

    References
    ----------
    .. [1] Miljković, N., & Šekara, T. B. (2022). A New Weighted Time Window-based Method to Detect B-point in
        Impedance Cardiogram (Version 3). arXiv. https://doi.org/10.48550/ARXIV.2207.04490

    """
    fig, axs = plt.subplots(nrows=2, sharex=True, **kwargs)
    kwargs.setdefault("legend_max_cols", 4)
    kwargs.setdefault("legend_outside", True)
    kwargs.setdefault("legend_orientation", "horizontal")
    kwargs.setdefault("legend_loc", _get_legend_loc(kwargs))
    kwargs.setdefault("rect", (0, 0, 1.0, 0.85))
    rect = _get_rect(kwargs)

    if algo_params is None:
        algo_params = {}

    heartbeat_subset = _sanitize_heartbeat_subset(heartbeat_subset)
    ecg_data, icg_data = _get_data(datapoint, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset)
    heartbeats = _get_heartbeats(datapoint, heartbeat_subset)
    heartbeat_borders = _get_heartbeat_borders(icg_data, heartbeats)

    b_point_samples_reference = _get_reference_labels(datapoint, heartbeat_subset)["b_points"]

    algo_params_c_point = {key: val for key, val in algo_params.items() if key in ["window_c_correction"]}
    algo_params_b_point = {key: val for key, val in algo_params.items() if key not in algo_params_c_point}
    c_point_algo = CPointExtractionScipyFindPeaks(**algo_params_c_point)
    c_point_algo.extract(icg=icg_data, heartbeats=heartbeats, sampling_rate_hz=datapoint.sampling_rate_icg)

    b_point_algo = BPointExtractionMiljkovic2022(**algo_params_b_point)
    b_point_algo.extract(
        icg=icg_data, heartbeats=heartbeats, c_points=c_point_algo.points_, sampling_rate_hz=datapoint.sampling_rate_icg
    )

    b_point_samples = b_point_algo.points_["b_point_sample"].dropna().astype(int)
    c_point_samples = c_point_algo.points_["c_point_sample"].dropna().astype(int)

    c_point_minus_250_samples = c_point_samples - int(250 / 1000 * datapoint.sampling_rate_icg)
    c_point_minus_250_samples.name = "c_point_sample_minus_250"
    search_window = pd.concat([c_point_minus_250_samples, c_point_samples], axis=1)

    icg_data = icg_data.squeeze()

    alpha = -0.1
    search_signal = pd.Series(index=icg_data.index, name="Transformed Search Signal")
    search_signal.iloc[:] = alpha

    window_signal = pd.Series(index=icg_data.index, name="Weighted Window")
    window_signal.iloc[:] = 0

    for _idx, row in search_window.iterrows():
        start_sample = row["c_point_sample_minus_250"]
        end_sample = row["c_point_sample"]
        icg_slice = icg_data.iloc[start_sample:end_sample].reset_index(drop=True)

        idx_start = icg_slice.idxmin()
        idx_stop = icg_slice.idxmax()
        icg_slice_window = icg_slice[idx_start:idx_stop]

        height = icg_slice.max() - icg_slice.min()

        # shift the segment so that the minimal value equals zero
        icg_slice -= icg_slice.min()

        window = np.ones(shape=(len(icg_slice),))
        window *= alpha
        window_slope = np.linspace(alpha + height, 0, num=len(icg_slice_window) + 1, endpoint=True)
        window[idx_stop - (idx_stop - idx_start) : idx_stop + 1] = window_slope

        icg_slice = icg_slice * window

        window_signal.iloc[start_sample:end_sample] = window
        search_signal.iloc[start_sample:end_sample] = icg_slice

    # scale the window signal to be in the range of the search signal
    window_signal /= window_signal.max()
    window_signal *= search_signal.max()

    _plot_signals_one_axis(
        df=icg_data,
        ax=axs[0],
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        color=cmaps.tech[0],
        columns=["ICG ($dZ/dt$)"],
        **kwargs,
    )

    _plot_signals_one_axis(
        df=search_signal,
        ax=axs[1],
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        color=cmaps.tech_dark[0],
        **kwargs,
    )

    _plot_signals_one_axis(
        df=window_signal,
        ax=axs[1],
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        color=cmaps.tech_dark[1],
        **kwargs,
    )

    _add_heartbeat_borders(heartbeats=heartbeat_borders, ax=axs[0], **kwargs)
    _add_heartbeat_borders(heartbeats=heartbeat_borders, ax=axs[1], **kwargs)

    _add_icg_c_points(
        icg_data,
        c_point_samples,
        ax=axs[0],
        **kwargs,
    )
    _add_icg_c_points(
        icg_data,
        c_point_minus_250_samples,
        ax=axs[0],
        c_point_color=cmaps.wiso_light[1],
        c_point_label="C-Points - 250 ms",
        **kwargs,
    )
    _add_icg_b_points(
        icg_data,
        b_point_samples_reference,
        ax=axs[0],
        b_point_label="Reference B-Points",
        b_point_color=cmaps.phil_dark[0],
        **kwargs,
    )
    _add_icg_b_points(
        icg_data,
        b_point_samples,
        ax=axs[0],
        b_point_label="Detected B-Points",
        **kwargs,
    )

    # get zero crossings of icg
    zero_crossings = np.where(np.diff(np.signbit(icg_data)))[0]

    for idx, row in search_window.iterrows():
        start_sample = row["c_point_sample_minus_250"]
        end_sample = row["c_point_sample"]
        start = icg_data.index[row["c_point_sample_minus_250"]]
        end = icg_data.index[row["c_point_sample"]]

        icg_slice = icg_data.iloc[start_sample:end_sample].reset_index(drop=True)

        idx_start = icg_slice.idxmin()
        idx_stop = icg_slice.idxmax()
        icg_slice_window = icg_slice[idx_start:idx_stop]

        height = icg_slice.max() - icg_slice.min()

        # shift the segment so that the minimal value equals zero
        icg_slice -= icg_slice.min()

        window = np.ones(shape=(len(icg_slice),))
        window *= alpha
        window_slope = np.linspace(alpha + height, 0, num=len(icg_slice_window) + 1, endpoint=True)
        window[idx_stop - (idx_stop - idx_start) : idx_stop + 1] = window_slope

        icg_slice = icg_slice * window

        search_signal.iloc[start_sample:end_sample] = icg_slice

        # peak detection on the transformed signal with minimal peak distance of 50ms and a height threshold of the
        # maximum value divided by 2000
        peaks, heights = find_peaks(
            icg_slice, distance=int(0.05 * datapoint.sampling_rate_icg), height=icg_slice.max() / 2000
        )
        heights = heights["peak_heights"]

        c_point = c_point_samples[idx]
        start_window = c_point_minus_250_samples[idx]

        _add_icg_b_points(
            search_signal,
            peaks + start_sample,
            ax=axs[1],
            b_point_label="Cand. Peaks",
            b_point_marker="X",
            b_point_color=cmaps.med[1],
            **kwargs,
        )

        if len(peaks) == 1:
            # get the closest zero crossing *before* the C-point
            zero_crossings_diff = zero_crossings - c_point
            zero_crossings_diff = zero_crossings_diff[zero_crossings_diff < 0]
            zero_crossing_idx = np.argmax(zero_crossings_diff)
            b_point = zero_crossings[zero_crossing_idx]

            _add_icg_b_points(
                search_signal,
                b_point,
                ax=axs[1],
                b_point_label="<2 Peaks: Zero Crossing C-Point",
                b_point_marker="X",
                b_point_color=cmaps.phil[0],
                **kwargs,
            )
        else:
            # get the two highest peaks
            peaks = peaks[-2:]
            # define the b_point as the minimum between the two highest peaks
            search_window = icg_slice[peaks[0] : peaks[-1]]
            b_point = np.argmin(search_window) + peaks[0]
            b_point = start_window + b_point

            _add_icg_b_points(
                search_signal,
                b_point,
                ax=axs[1],
                b_point_label=">=2 Peaks: Min Bet. 2 Highest Peaks",
                b_point_marker="X",
                b_point_color=cmaps.phil[1],
                **kwargs,
            )

        #
        #     ax.annotate(
        #         r"$d_{max}$",
        #         xy=(icg_data.index[b_point_sample], line_vals.iloc[np.argmax(distance)].iloc[0]),
        #         xytext=(-10, -5),
        #         textcoords="offset points",
        #         bbox=_get_annotation_bbox_no_edge(),
        #         ha="right",
        #         zorder=10,
        #     )
        #
        axs[0].axvspan(start, end, color=cmaps.tech_light[0], alpha=0.3, zorder=0, label="B-Point Search Windows")
        axs[1].axvspan(start, end, color=cmaps.tech_light[0], alpha=0.3, zorder=0, label="B-Point Search Windows")

    _handle_legend_two_axes(fig=fig, axs=axs, **kwargs)

    if normalize_time or not isinstance(icg_data.index, pd.DatetimeIndex):
        axs[1].set_xlabel("Time [s]")

    fig.tight_layout(rect=rect)
    return fig, axs


def plot_b_point_extraction_forouzanfar2018(  # noqa: PLR0915
    datapoint: BasePepDatasetWithAnnotations,
    *,
    heartbeat_subset: Sequence[int] | None = None,
    normalize_time: bool = False,
    algo_params: dict | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot example of B-point extraction using the method by Forouzanfar et al. (2018) [1].

    The algorithm is implemented as :class:``BPointExtractionForouzanfar2018``.

    Parameters
    ----------
    datapoint : BasePepDatasetWithAnnotations
        Datapoint to plot.
    heartbeat_subset : list of int, optional
        List of heartbeat_ids to plot. If None, all heartbeats are plotted.
    normalize_time : bool, optional
        Whether to normalize the time axis to seconds, starting at 0, or not. Default: False
    algo_params : dict, optional
        Parameters passed to the algorithm instance for C-point and B-point extraction.
        See :class:``pepbench.algorithms.icg.CPointExtractionScipyFindPeaks`` and
            :class:``pepbench.algorithms.icg.BPointExtractionDrost2022`` for available parameters.
        Default: None (i.e., the default parameters of the algorithms are used).
    kwargs : dict
        Additional keyword arguments to pass to the plotting functions. Examples are:
        --- General ---
        * ``figsize``: tuple
            Size of the figure.
        * ``legend_loc``: str
            Location of the legend
        * ``legend_outside``: bool
            Whether to place the legend outside the plot or not.
        * ``legend_orientation``: str
            Orientation of the legend, either "horizontal" or "vertical".
        * ``legend_max_cols``: int
            Maximum number of columns for the legend if ``legend_orientation`` is "horizontal".
        * ``rect``: tuple
            Rectangle coordinates for tight layout, i.e, the bounding box (x0, y0, x1, y1) that the subplots will fit
            into.
        * ``use_tight``: bool
            Whether to use tight layout or not. Default: True
        --- Heartbeat Borders ---
        * ``heartbeat_border_color``: str
            Color of the heartbeat borders.
        --- R-Peaks ---
        * ``r_peak_marker``: str
            Marker style for R-peaks.
        * ``r_peak_linestyle``: str
            Line style for R-peaks.
        * ``r_peak_linewidth``: float
            Line width for R-peaks.
        * ``r_peak_alpha``: float
            Alpha value for the R-peak vertical lines
        * ``r_peak_plot_marker``: bool
            Whether to plot markers at the R-peaks or not.
        * ``r_peak_plot_vline``: bool
            Whether to plot vertical lines at the R-peaks or not.
        --- B-Points ---
        * ``b_point_marker``: str
            Marker style for B-points.
        * ``b_point_linestyle``: str
            Line style for B-points.
        * ``b_point_linewidth``: float
            Line width for B-points.
        * ``b_point_alpha``: float
            Alpha value for B-point vertical lines.


    Return
    ------
    fig : :class:`matplotlib.figure.Figure`
        Figure object.
    axs : list of :class:`matplotlib.axes.Axes`
        list of Axes objects, one for each subplot.

    See Also
    --------
    :class:``pepbench.algorithms.icg.BPointExtractionForouzanfar2018``
        Algorithm implementation.

    References
    ----------
    .. [1] Forouzanfar, M., Baker, F. C., De Zambotti, M., McCall, C., Giovangrandi, L., & Kovacs, G. T. A. (2018).
        Toward a better noninvasive assessment of preejection period: A novel automatic algorithm for B-point detection
        and correction on thoracic impedance cardiogram. Psychophysiology, 55(8), e13072.
        https://doi.org/10.1111/psyp.13072

    """
    fig, axs = plt.subplots(nrows=2, sharex=True, **kwargs)
    kwargs.setdefault("legend_outside", True)
    kwargs.setdefault("legend_orientation", "horizontal")
    kwargs.setdefault("legend_loc", _get_legend_loc(kwargs))
    kwargs.setdefault("legend_max_cols", 4)
    kwargs.setdefault("rect", (0, 0, 1, 0.8))
    rect = _get_rect(kwargs)

    if algo_params is None:
        algo_params = {}

    heartbeat_subset = _sanitize_heartbeat_subset(heartbeat_subset)
    ecg_data, icg_data = _get_data(datapoint, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset)
    heartbeats = _get_heartbeats(datapoint, heartbeat_subset)
    heartbeat_borders = _get_heartbeat_borders(icg_data, heartbeats)

    algo_params_c_point = {key: val for key, val in algo_params.items() if key in ["window_c_correction"]}
    algo_params_b_point = {key: val for key, val in algo_params.items() if key not in algo_params_c_point}
    c_point_algo = CPointExtractionScipyFindPeaks(**algo_params_c_point)
    c_point_algo.extract(icg=icg_data, heartbeats=heartbeats, sampling_rate_hz=datapoint.sampling_rate_icg)

    b_point_algo = BPointExtractionForouzanfar2018(**algo_params_b_point)
    b_point_algo.extract(
        icg=icg_data, heartbeats=heartbeats, c_points=c_point_algo.points_, sampling_rate_hz=datapoint.sampling_rate_icg
    )

    icg_data = icg_data.squeeze()
    icg_2nd_der = np.gradient(icg_data)
    icg_3rd_der = np.gradient(icg_2nd_der)
    icg_2nd_der = pd.DataFrame(icg_2nd_der, index=icg_data.index, columns=["ICG 2nd Deriv. $(d^2Z/dt^2)$"])
    icg_3rd_der = pd.DataFrame(icg_3rd_der, index=icg_data.index, columns=["ICG 3rd Deriv. $(d^3Z/dt^3)$"])

    c_point_samples = c_point_algo.points_["c_point_sample"].dropna().astype(int)
    b_point_samples_reference = _get_reference_labels(datapoint, heartbeat_subset)["b_points"]

    _plot_signals_one_axis(
        df=icg_data,
        ax=axs[0],
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        color=cmaps.tech[0],
        **kwargs,
    )

    # normalize 3rd der to have the same scale as the 2nd der
    icg_3rd_der = icg_3rd_der / float(icg_3rd_der.abs().max().iloc[0]) * float(icg_2nd_der.abs().max().iloc[0])

    _plot_signals_one_axis(
        df=icg_3rd_der,
        ax=axs[1],
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        color=cmaps.tech_dark[0],
        **kwargs,
    )
    # _plot_signals_one_axis(
    #     df=icg_2nd_der,
    #     ax=axs[1],
    #     normalize_time=normalize_time,
    #     heartbeat_subset=heartbeat_subset,
    #     color=cmaps.tech_dark[0],
    #     **kwargs,
    # )
    # axs[0].axhline(0, color="black", linestyle="--", linewidth=1, zorder=0)
    axs[1].axhline(0, color="black", linestyle="--", linewidth=1, zorder=0)

    _add_heartbeat_borders(heartbeats=heartbeat_borders, ax=axs[0], **kwargs)
    _add_heartbeat_borders(heartbeats=heartbeat_borders, ax=axs[1], **kwargs)

    _add_icg_c_points(icg_data, c_point_samples, ax=axs[0], **kwargs)
    _add_icg_c_points(icg_2nd_der, c_point_samples, ax=axs[1], **kwargs)

    for idx, _row in heartbeats[1:].iterrows():
        if np.isnan(c_point_samples[idx]) or np.isnan(c_point_samples[idx - 1]):
            continue

        c_point = c_point_samples.loc[idx]
        # Compute the beat to beat interval
        c_point_b2b = c_point_samples.loc[idx] - c_point_samples.loc[idx - 1]
        search_interval = int(c_point_b2b / 3)
        start = c_point - search_interval

        axs[0].axvspan(
            icg_data.index[start],
            icg_data.index[c_point],
            color=cmaps.tech_light[0],
            alpha=0.3,
            zorder=0,
            label="A-Point Search Windows",
        )
        axs[1].axvspan(
            icg_data.index[start],
            icg_data.index[c_point],
            color=cmaps.tech_light[0],
            alpha=0.3,
            zorder=0,
            label="A-Point Search Windows",
        )

        # Detect the local minimum (A-Point) within one third of the beat to beat interval prior to the C-Point
        a_point = b_point_algo._get_a_point(icg_data, search_interval, c_point) + (c_point - search_interval)

        icg_segment = icg_data.iloc[a_point : c_point + 1]
        # icg_2nd_der_segment = icg_2nd_der.iloc[a_point : c_point + 1]
        c_amplitude = icg_data.iloc[c_point]

        # Get the most prominent monotonic increasing segment between the A-Point and the C-Point
        start_sample, end_sample = b_point_algo._get_most_prominent_monotonic_increasing_segment(
            icg_segment, c_amplitude
        )

        start_sample += a_point
        end_sample += a_point
        icg_monotonic_increasing_segment = icg_data.iloc[start_sample : end_sample + 1]
        icg_monotonic_increasing_segment.name = "Monotonic Increasing Segment"

        if (start_sample == a_point) & (end_sample == a_point):
            # no monotonic increasing segment found
            continue

        # Next step: get the first third of the monotonic increasing segment
        start = start_sample
        end = end_sample - int((2 / 3) * (end_sample - start_sample))

        axs[0].axvspan(
            icg_data.index[start],
            icg_data.index[end],
            color=cmaps.fau_light[0],
            alpha=0.3,
            zorder=0,
            label="Zero Crossing Search Windows",
        )
        axs[1].axvspan(
            icg_data.index[start],
            icg_data.index[end],
            color=cmaps.fau_light[0],
            alpha=0.3,
            zorder=0,
            label="Zero Crossing Search Windows",
        )

        # 2nd derivative of the segment
        monotonic_segment_2nd_der = icg_2nd_der.iloc[start:end]
        monotonic_segment_2nd_der.columns = ["2nd_der"]
        # 3rd derivative of the segment
        monotonic_segment_3rd_der = icg_3rd_der.iloc[start:end]
        monotonic_segment_3rd_der.columns = ["3rd_der"]

        # Calculate the amplitude difference between the C-Point and the A-Point
        height = icg_data.iloc[c_point] - icg_data.iloc[a_point]

        # Compute the significant zero_crossings
        significant_zero_crossings = b_point_algo._get_zero_crossings_3rd_derivative(
            monotonic_segment_3rd_der, monotonic_segment_2nd_der, height
        )
        significant_zero_crossings += start

        # Compute the significant local maximums of the 3rd derivative of the most prominent monotonic segment
        significant_local_maximums = b_point_algo._get_local_maxima_3rd_derivative(monotonic_segment_3rd_der, height)
        significant_local_maximums += start

        # Label the last zero crossing/ local maximum as the B-Point
        # If there are no zero crossings or local maximums use the first Point of the segment as B-Point
        significant_features = pd.concat([significant_zero_crossings, significant_local_maximums], axis=0)
        b_point = significant_features.iloc[np.argmin(c_point - significant_features)].iloc[0]

        icg_monotonic_increasing_segment.plot(ax=axs[0], color=cmaps.fau[0])

        _add_icg_c_points(
            icg_data,
            a_point,
            ax=axs[0],
            c_point_color=cmaps.wiso_light[1],
            c_point_label="A-Points",
            **kwargs,
        )
        _add_icg_c_points(
            icg_2nd_der,
            a_point,
            ax=axs[1],
            c_point_color=cmaps.wiso_light[1],
            c_point_label="A-Points",
            **kwargs,
        )

        _add_icg_b_points(
            icg_3rd_der,
            significant_zero_crossings.squeeze(),
            ax=axs[1],
            b_point_label="$d^3Z/dt^3$ Zero Crossings",
            b_point_color=cmaps.med[0],
            b_point_marker="X",
            **kwargs,
        )
        _add_icg_b_points(
            icg_3rd_der,
            significant_local_maximums.squeeze(),
            ax=axs[1],
            b_point_label="$d^3Z/dt^3$ Local Max.",
            b_point_color=cmaps.nat[0],
            b_point_marker="X",
            **kwargs,
        )

        _add_icg_b_points(
            icg_data,
            b_point_samples_reference,
            ax=axs[0],
            b_point_label="Reference B-Points",
            b_point_color=cmaps.phil_dark[0],
            **kwargs,
        )
        _add_icg_b_points(
            icg_data,
            b_point,
            ax=axs[0],
            b_point_label="Detected B-Points",
            **kwargs,
        )

    _handle_legend_two_axes(fig=fig, axs=axs, **kwargs)

    # set new xlims, drop first heartbeat as it is not used
    heartbeats = heartbeats.iloc[1:]
    x_start = icg_data.index[heartbeats.iloc[0]["start_sample"] - 10]
    for ax in axs:
        ax.set_xlim(x_start, None)

    fig.tight_layout(rect=rect)
    fig.align_ylabels()

    return fig, axs
