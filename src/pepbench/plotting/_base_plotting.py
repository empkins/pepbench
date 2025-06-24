from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
from biopsykit.signals._base_extraction import BaseExtraction
from biopsykit.signals.ecg.event_extraction import BaseEcgExtractionWithHeartbeats
from biopsykit.signals.icg.event_extraction import BaseBPointExtraction, CPointExtractionScipyFindPeaks
from fau_colors import cmaps
from matplotlib import pyplot as plt
from matplotlib import transforms
from scipy import stats

from pepbench.datasets import BasePepDataset, BasePepDatasetWithAnnotations
from pepbench.plotting._utils import (
    _add_ecg_q_peak_artefacts,
    _add_ecg_q_peaks,
    _add_heartbeat_borders,
    _add_icg_b_point_artefacts,
    _add_icg_b_points,
    _add_pep_from_reference,
    _add_pep_from_results,
    _get_bbox_coords,
    _get_data,
    _get_fig_ax,
    _get_fig_axs,
    _get_heartbeats,
    _get_labels_from_challenge_results,
    _get_legend_loc,
    _get_rect,
    _get_reference_labels,
    _handle_legend_one_axis,
    _handle_legend_two_axes,
    _sanitize_heartbeat_subset,
    add_fancy_patch_around,
)

__all__ = [
    "_plot_blandaltman",
    "_plot_paired",
    "plot_signals",
    "plot_signals_from_challenge_results",
    "plot_signals_with_algorithm_results",
    "plot_signals_with_reference_labels",
    "plot_signals_with_reference_pep",
]


def plot_signals(
    datapoint: BasePepDataset,
    *,
    collapse: bool = False,
    normalize_time: bool = False,
    heartbeat_subset: Sequence[int] | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes | Sequence[plt.Axes]]:
    """Plot ECG and ICG signals.

    Parameters
    ----------
    datapoint : :class:`~pepbench.datasets.BasePepDataset`
        Dataset to plot.
    collapse : bool, optional
        If ``True``, plot ECG and ICG signals in one axis. If ``False``, plot ECG and ICG signals in two axes.
        Default: ``False``.
    normalize_time : bool, optional
        If ``True``, normalize time to seconds. If ``False``, use the original time format.
        Default: ``False``.
    heartbeat_subset : list of int, optional
        List of heartbeats (as indices) to plot. If ``None``, plot all heartbeats.
        Default: ``None``.
    kwargs : dict, optional
        Additional keyword arguments.

    """
    if collapse:
        return _plot_signals_one_axis(
            datapoint=datapoint,
            normalize_time=normalize_time,
            heartbeat_subset=heartbeat_subset,
            **kwargs,
        )
    return _plot_signals_two_axes(
        datapoint=datapoint,
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        **kwargs,
    )


def plot_signals_with_reference_labels(  # noqa: C901
    datapoint: BasePepDatasetWithAnnotations,
    *,
    heartbeat_subset: Sequence[int] | None = None,
    collapse: bool = False,
    normalize_time: bool = False,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes | Sequence[plt.Axes]]:
    kwargs.setdefault("sharex", True)
    kwargs.setdefault("legend_max_cols", 6)
    kwargs.setdefault("legend_loc", _get_legend_loc(kwargs))
    plot_ecg = kwargs.get("plot_ecg", True)
    plot_icg = kwargs.get("plot_icg", True)
    plot_artefacts = kwargs.get("plot_artefacts", False)
    rect = _get_rect(kwargs)

    fig, ax = plot_signals(
        datapoint,
        collapse=collapse,
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        **kwargs,
    )
    ecg_data, icg_data = _get_data(datapoint, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset)

    reference_labels = _get_reference_labels(datapoint, heartbeat_subset=heartbeat_subset)
    heartbeats = reference_labels["heartbeats"]
    q_peaks = reference_labels["q_peaks"]
    q_peak_artefacts = reference_labels["q_peak_artefacts"]
    b_points = reference_labels["b_points"]
    b_point_artefacts = reference_labels["b_point_artefacts"]

    # plot q-peak onsets and b-points
    if collapse:
        _add_heartbeat_borders(ecg_data.index[list(heartbeats["start_sample"])], ax, **kwargs)
        if plot_ecg:
            _add_ecg_q_peaks(ecg_data, q_peaks, ax, **kwargs)
        if plot_icg:
            _add_icg_b_points(icg_data, b_points, ax, **kwargs)
        if plot_artefacts:
            if not q_peak_artefacts.empty:
                _add_ecg_q_peak_artefacts(ecg_data, q_peak_artefacts, ax, **kwargs)
            if not b_point_artefacts.empty:
                _add_icg_b_point_artefacts(icg_data, b_point_artefacts, ax, **kwargs)

        _handle_legend_one_axis(fig, ax, **kwargs)
    else:
        _add_heartbeat_borders(ecg_data.index[list(heartbeats["start_sample"])], ax[0], **kwargs)
        _add_heartbeat_borders(ecg_data.index[list(heartbeats["start_sample"])], ax[1], **kwargs)
        if plot_ecg:
            _add_ecg_q_peaks(ecg_data, q_peaks, ax[0], **kwargs)
        if plot_icg:
            _add_icg_b_points(icg_data, b_points, ax[1], **kwargs)
        if plot_artefacts:
            if not q_peak_artefacts.empty:
                _add_ecg_q_peak_artefacts(ecg_data, q_peak_artefacts, ax[0], **kwargs)
            if not b_point_artefacts.empty:
                _add_icg_b_point_artefacts(icg_data, b_point_artefacts, ax[1], **kwargs)

        _handle_legend_two_axes(fig, ax, **kwargs)

    fig.tight_layout(rect=rect)
    return fig, ax


def plot_signals_with_reference_pep(
    datapoint: BasePepDatasetWithAnnotations,
    *,
    collapse: bool = False,
    normalize_time: bool = False,
    heartbeat_subset: Sequence[int] | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    kwargs.setdefault("legend_orientation", "vertical")
    kwargs.setdefault("legend_outside", False)
    kwargs.setdefault("legend_max_cols", 6)
    kwargs.setdefault("legend_loc", _get_legend_loc(kwargs))
    rect = _get_rect(kwargs)

    fig, axs = plot_signals_with_reference_labels(
        datapoint,
        collapse=collapse,
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        **kwargs,
    )

    ecg_data, icg_data = _get_data(datapoint, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset)

    reference_labels = _get_reference_labels(datapoint, heartbeat_subset=heartbeat_subset)

    reference_labels_ecg = (
        pd.concat(
            [reference_labels["q_peaks"], reference_labels["q_peak_artefacts"]], names=["sample_relative", "label"]
        )
        .sort_index()
        .reset_index()
    )
    reference_labels_icg = (
        pd.concat(
            [reference_labels["b_points"], reference_labels["b_point_artefacts"]], names=["sample_relative", "label"]
        )
        .sort_index()
        .reset_index()
    )
    reference_labels_combined = pd.concat({"ecg": reference_labels_ecg, "icg": reference_labels_icg}, axis=1)

    if collapse:
        _add_pep_from_reference(ecg_data, icg_data, reference_labels_combined, axs, **kwargs)
        _handle_legend_one_axis(fig, axs, **kwargs)
    else:
        _add_pep_from_reference(ecg_data, icg_data, reference_labels_combined, axs[0], **kwargs)
        _add_pep_from_reference(ecg_data, icg_data, reference_labels_combined, axs[1], **kwargs)
        _handle_legend_two_axes(fig, axs, **kwargs)

    fig.tight_layout(rect=rect)
    return fig, axs


def plot_signals_with_algorithm_results(
    datapoint: BasePepDatasetWithAnnotations,
    *,
    collapse: bool = False,
    algorithm: BaseExtraction,
    normalize_time: bool = False,
    heartbeat_subset: Sequence[int] | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes | Sequence[plt.Axes]]:
    kwargs.setdefault("legend_loc", _get_legend_loc(kwargs))
    kwargs.setdefault("legend_max_cols", 5)
    rect = _get_rect(kwargs)

    fig, axs = plot_signals_with_reference_labels(
        datapoint,
        collapse=collapse,
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        b_point_label="Reference B-Points",
        **kwargs,
    )

    heartbeat_subset = _sanitize_heartbeat_subset(heartbeat_subset)
    ecg_data, icg_data = _get_data(datapoint, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset)
    heartbeats = _get_heartbeats(datapoint, heartbeat_subset)

    if isinstance(algorithm, BaseEcgExtractionWithHeartbeats):
        algorithm.extract(
            ecg=ecg_data,
            heartbeats=heartbeats,
            sampling_rate_hz=datapoint.sampling_rate_ecg,
        )
        q_peaks = algorithm.points_["q_peak_sample"]
        q_peaks = q_peaks.loc[heartbeats.index].dropna()

        if collapse:
            _add_ecg_q_peaks(
                ecg_data,
                q_peaks,
                axs,
                q_peak_label="Detected Q-Peaks",
                q_peak_color=cmaps.med_dark[0],
                **kwargs,
            )
        else:
            _add_ecg_q_peaks(
                ecg_data,
                q_peaks,
                axs[0],
                q_peak_label="Detected Q-Peaks",
                q_peak_color=cmaps.med_dark[0],
                **kwargs,
            )
    if isinstance(algorithm, BaseBPointExtraction):
        c_point_algo = CPointExtractionScipyFindPeaks()
        c_points = c_point_algo.extract(
            icg=icg_data, heartbeats=heartbeats, sampling_rate_hz=datapoint.sampling_rate_icg
        ).points_
        algorithm.extract(
            icg=icg_data,
            heartbeats=heartbeats,
            c_points=c_points,
            sampling_rate_hz=datapoint.sampling_rate_icg,
        )
        b_points = algorithm.points_["b_point_sample"]
        b_points = b_points.loc[heartbeats.index].dropna()
        if collapse:
            _add_icg_b_points(
                icg_data,
                b_points,
                axs,
                b_point_label="Detected B-Points",
                b_point_color=cmaps.phil_dark[0],
                **kwargs,
            )
        else:
            _add_icg_b_points(
                icg_data,
                b_points,
                axs[1],
                b_point_label="Detected B-Points",
                b_point_color=cmaps.phil_dark[0],
                **kwargs,
            )

    if collapse:
        _handle_legend_one_axis(fig, axs, **kwargs)
    else:
        _handle_legend_two_axes(fig, axs, **kwargs)

    if not collapse:
        fig.align_ylabels()
    fig.tight_layout(rect=rect)
    return fig, axs


def plot_signals_from_challenge_results(
    datapoint: BasePepDatasetWithAnnotations,
    pep_results_per_sample: pd.DataFrame,
    *,
    collapse: bool = False,
    normalize_time: bool = False,
    heartbeat_subset: Sequence[int] | None = None,
    add_pep: bool = False,
    **kwargs: Any,
) -> tuple[plt.Figure, Sequence[plt.Axes]]:
    kwargs.setdefault("legend_loc", _get_legend_loc(kwargs))
    kwargs.setdefault("legend_max_cols", 5)
    rect = _get_rect(kwargs)

    fig, axs = plot_signals(
        datapoint, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset, collapse=collapse, **kwargs
    )

    ecg_data, icg_data = _get_data(datapoint, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset)

    labels_from_challenge = _get_labels_from_challenge_results(pep_results_per_sample, heartbeat_subset)

    print(ecg_data)
    print(labels_from_challenge["heartbeats_start"])

    heartbeats_start = ecg_data.index[labels_from_challenge["heartbeats_start"]]
    heartbeats_end = ecg_data.index[labels_from_challenge["heartbeats_end"] - 1]
    q_peak_labels_reference = labels_from_challenge["q_peak_labels_reference"]
    q_peak_labels_estimated = labels_from_challenge["q_peak_labels_estimated"]
    b_point_labels_reference = labels_from_challenge["b_point_labels_reference"]
    b_point_labels_estimated = labels_from_challenge["b_point_labels_estimated"]

    labels_reference = pd.concat({"ecg": q_peak_labels_reference, "icg": b_point_labels_reference}, axis=1)
    labels_estimated = pd.concat({"ecg": q_peak_labels_estimated, "icg": b_point_labels_estimated}, axis=1)

    if collapse:
        ax_ecg = axs
        ax_icg = axs
    else:
        ax_ecg = axs[0]
        ax_icg = axs[1]

    _add_heartbeat_borders(heartbeats_start, ax_ecg)
    _add_heartbeat_borders(heartbeats_end, ax_ecg)
    if not collapse:
        _add_heartbeat_borders(heartbeats_start, ax_icg)
        _add_heartbeat_borders(heartbeats_end, ax_icg)

    _add_ecg_q_peaks(
        ecg_data,
        q_peak_labels_reference,
        ax_ecg,
        q_peak_color=cmaps.med[0],
        q_peak_label="Q-Peak Reference",
        plot_artifacts=False,
    )
    _add_ecg_q_peaks(
        ecg_data,
        q_peak_labels_estimated,
        ax_ecg,
        q_peak_color=cmaps.med_dark[0],
        q_peak_label="Q-Peak Estimated",
        plot_artifacts=False,
    )

    _add_icg_b_points(
        icg_data,
        b_point_labels_reference,
        ax_icg,
        b_point_color=cmaps.phil[0],
        b_point_label="B-Point Reference",
        plot_artifacts=False,
    )
    _add_icg_b_points(
        icg_data,
        b_point_labels_estimated,
        ax_icg,
        b_point_color=cmaps.phil_dark[0],
        b_point_label="B-Point Estimated",
        plot_artifacts=False,
    )

    if add_pep:
        _add_pep_from_results(
            ecg_data,
            icg_data,
            labels_reference,
            ax=ax_icg,
            pep_color=cmaps.nat[0],
            pep_hatch="////",
            pep_label="PEP Reference",
        )
        _add_pep_from_results(
            ecg_data,
            icg_data,
            labels_estimated,
            ax=ax_icg,
            pep_color=cmaps.nat_dark[0],
            pep_hatch=r"\\\\",
            pep_label="PEP Estimated",
        )
        if not collapse:
            _add_pep_from_results(
                ecg_data,
                icg_data,
                labels_reference,
                ax=ax_ecg,
                pep_color=cmaps.nat[0],
                pep_hatch="////",
                pep_label="PEP Reference",
            )
            _add_pep_from_results(
                ecg_data,
                icg_data,
                labels_estimated,
                ax=ax_ecg,
                pep_color=cmaps.nat_dark[0],
                pep_hatch=r"\\\\",
                pep_label="PEP Estimated",
            )

    if collapse:
        _handle_legend_one_axis(fig, axs, **kwargs)
    else:
        _handle_legend_two_axes(fig, axs, **kwargs)

    fig.tight_layout(rect=rect)

    return fig, axs


def _plot_signals_one_axis(
    *,
    datapoint: BasePepDataset | None = None,
    df: pd.DataFrame | None = None,
    normalize_time: bool = False,
    heartbeat_subset: Sequence[int] | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    kwargs.setdefault("legend_loc", _get_legend_loc(kwargs))
    kwargs.setdefault("legend_max_cols", 5)
    plot_ecg = kwargs.get("plot_ecg", True)
    plot_icg = kwargs.get("plot_icg", True)
    color = kwargs.get("color", cmaps.fau[0])

    if datapoint is not None and df is not None:
        raise ValueError("Either `datapoint` or `df` must be provided, but not both.")
    if datapoint is None and df is None:
        raise ValueError("Either `datapoint` or `df` must be provided.")

    rect = _get_rect(kwargs)

    fig, ax = _get_fig_ax(kwargs)
    kwargs.pop("ax", None)

    x_label = "Time [hh:mm:ss]"

    if datapoint is not None:
        ecg_data, icg_data = _get_data(datapoint, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset)

        if plot_ecg:
            ecg_data.columns = ["ECG"]
            ecg_data.plot(ax=ax)
            ax.legend()
        if plot_icg:
            icg_data.columns = ["ICG ($dZ/dt$)"]
            icg_data.plot(ax=ax)

        if normalize_time or not all(isinstance(data.index, pd.DatetimeIndex) for data in [ecg_data, icg_data]):
            x_label = "Time [s]"
    else:
        if isinstance(df, pd.Series):
            df = df.to_frame()
        df.columns = kwargs.get("columns", df.columns)
        df.plot(ax=ax, color=color)
        if normalize_time or not isinstance(df.index, pd.DatetimeIndex):
            x_label = "Time [s]"

    ax.set_xlabel(x_label)
    ax.set_ylabel("Amplitude [a.u.]")

    _handle_legend_one_axis(fig, ax, **kwargs)

    if kwargs.get("use_tight", True):
        fig.tight_layout(rect=rect)

    return fig, ax


def _plot_signals_two_axes(
    *,
    datapoint: BasePepDataset,
    normalize_time: bool | None = False,
    heartbeat_subset: Sequence[int] | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, Sequence[plt.Axes]]:
    kwargs.setdefault("nrows", 2)
    kwargs.setdefault("legend_loc", _get_legend_loc(kwargs))
    kwargs.setdefault("legend_max_cols", 5)

    rect = kwargs.get("rect", _get_rect(kwargs))

    fig, axs = _get_fig_axs(kwargs)

    colors = iter(cmaps.faculties)

    ecg_data, icg_data = _get_data(datapoint, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset)
    ecg_data.columns = ["ECG"]
    icg_data.columns = ["ICG ($dZ/dt$)"]

    ecg_data.plot(ax=axs[0], color=next(colors), title="Electrocardiogram (ECG)")
    icg_data.plot(ax=axs[1], color=next(colors), title="Impedance Cardiogram (ICG)")

    _handle_legend_two_axes(fig, axs, **kwargs)

    for ax in axs:
        if normalize_time or not all(isinstance(data.index, pd.DatetimeIndex) for data in [ecg_data, icg_data]):
            ax.set_xlabel("Time [s]")
        else:
            ax.set_xlabel("Time [hh:mm:ss]")
        ax.set_ylabel("Amplitude [a.u.]")

    fig.align_ylabels()
    fig.tight_layout(rect=rect)

    return fig, axs


def _plot_blandaltman(  # noqa: PLR0915
    x: pd.Series | np.ndarray | list,
    y: pd.Series | np.ndarray | list,
    agreement: float = 1.96,
    xaxis: str = "mean",
    confidence: float = 0.95,
    annotate: bool = True,
    ax: plt.Axes | None = None,
    **kwargs: Any,
) -> plt.Axes:
    """
    Generate a Bland-Altman plot to compare two sets of measurements.

    Parameters
    ----------
    x, y : pd.Series, np.array, or list
        First and second measurements.
    agreement : float
        Multiple of the standard deviation to plot agreement limits.
        The defaults is 1.96, which corresponds to 95% confidence interval if
        the differences are normally distributed.
    xaxis : str
        Define which measurements should be used as the reference (x-axis).
        Default is to use the average of x and y ("mean"). Accepted values are
        "mean", "x" or "y".
    confidence : float
        If not None, plot the specified percentage confidence interval of
        the mean and limits of agreement. The CIs of the mean difference and
        agreement limits describe a possible error in the
        estimate due to a sampling error. The greater the sample size,
        the narrower the CIs will be.
    annotate : bool
        If True (default), annotate the values for the mean difference
        and agreement limits.
    ax : matplotlib axes
        Axis on which to draw the plot.
    **kwargs : optional
        Optional argument(s) passed to :py:func:`matplotlib.pyplot.scatter`.

    Returns
    -------
    ax : Matplotlib Axes instance
        Returns the Axes object with the plot for further tweaking.

    Notes
    -----
    Bland-Altman plots [1]_ are extensively used to evaluate the agreement
    among two different instruments or two measurements techniques.
    They allow identification of any systematic difference between the
    measurements (i.e., fixed bias) or possible outliers.

    The mean difference (= x - y) is the estimated bias, and the SD of the
    differences measures the random fluctuations around this mean.
    If the mean value of the difference differs significantly from 0 on the
    basis of a 1-sample t-test, this indicates the presence of fixed bias.
    If there is a consistent bias, it can be adjusted for by subtracting the
    mean difference from the new method.

    It is common to compute 95% limits of agreement for each comparison
    (average difference ± 1.96 standard deviation of the difference), which
    tells us how far apart measurements by 2 methods were more likely to be
    for most individuals. If the differences within mean ± 1.96 SD are not
    clinically important, the two methods may be used interchangeably.
    The 95% limits of agreement can be unreliable estimates of the population
    parameters especially for small sample sizes so, when comparing methods
    or assessing repeatability, it is important to calculate confidence
    intervals for the 95% limits of agreement.

    The code is an adaptation of the
    `PyCompare <https://github.com/jaketmp/pyCompare>`_ package. The present
    implementation is a simplified version; please refer to the original
    package for more advanced functionalities.

    References
    ----------
    .. [1] Bland, J. M., & Altman, D. (1986). Statistical methods for assessing
           agreement between two methods of clinical measurement. The lancet,
           327(8476), 307-310.

    .. [2] Giavarina, D. (2015). Understanding bland altman analysis.
           Biochemia medica, 25(2), 141-151.

    Examples
    --------
    Bland-Altman plot (example data from [2]_)
    """
    # Safety check
    assert xaxis in ["mean", "x", "y"]
    # Get names before converting to NumPy array
    xname = x.name if isinstance(x, pd.Series) else "x"
    yname = y.name if isinstance(y, pd.Series) else "y"
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.ndim == 1
    assert y.ndim == 1
    assert x.size == y.size
    assert not np.isnan(x).any(), "Missing values in x or y are not supported."
    assert not np.isnan(y).any(), "Missing values in x or y are not supported."

    _annotate_kwargs = {key: val for key, val in kwargs.items() if key.startswith("annotate")}
    # remove annotate kwargs from kwargs
    kwargs = {key: val for key, val in kwargs.items() if not key.startswith("annotate")}

    # Update default kwargs with specified inputs
    _scatter_kwargs = {"color": "tab:blue", "alpha": 0.8}
    _scatter_kwargs.update(kwargs)

    # Calculate mean, STD and SEM of x - y
    n = x.size
    dof = n - 1
    diff = x - y
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    mean_diff_se = np.sqrt(std_diff**2 / n)
    # Limits of agreements
    high = mean_diff + agreement * std_diff
    low = mean_diff - agreement * std_diff
    high_low_se = np.sqrt(3 * std_diff**2 / n)

    # Define x-axis
    if xaxis == "mean":
        xval = np.vstack((x, y)).mean(0)
        xlabel = f"Mean of {xname} and {yname}"
    elif xaxis == "x":
        xval = x
        xlabel = xname
    else:
        xval = y
        xlabel = yname

    # Start the plot
    if ax is None:
        ax = plt.gca()

    # Plot the mean diff, limits of agreement and scatter
    ax.scatter(xval, diff, **_scatter_kwargs)
    ax.axhline(mean_diff, color="k", linestyle="-", lw=2)
    ax.axhline(high, color="k", linestyle=":", lw=1.5)
    ax.axhline(low, color="k", linestyle=":", lw=1.5)

    # Annotate values
    if annotate:
        loa_range = high - low
        offset = (loa_range / 100.0) * 1.5
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        annotate_fontsize = _annotate_kwargs.get("annotate_fontsize", "medium")
        annotate_bbox = _annotate_kwargs.get("annotate_bbox", False)

        bbox = transforms.Bbox.null()
        xloc = 0.98
        t = ax.text(
            xloc,
            mean_diff + offset,
            "Mean",
            ha="right",
            va="bottom",
            transform=trans,
            # bbox=bbox,
            fontdict={"fontsize": annotate_fontsize},
        )
        bbox.update_from_data_xy(_get_bbox_coords(t, ax), ignore=False)

        t = ax.text(
            xloc,
            mean_diff - offset,
            f"{mean_diff:.2f}",
            ha="right",
            va="top",
            transform=trans,
            # bbox=bbox,
            fontdict={"fontsize": annotate_fontsize},
        )
        bbox.update_from_data_xy(_get_bbox_coords(t, ax), ignore=False)

        t = ax.text(
            xloc,
            high + offset,
            f"+{agreement:.2f} SD",
            ha="right",
            va="bottom",
            transform=trans,
            # bbox=bbox,
            fontdict={"fontsize": annotate_fontsize},
        )
        bbox.update_from_data_xy(_get_bbox_coords(t, ax), ignore=False)

        t = ax.text(
            xloc,
            high - offset,
            f"{high:.2f}",
            ha="right",
            va="top",
            transform=trans,
            # bbox=bbox,
            fontdict={"fontsize": annotate_fontsize},
        )
        bbox.update_from_data_xy(_get_bbox_coords(t, ax), ignore=False)

        t = ax.text(
            xloc,
            low + offset,
            f"{low:.2f}",
            ha="right",
            va="bottom",
            transform=trans,
            # bbox=bbox,
            fontdict={"fontsize": annotate_fontsize},
        )
        bbox.update_from_data_xy(_get_bbox_coords(t, ax), ignore=False)

        t = ax.text(
            xloc,
            low - offset,
            f"-{agreement:.2f} SD",
            ha="right",
            va="top",
            transform=trans,
            # bbox=bbox,
            fontsize=annotate_fontsize,
        )
        bbox.update_from_data_xy(_get_bbox_coords(t, ax), ignore=False)
        if annotate_bbox:
            add_fancy_patch_around(ax, bbox)

    # Add 95% confidence intervals for mean bias and limits of agreement
    if confidence is not None:
        assert 0 < confidence < 1
        ci = {
            "mean": stats.t.interval(confidence, dof, loc=mean_diff, scale=mean_diff_se),
            "high": stats.t.interval(confidence, dof, loc=high, scale=high_low_se),
            "low": stats.t.interval(confidence, dof, loc=low, scale=high_low_se),
        }
        ax.axhspan(ci["mean"][0], ci["mean"][1], facecolor="tab:grey", alpha=0.2)
        ax.axhspan(ci["high"][0], ci["high"][1], facecolor=_scatter_kwargs["color"], alpha=0.2)
        ax.axhspan(ci["low"][0], ci["low"][1], facecolor=_scatter_kwargs["color"], alpha=0.2)

    # Labels
    ax.set_ylabel(f"{xname} - {yname}")
    ax.set_xlabel(xlabel)
    sns.despine(ax=ax)
    return ax


def _plot_paired(  # noqa: PLR0915, PLR0912, C901
    data: pd.DataFrame,
    dv: str,
    within: str,
    subject: str,
    order: Sequence[str] | None = None,
    boxplot: bool = True,
    boxplot_in_front: bool = False,
    orient: str = "v",
    ax: plt.Axes | None = None,
    colors: Sequence[str] | None = None,
    pointplot_kwargs: dict | None = None,
    boxplot_kwargs: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Paired plot.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        Long-format dataFrame.
    dv : string
        Name of column containing the dependent variable.
    within : string
        Name of column containing the within-subject factor.
    subject : string
        Name of column containing the subject identifier.
    order : list of str
        List of values in ``within`` that define the order of elements on the
        x-axis of the plot. If None, uses alphabetical order.
    boxplot : boolean
        If True, add a boxplot to the paired lines using the
        :py:func:`seaborn.boxplot` function.
    boxplot_in_front : boolean
        If True, the boxplot is plotted on the foreground (i.e. above the
        individual lines) and with a slight transparency. This makes the
        overall plot more readable when plotting a large numbers of subjects.

        .. versionadded:: 0.3.8
    orient : string
        Plot the boxplots vertically and the subjects on the x-axis if
        ``orient='v'`` (default). Set to ``orient='h'`` to rotate the plot by
        by 90 degrees.

        .. versionadded:: 0.3.9
    ax : matplotlib axes
        Axis on which to draw the plot.
    colors : list of str
        Line colors names. Default is green when value increases from A to B,
        indianred when value decreases from A to B and grey when the value is
        the same in both measurements.
    pointplot_kwargs : dict
        Dictionnary of optional arguments that are passed to the
        :py:func:`seaborn.pointplot` function.
    boxplot_kwargs : dict
        Dictionnary of optional arguments that are passed to the
        :py:func:`seaborn.boxplot` function.

    Returns
    -------
    ax : Matplotlib Axes instance
        Returns the Axes object with the plot for further tweaking.

    Notes
    -----
    Data must be a long-format pandas DataFrame. Missing values are automatically removed using a
    strict listwise approach (= complete-case analysis).

    Examples
    --------
    Default paired plot:

    .. plot::

        >>> import pingouin as pg
        >>> df = pg.read_dataset("mixed_anova").query("Time != 'January'")
        >>> df = df.query("Group == 'Meditation' and Subject > 40")
        >>> fig, ax = pg._plot_paired(data=df, dv="Scores", within="Time", subject="Subject")

    Paired plot on an existing axis (no boxplot and uniform color):

    .. plot::

        >>> import pingouin as pg
        >>> import matplotlib.pyplot as plt
        >>> df = pg.read_dataset("mixed_anova").query("Time != 'January'")
        >>> df = df.query("Group == 'Meditation' and Subject > 40")
        >>> fig, ax1 = plt.subplots(1, 1, figsize=(5, 4))
        >>> pg._plot_paired(
        ...     data=df,
        ...     dv="Scores",
        ...     within="Time",
        ...     subject="Subject",
        ...     ax=ax1,
        ...     boxplot=False,
        ...     colors=["grey", "grey", "grey"],
        ... )  # doctest: +SKIP

    Horizontal paired plot with three unique within-levels:

    .. plot::

        >>> import pingouin as pg
        >>> import matplotlib.pyplot as plt
        >>> df = pg.read_dataset("mixed_anova").query("Group == 'Meditation'")
        >>> # df = df.query("Group == 'Meditation' and Subject > 40")
        >>> fig, ax = pg._plot_paired(
        >>>     data=df, dv="Scores", within="Time", subject="Subject", orient="h"
        >>> )  # doctest: +SKIP

    With the boxplot on the foreground:

    .. plot::

        >>> import pingouin as pg
        >>> df = pg.read_dataset("mixed_anova").query("Time != 'January'")
        >>> df = df.query("Group == 'Control'")
        >>> fig, ax = pg._plot_paired(data=df, dv="Scores", within="Time", subject="Subject", boxplot_in_front=True)
    """
    from pingouin.utils import _check_dataframe

    # Set default colors
    if colors is None:
        colors = ["green", "grey", "indianred"]

    if pointplot_kwargs is None:
        pointplot_kwargs = {}
    if boxplot_kwargs is None:
        boxplot_kwargs = {}

    # Update default kwargs with specified inputs
    _pointplot_kwargs = {"scale": 0.6, "marker": "."}
    _pointplot_kwargs.update(pointplot_kwargs)
    _boxplot_kwargs = {"color": "lightslategrey", "width": 0.2}
    _boxplot_kwargs.update(boxplot_kwargs)
    # Extract pointplot alpha, if set
    pp_alpha = _pointplot_kwargs.pop("alpha", 1.0)

    # Calculate size of the plot elements by scale as in Seaborn pointplot
    scale = _pointplot_kwargs.pop("scale")
    lw = plt.rcParams["lines.linewidth"] * 1.8 * scale  # get the linewidth
    mew = lw * 0.75  # get the markeredgewidth
    markersize = np.pi * np.square(lw) * 2  # get the markersize

    # Set boxplot in front of Line2D plot (zorder=2 for both) and add alpha
    if boxplot_in_front:
        _boxplot_kwargs.update(
            {
                "boxprops": {"zorder": 2},
                "whiskerprops": {"zorder": 2},
                "zorder": 2,
            }
        )

    # Validate args
    data = _check_dataframe(data=data, dv=dv, within=within, subject=subject, effects="within")

    # Pivot and melt the table. This has several effects:
    # 1) Force missing values to be explicit (a NaN cell is created)
    # 2) Automatic collapsing to the mean if multiple within factors are present
    # 3) If using dropna, remove rows with missing values (listwise deletion).
    # The latter is the same behavior as JASP (= strict complete-case analysis).
    data_piv = data.pivot_table(index=subject, columns=within, values=dv, observed=True)
    data_piv = data_piv.dropna()
    data = data_piv.melt(ignore_index=False, value_name=dv).reset_index()

    # Extract within-subject level (alphabetical order)
    x_cat = np.unique(data[within])

    if order is None:
        order = x_cat
    else:
        assert len(order) == len(x_cat), (
            "Order must have the same number of elements as the number of levels in `within`."
        )

    # Substitute within by integer order of the ordered columns to allow for
    # changing the order of numeric withins.
    data["wthn"] = data[within].replace({_ordr: str(i) for i, _ordr in enumerate(order)})
    data["wthn"] = data["wthn"].astype(int)
    order_num = range(len(order))  # Make numeric order

    # Start the plot
    if ax is None:
        fig, ax = _get_fig_ax({})
    else:
        fig = ax.get_figure()

    # Set x and y depending on orientation using the num. replacement within
    _x = "wthn" if orient == "v" else dv
    _y = dv if orient == "v" else "wthn"

    for cat in range(len(x_cat) - 1):
        _order = (order_num[cat], order_num[cat + 1])
        # Extract data of the current subject-combination
        data_now = data.loc[data["wthn"].isin(_order), [dv, "wthn", subject]]
        # Select colors for all lines between the current subjects
        y1 = data_now.loc[data_now["wthn"] == _order[0], dv].to_numpy()
        y2 = data_now.loc[data_now["wthn"] == _order[1], dv].to_numpy()
        # Line and scatter colors depending on subject dv trend
        _colors = np.where(y1 < y2, colors[0], np.where(y1 > y2, colors[2], colors[1]))
        # Line and scatter colors as hue-indexed dictionary
        _colors = dict(zip(data_now[subject].unique(), _colors, strict=False))
        # Plot individual lines using Seaborn
        sns.lineplot(
            data=data_now,
            x=_x,
            y=_y,
            hue=subject,
            palette=_colors,
            ls="-",
            lw=lw,
            legend=False,
            ax=ax,
        )
        # Plot individual markers using Seaborn
        sns.scatterplot(
            data=data_now,
            x=_x,
            y=_y,
            hue=subject,
            palette=_colors,
            edgecolor="face",
            lw=mew,
            sizes=[markersize] * data_now.shape[0],
            legend=False,
            ax=ax,
            **_pointplot_kwargs,
        )

    # Set zorder and alpha of pointplot markers and lines
    _ = plt.setp(ax.collections, alpha=pp_alpha, zorder=2)  # Set marker alpha
    _ = plt.setp(ax.lines, alpha=pp_alpha, zorder=2)  # Set line alpha

    if boxplot:
        # Set boxplot x and y depending on orientation
        _xbp = within if orient == "v" else dv
        _ybp = dv if orient == "v" else within
        sns.boxplot(data=data, x=_xbp, y=_ybp, order=order, ax=ax, orient=orient, **_boxplot_kwargs)

        # Set alpha to patch of boxplot but not to whiskers
        for patch in ax.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, 0.75))
    else:
        # If no boxplot, axis needs manual styling as in Seaborn pointplot
        if orient == "v":
            xlabel, ylabel = within, dv
            ax.set_xticks(np.arange(len(x_cat)))
            ax.set_xticklabels(order)
            ax.xaxis.grid(False)
            ax.set_xlim(-0.5, len(x_cat) - 0.5, auto=None)
        else:
            xlabel, ylabel = dv, within
            ax.set_yticks(np.arange(len(x_cat)))
            ax.set_yticklabels(order)
            ax.yaxis.grid(False)
            ax.set_ylim(-0.5, len(x_cat) - 0.5, auto=None)
            ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # Despine and trim
    sns.despine(trim=True, ax=ax)
    return fig, ax
