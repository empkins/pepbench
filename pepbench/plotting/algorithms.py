"""Module for visualizing Q-wave onset detection and B-point detection algorithms."""

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
import pandas as pd
from biopsykit.signals.ecg.event_extraction import QPeakExtractionNeurokitDwt, QWaveOnsetExtractionVanLien2013
from biopsykit.signals.icg.event_extraction import (
    BPointExtractionArbol2017,
    BPointExtractionDebski1993,
    BPointExtractionDrost2022,
    BPointExtractionForouzanfar2018,
    CPointExtractionScipyFindPeaks,
    BPointExtractionSherwood1990,
)
from fau_colors import cmaps
from matplotlib import pyplot as plt

from pepbench.datasets import BaseUnifiedPepExtractionDataset
from pepbench.plotting._base_plotting import _plot_signals_one_axis
from pepbench.plotting._utils import (
    _add_ecg_q_wave_onsets,
    _add_ecg_r_peaks,
    _add_heartbeat_borders,
    _add_icg_b_points,
    _add_icg_c_points,
    _get_annotation_bbox_no_edge,
    _get_data,
    _get_legend_loc,
    _get_rect,
    _get_reference_labels,
    _handle_legend_one_axis,
    _handle_legend_two_axes,
    _sanitize_heartbeat_subset,
)

__all__ = [
    "plot_q_peak_extraction_neurokit_dwt",
    "plot_q_wave_onset_extraction_van_lien_2013",
    "plot_b_point_extraction_sherwood1990",
    "plot_b_point_extraction_debski1993",
    "plot_b_point_extraction_drost2022",
    "plot_b_point_extraction_arbol2017",
    "plot_b_point_extraction_forouzanfar2018",
]


def plot_q_peak_extraction_neurokit_dwt(
    datapoint: BaseUnifiedPepExtractionDataset,
    *,
    heartbeat_subset: Optional[Sequence[int]] = None,
    use_clean: Optional[bool] = True,
    normalize_time: Optional[bool] = False,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    kwargs.setdefault("legend_outside", True)
    kwargs.setdefault("legend_orientation", "horizontal")
    kwargs.setdefault("legend_loc", _get_legend_loc(**kwargs))

    rect = _get_rect(**kwargs)

    heartbeat_subset = _sanitize_heartbeat_subset(heartbeat_subset)
    ecg_data, _ = _get_data(
        datapoint, use_clean=use_clean, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset
    )

    if len(ecg_data) < 4 * datapoint.sampling_rate_ecg:
        raise ValueError("ECG data is too short for Q-peak detection. Please provide more heartbeats.")

    heartbeats = _get_heartbeats(datapoint, heartbeat_subset)
    heartbeat_borders = _get_heartbeat_borders(ecg_data, heartbeats)

    q_wave_onset_algo = QPeakExtractionNeurokitDwt()
    q_wave_onset_algo.extract(ecg_data, heartbeats=heartbeats, sampling_rate_hz=datapoint.sampling_rate_ecg)

    q_wave_onset_samples = q_wave_onset_algo.points_["q_wave_onset_sample"].dropna()
    q_wave_onset_samples_reference = _get_reference_labels(datapoint, heartbeat_subset)["q_wave_onsets"]

    fig, ax = _plot_signals_one_axis(
        datapoint=datapoint,
        use_clean=use_clean,
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        plot_icg=False,
        **kwargs,
    )

    _add_ecg_q_wave_onsets(
        ecg_data,
        q_wave_onset_samples_reference,
        ax=ax,
        q_wave_label="Reference Q-Wave Onsets",
        q_wave_color=cmaps.med_dark[0],
        **kwargs,
    )
    _add_ecg_q_wave_onsets(
        ecg_data,
        q_wave_onset_samples,
        ax=ax,
        q_wave_label="Detected Q-Wave Onsets",
        q_wave_color=cmaps.med[0],
        **kwargs,
    )

    _add_heartbeat_borders(heartbeat_borders, ax=ax, **kwargs)

    _handle_legend_one_axis(fig=fig, ax=ax, **kwargs)
    fig.tight_layout(rect=rect)

    return fig, ax


def plot_q_wave_onset_extraction_van_lien_2013(
    datapoint: BaseUnifiedPepExtractionDataset,
    *,
    heartbeat_subset: Optional[Sequence[int]] = None,
    use_clean: Optional[bool] = True,
    normalize_time: Optional[bool] = False,
    algo_params: Optional[dict] = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot example of Q-wave onset extraction using the Van Lien (2013) algorithm."""
    kwargs.setdefault("legend_outside", True)
    kwargs.setdefault("legend_orientation", "horizontal")
    kwargs.setdefault("legend_loc", _get_legend_loc(**kwargs))
    rect = _get_rect(**kwargs)

    if algo_params is None:
        algo_params = {}

    heartbeat_subset = _sanitize_heartbeat_subset(heartbeat_subset)
    ecg_data, _ = _get_data(
        datapoint, use_clean=use_clean, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset
    )

    heartbeats = _get_heartbeats(datapoint, heartbeat_subset)
    heartbeat_borders = _get_heartbeat_borders(ecg_data, heartbeats)

    q_wave_onset_algo = QWaveOnsetExtractionVanLien2013(**algo_params)
    q_wave_onset_algo.extract(ecg=ecg_data, heartbeats=heartbeats, sampling_rate_hz=datapoint.sampling_rate_ecg)

    r_peak_samples = heartbeats["r_peak_sample"].astype(int)
    q_wave_onset_samples = q_wave_onset_algo.points_["q_wave_onset_sample"].astype(int)
    q_wave_onset_samples_reference = _get_reference_labels(datapoint, heartbeat_subset)["q_wave_onsets"]

    time_interval_ms = q_wave_onset_algo.get_params()["time_interval_ms"]

    kwargs.setdefault("r_peak_linewidth", 2)
    kwargs.setdefault("r_peak_linestyle", "--")
    kwargs.setdefault("r_peak_marker", "X")
    kwargs.setdefault("q_wave_linewidth", 2)

    fig, ax = _plot_signals_one_axis(
        datapoint=datapoint,
        use_clean=use_clean,
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        plot_icg=False,
        **kwargs,
    )
    _add_heartbeat_borders(heartbeat_borders, ax=ax, **kwargs)
    _add_ecg_r_peaks(ecg_data, r_peak_samples, ax=ax, **kwargs)
    _add_ecg_q_wave_onsets(
        ecg_data,
        q_wave_onset_samples_reference,
        q_wave_label="Reference Q-Wave Onsets",
        q_wave_color=cmaps.med_dark[0],
        ax=ax,
        **kwargs,
    )
    _add_ecg_q_wave_onsets(
        ecg_data,
        q_wave_onset_samples,
        q_wave_label="Detected Q-Wave Onsets",
        ax=ax,
        **kwargs,
    )

    # draw arrow from R-peak to Q-wave onset
    for r_peak, q_wave_onset in zip(r_peak_samples, q_wave_onset_samples):
        x_q_wave_onset = ecg_data.index[q_wave_onset]
        x_r_peak = ecg_data.index[r_peak]
        y = ecg_data.iloc[r_peak]
        middle_x = x_q_wave_onset + (x_r_peak - x_q_wave_onset) / 2
        # align text to the center of the array
        ax.annotate(
            "",
            xy=(x_q_wave_onset, y),
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


def plot_b_point_extraction_sherwood1990(
    datapoint: BaseUnifiedPepExtractionDataset,
    *,
    heartbeat_subset: Optional[Sequence[int]] = None,
    use_clean: Optional[bool] = True,
    normalize_time: Optional[bool] = False,
    algo_params: Optional[dict] = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(**kwargs)
    kwargs.setdefault("legend_outside", True)
    kwargs.setdefault("legend_orientation", "horizontal")
    kwargs.setdefault("legend_loc", _get_legend_loc(**kwargs))
    kwargs.setdefault("legend_max_cols", 4)
    rect = _get_rect(**kwargs)

    if algo_params is None:
        algo_params = {}

    heartbeat_subset = _sanitize_heartbeat_subset(heartbeat_subset)
    ecg_data, icg_data = _get_data(
        datapoint, use_clean=use_clean, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset
    )
    heartbeats = _get_heartbeats(datapoint, heartbeat_subset)
    heartbeat_borders = _get_heartbeat_borders(icg_data, heartbeats)

    icg_data = icg_data.squeeze()

    algo_params_c_point = {
        key: val for key, val in algo_params.items() if key in ["window_c_correction", "save_candidates"]
    }
    algo_params_b_point = {key: val for key, val in algo_params.items() if key not in algo_params_c_point}
    c_point_algo = CPointExtractionScipyFindPeaks(**algo_params_c_point)
    c_point_algo.extract(icg=icg_data, heartbeats=heartbeats, sampling_rate_hz=datapoint.sampling_rate_icg)

    b_point_algo = BPointExtractionSherwood1990(**algo_params_b_point)
    b_point_algo.extract(
        icg=icg_data, heartbeats=heartbeats, c_points=c_point_algo.points_, sampling_rate_hz=datapoint.sampling_rate_icg
    )

    b_point_samples_reference = _get_reference_labels(datapoint, heartbeat_subset)["b_points"]
    b_point_samples = b_point_algo.points_["b_point_sample"].dropna().astype(int)

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
        datapoint=datapoint,
        use_clean=use_clean,
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        plot_ecg=False,
        ax=ax,
        **kwargs,
    )

    _add_heartbeat_borders(heartbeats=heartbeat_borders, ax=ax, **kwargs)

    _add_icg_c_points(icg_data, c_point_samples, ax=ax, **kwargs)

    _add_icg_b_points(
        icg_data,
        b_point_samples_reference,
        ax=ax,
        b_point_label="Reference B Points",
        b_point_color=cmaps.phil_dark[0],
        **kwargs,
    )

    _add_icg_b_points(
        icg_data,
        zero_crossings_filtered,
        ax=ax,
        b_point_label="Zero Crossings before C-Point",
        b_point_color=cmaps.phil[1],
        b_point_plot_marker=False,
        **kwargs,
    )

    _add_icg_b_points(
        icg_data,
        b_point_samples,
        ax=ax,
        b_point_label="Detected B Points",
        **kwargs,
    )

    ax.axhline(0, color=cmaps.tech[1], linestyle="dashed", linewidth=1, label="Zero Line")

    _handle_legend_one_axis(fig=fig, ax=ax, **kwargs)

    fig.tight_layout(rect=rect)
    return fig, ax


def plot_b_point_extraction_debski1993(
    datapoint: BaseUnifiedPepExtractionDataset,
    *,
    heartbeat_subset: Optional[Sequence[int]] = None,
    use_clean: Optional[bool] = True,
    normalize_time: Optional[bool] = False,
    algo_params: Optional[dict] = None,
    **kwargs: Any,
) -> tuple[plt.Figure, Sequence[plt.Axes]]:
    fig, axs = plt.subplots(nrows=2, sharex=True, **kwargs)
    kwargs.setdefault("legend_outside", True)
    kwargs.setdefault("legend_orientation", "horizontal")
    kwargs.setdefault("legend_loc", _get_legend_loc(**kwargs))
    rect = _get_rect(**kwargs)

    if algo_params is None:
        algo_params = {}

    heartbeat_subset = _sanitize_heartbeat_subset(heartbeat_subset)
    ecg_data, icg_data = _get_data(
        datapoint, use_clean=use_clean, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset
    )
    heartbeats = _get_heartbeats(datapoint, heartbeat_subset)
    heartbeat_borders = _get_heartbeat_borders(icg_data, heartbeats)

    icg_data = icg_data.squeeze()
    # compute ICG derivation
    icg_2nd_der = np.gradient(icg_data)
    icg_2nd_der = pd.DataFrame(icg_2nd_der, index=icg_data.index, columns=["ICG Deriv. $(d^2Z/dt^2)$"])

    algo_params_c_point = {
        key: val for key, val in algo_params.items() if key in ["window_c_correction", "save_candidates"]
    }
    algo_params_b_point = {key: val for key, val in algo_params.items() if key not in algo_params_c_point}
    c_point_algo = CPointExtractionScipyFindPeaks(**algo_params_c_point)
    c_point_algo.extract(icg=icg_data, heartbeats=heartbeats, sampling_rate_hz=datapoint.sampling_rate_icg)

    b_point_algo = BPointExtractionDebski1993(**algo_params_b_point)
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
        use_clean=use_clean,
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        plot_ecg=True,
        ax=axs[0],
        **kwargs,
    )
    _plot_signals_one_axis(
        df=icg_2nd_der,
        use_clean=use_clean,
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
        b_point_label="Reference B Points",
        b_point_color=cmaps.phil_dark[0],
        **kwargs,
    )
    _add_icg_b_points(
        icg_data,
        b_point_samples,
        ax=axs[0],
        b_point_label="Detected B Points",
        **kwargs,
    )

    for _idx, row in search_window.iterrows():
        start = icg_2nd_der.index[row["r_peak_sample"]]
        end = icg_2nd_der.index[row["c_point_sample"]]
        axs[0].axvspan(start, end, color=cmaps.fau_light[1], alpha=0.3, zorder=0, label="B Point Search Windows")
        axs[1].axvspan(start, end, color=cmaps.fau_light[1], alpha=0.3, zorder=0, label="B Point Search Windows")

    _handle_legend_two_axes(
        fig=fig,
        axs=axs,
        **kwargs,
    )

    fig.align_ylabels()
    fig.tight_layout(rect=rect)

    return fig, axs


def plot_b_point_extraction_arbol2017(
    datapoint: BaseUnifiedPepExtractionDataset,
    *,
    heartbeat_subset: Optional[Sequence[int]] = None,
    use_clean: Optional[bool] = True,
    normalize_time: Optional[bool] = False,
    algo_params: Optional[dict] = None,
    **kwargs: Any,
) -> tuple[plt.Figure, Sequence[plt.Axes]]:
    fig, axs = plt.subplots(nrows=2, sharex=True, **kwargs)
    kwargs.setdefault("legend_outside", True)
    kwargs.setdefault("legend_orientation", "horizontal")
    kwargs.setdefault("legend_loc", _get_legend_loc(**kwargs))
    rect = _get_rect(**kwargs)

    if algo_params is None:
        algo_params = {}

    heartbeat_subset = _sanitize_heartbeat_subset(heartbeat_subset)
    ecg_data, icg_data = _get_data(
        datapoint, use_clean=use_clean, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset
    )
    heartbeats = _get_heartbeats(datapoint, heartbeat_subset)
    heartbeat_borders = _get_heartbeat_borders(icg_data, heartbeats)

    icg_data = icg_data.squeeze()
    icg_3rd_der = np.gradient(np.gradient(icg_data))
    icg_3rd_der = pd.DataFrame(icg_3rd_der, index=icg_data.index, columns=["ICG 3rd Deriv. $(d^3Z/dt^3)$"])

    algo_params_c_point = {
        key: val for key, val in algo_params.items() if key in ["window_c_correction", "save_candidates"]
    }
    algo_params_b_point = {key: val for key, val in algo_params.items() if key not in algo_params_c_point}
    c_point_algo = CPointExtractionScipyFindPeaks(**algo_params_c_point)
    c_point_algo.extract(icg=icg_data, heartbeats=heartbeats, sampling_rate_hz=datapoint.sampling_rate_icg)

    b_point_algo = BPointExtractionArbol2017(**algo_params_b_point)
    b_point_algo.extract(
        icg=icg_data, heartbeats=heartbeats, c_points=c_point_algo.points_, sampling_rate_hz=datapoint.sampling_rate_icg
    )

    c_point_samples = c_point_algo.points_["c_point_sample"].dropna().astype(int)
    c_point_minus_150_samples = c_point_samples - int(150 / 1000 * datapoint.sampling_rate_icg)
    c_point_minus_150_samples.name = "c_point_sample_minus_150"
    search_window = pd.concat([c_point_minus_150_samples, c_point_samples], axis=1)

    b_point_samples = b_point_algo.points_["b_point_sample"].dropna().astype(int)
    b_point_samples_reference = _get_reference_labels(datapoint, heartbeat_subset)["b_points"]

    _plot_signals_one_axis(
        df=icg_data,
        ax=axs[0],
        use_clean=use_clean,
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        color=cmaps.tech[0],
        **kwargs,
    )
    _plot_signals_one_axis(
        df=icg_3rd_der,
        ax=axs[1],
        use_clean=use_clean,
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
        b_point_label="Reference B Points",
        b_point_color=cmaps.phil_dark[0],
        **kwargs,
    )
    _add_icg_b_points(
        icg_data,
        b_point_samples,
        ax=axs[0],
        b_point_label="Detected B Points",
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
        c_point_minus_150_samples,
        ax=axs[0],
        c_point_color=cmaps.wiso_light[1],
        c_point_label="C Points - 150 ms",
        **kwargs,
    )
    _add_icg_c_points(icg_data, c_point_samples, ax=axs[1], c_point_plot_marker=False, **kwargs)
    _add_icg_c_points(
        icg_data,
        c_point_minus_150_samples,
        ax=axs[1],
        c_point_color=cmaps.wiso_light[1],
        c_point_plot_marker=False,
        **kwargs,
    )
    for _idx, row in search_window.iterrows():
        start = icg_data.index[row["c_point_sample_minus_150"]]
        end = icg_data.index[row["c_point_sample"]]
        axs[0].axvspan(start, end, color=cmaps.tech_light[0], alpha=0.3, zorder=0, label="B Point Search Windows")
        axs[1].axvspan(start, end, color=cmaps.tech_light[0], alpha=0.3, zorder=0, label="B Point Search Windows")

    _handle_legend_two_axes(
        fig=fig,
        axs=axs,
        **kwargs,
    )

    fig.tight_layout(rect=rect)
    fig.align_ylabels()

    return fig, axs


def plot_b_point_extraction_drost2022(
    datapoint: BaseUnifiedPepExtractionDataset,
    *,
    heartbeat_subset: Optional[Sequence[int]] = None,
    use_clean: Optional[bool] = True,
    normalize_time: Optional[bool] = False,
    algo_params: Optional[dict] = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    kwargs.setdefault("legend_max_cols", 4)
    kwargs.setdefault("legend_outside", True)
    kwargs.setdefault("legend_orientation", "horizontal")
    kwargs.setdefault("legend_loc", _get_legend_loc(**kwargs))
    rect = _get_rect(**kwargs)

    if algo_params is None:
        algo_params = {}

    heartbeat_subset = _sanitize_heartbeat_subset(heartbeat_subset)
    ecg_data, icg_data = _get_data(
        datapoint, use_clean=use_clean, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset
    )
    heartbeats = _get_heartbeats(datapoint, heartbeat_subset)
    heartbeat_borders = _get_heartbeat_borders(icg_data, heartbeats)

    icg_data = icg_data.squeeze()
    algo_params_c_point = {
        key: val for key, val in algo_params.items() if key in ["window_c_correction", "save_candidates"]
    }
    algo_params_b_point = {key: val for key, val in algo_params.items() if key not in algo_params_c_point}
    c_point_algo = CPointExtractionScipyFindPeaks(**algo_params_c_point)
    c_point_algo.extract(icg=icg_data, heartbeats=heartbeats, sampling_rate_hz=datapoint.sampling_rate_icg)

    b_point_algo = BPointExtractionDrost2022(**algo_params_b_point)
    b_point_algo.extract(
        icg=icg_data, heartbeats=heartbeats, c_points=c_point_algo.points_, sampling_rate_hz=datapoint.sampling_rate_icg
    )

    c_point_samples = c_point_algo.points_["c_point_sample"].dropna().astype(int)
    c_point_minus_150_samples = c_point_samples - int(150 / 1000 * datapoint.sampling_rate_icg)
    c_point_minus_150_samples.name = "c_point_sample_minus_150"
    search_window = pd.concat([c_point_minus_150_samples, c_point_samples], axis=1)

    b_point_samples_reference = _get_reference_labels(datapoint, heartbeat_subset)["b_points"]

    fig, ax = _plot_signals_one_axis(
        df=icg_data,
        use_clean=use_clean,
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
        c_point_label="C Points - 150 ms",
        **kwargs,
    )
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
        line_vals.index /= datapoint.sampling_rate_icg
        line_vals.index += start

        icg_slice = icg_data.iloc[start_sample:end_sample]
        distance = line_vals.squeeze().to_numpy() - icg_slice.squeeze().to_numpy()
        b_point_sample = start_sample + np.argmax(distance)

        ax.plot(
            line_vals.index,
            line_vals["result"],
            color=cmaps.wiso_dark[1],
            linestyle="--",
            linewidth=2,
            label="Straight Line Connection",
        )

        _add_icg_b_points(
            icg_data,
            b_point_samples_reference,
            ax=ax,
            b_point_label="Reference B Points",
            b_point_color=cmaps.phil_dark[0],
            **kwargs,
        )
        _add_icg_b_points(
            icg_data,
            b_point_sample,
            ax=ax,
            b_point_label="Detected B Points",
            **kwargs,
        )

        # ax.plot(
        #     [icg_data.index[b_point_sample], icg_data.index[b_point_sample]],
        #     [icg_data.iloc[b_point_sample], line_vals.iloc[np.argmax(distance)]],
        #     zorder=10,
        #     label=r"$d_{max}$",
        #     color=cmaps.fau[0],
        # )

        ax.annotate(
            "",
            xy=(icg_data.index[b_point_sample], icg_data.iloc[b_point_sample]),
            xytext=(icg_data.index[b_point_sample], line_vals.iloc[np.argmax(distance)]),
            textcoords="data",
            arrowprops={"arrowstyle": "-", "color": cmaps.fau[0], "lw": 2},
            zorder=10,
        )

        ax.annotate(
            r"$d_{max}$",
            xy=(icg_data.index[b_point_sample], line_vals.iloc[np.argmax(distance)]),
            xytext=(-10, -5),
            textcoords="offset points",
            bbox=_get_annotation_bbox_no_edge(),
            ha="right",
            zorder=10,
        )

        ax.axvspan(start, end, color=cmaps.tech_light[0], alpha=0.3, zorder=0, label="B Point Search Windows")

    _handle_legend_one_axis(fig=fig, ax=ax, **kwargs)

    fig.tight_layout(rect=rect)
    return fig, ax


def plot_b_point_extraction_forouzanfar2018(  # noqa: PLR0915
    datapoint: BaseUnifiedPepExtractionDataset,
    *,
    heartbeat_subset: Optional[Sequence[int]] = None,
    use_clean: Optional[bool] = True,
    normalize_time: Optional[bool] = False,
    algo_params: Optional[dict] = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    fig, axs = plt.subplots(nrows=2, sharex=True, **kwargs)
    kwargs.setdefault("legend_outside", True)
    kwargs.setdefault("legend_orientation", "horizontal")
    kwargs.setdefault("legend_loc", _get_legend_loc(**kwargs))
    kwargs.setdefault("legend_max_cols", 4)
    kwargs.setdefault("rect", (0, 0, 1, 0.8))
    rect = _get_rect(**kwargs)

    if algo_params is None:
        algo_params = {}

    heartbeat_subset = _sanitize_heartbeat_subset(heartbeat_subset)
    ecg_data, icg_data = _get_data(
        datapoint, use_clean=use_clean, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset
    )
    heartbeats = _get_heartbeats(datapoint, heartbeat_subset)
    heartbeat_borders = _get_heartbeat_borders(icg_data, heartbeats)

    icg_data = icg_data.squeeze()
    icg_2nd_der = np.gradient(icg_data)
    icg_3rd_der = np.gradient(icg_2nd_der)
    icg_2nd_der = pd.DataFrame(icg_2nd_der, index=icg_data.index, columns=["ICG 2nd Deriv. $(d^2Z/dt^2)$"])
    icg_3rd_der = pd.DataFrame(icg_3rd_der, index=icg_data.index, columns=["ICG 3rd Deriv. $(d^3Z/dt^3)$"])

    algo_params_c_point = {
        key: val for key, val in algo_params.items() if key in ["window_c_correction", "save_candidates"]
    }
    algo_params_b_point = {key: val for key, val in algo_params.items() if key not in algo_params_c_point}
    c_point_algo = CPointExtractionScipyFindPeaks(**algo_params_c_point)
    c_point_algo.extract(icg=icg_data, heartbeats=heartbeats, sampling_rate_hz=datapoint.sampling_rate_icg)

    b_point_algo = BPointExtractionForouzanfar2018(**algo_params_b_point)
    b_point_algo.extract(
        icg=icg_data, heartbeats=heartbeats, c_points=c_point_algo.points_, sampling_rate_hz=datapoint.sampling_rate_icg
    )

    c_point_samples = c_point_algo.points_["c_point_sample"].dropna().astype(int)
    b_point_samples_reference = _get_reference_labels(datapoint, heartbeat_subset)["b_points"]

    _plot_signals_one_axis(
        df=icg_data,
        ax=axs[0],
        use_clean=use_clean,
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        color=cmaps.tech[0],
        **kwargs,
    )

    # normalize 3rd der to have the same scale as the 2nd der
    icg_3rd_der = icg_3rd_der / float(icg_3rd_der.abs().max()) * float(icg_2nd_der.abs().max())

    _plot_signals_one_axis(
        df=icg_3rd_der,
        ax=axs[1],
        use_clean=use_clean,
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        color=cmaps.tech_dark[1],
        **kwargs,
    )
    _plot_signals_one_axis(
        df=icg_2nd_der,
        ax=axs[1],
        use_clean=use_clean,
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        color=cmaps.tech_dark[0],
        **kwargs,
    )
    axs[0].axhline(0, color="black", linestyle="--", linewidth=1, zorder=0)
    axs[1].axhline(0, color="black", linestyle="--", linewidth=1, zorder=0)

    _add_heartbeat_borders(heartbeats=heartbeat_borders, ax=axs[0], **kwargs)
    _add_heartbeat_borders(heartbeats=heartbeat_borders, ax=axs[1], **kwargs)

    _add_icg_c_points(icg_data, c_point_samples, ax=axs[0], **kwargs)
    _add_icg_c_points(icg_2nd_der, c_point_samples, ax=axs[1], **kwargs)

    for idx, _row in heartbeats[1:].iterrows():
        if np.isnan(c_point_samples[idx]) or np.isnan(c_point_samples[idx - 1]):
            continue

        c_point = c_point_samples.iloc[idx]
        # Compute the beat to beat interval
        c_point_b2b = c_point_samples.iloc[idx] - c_point_samples.iloc[idx - 1]
        search_interval = int(c_point_b2b / 3)
        start = c_point - search_interval

        axs[0].axvspan(
            icg_data.index[start],
            icg_data.index[c_point],
            color=cmaps.tech_light[0],
            alpha=0.3,
            zorder=0,
            label="A Point Search Windows",
        )
        axs[1].axvspan(
            icg_data.index[start],
            icg_data.index[c_point],
            color=cmaps.tech_light[0],
            alpha=0.3,
            zorder=0,
            label="A Point Search Windows",
        )

        # Detect the local minimum (A-Point) within one third of the beat to beat interval prior to the C-Point
        a_point = b_point_algo._get_a_point(icg_data, search_interval, c_point) + (c_point - search_interval)

        icg_segment = icg_data.iloc[a_point : c_point + 1]
        icg_2nd_der_segment = icg_2nd_der.iloc[a_point : c_point + 1]
        c_amplitude = icg_data.iloc[c_point]

        # Get the most prominent monotonic increasing segment between the A-Point and the C-Point
        start_sample, end_sample = b_point_algo._get_monotonic_increasing_segments_2nd_der(
            icg_segment, icg_2nd_der_segment.to_numpy(), c_amplitude
        )

        start_sample += a_point
        end_sample += a_point
        icg_monotonic_increasing_segment = icg_data.iloc[start_sample : end_sample + 1]
        icg_monotonic_increasing_segment.columns = ["Mono. Incr. Segment"]

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
        significant_zero_crossings = b_point_algo._get_zero_crossings(
            monotonic_segment_3rd_der, monotonic_segment_2nd_der, height, datapoint.sampling_rate_icg
        )
        significant_zero_crossings += start

        # Compute the significant local maximums of the 3rd derivative of the most prominent monotonic segment
        significant_local_maximums = b_point_algo._get_local_maxima(
            monotonic_segment_3rd_der, height, datapoint.sampling_rate_icg
        )
        significant_local_maximums += start

        # Label the last zero crossing/ local maximum as the B-Point
        # If there are no zero crossings or local maximums use the first Point of the segment as B-Point
        significant_features = pd.concat([significant_zero_crossings, significant_local_maximums], axis=0)
        b_point = significant_features.iloc[np.argmin(c_point - significant_features)][0]

        _plot_signals_one_axis(
            df=icg_monotonic_increasing_segment,
            ax=axs[0],
            color=cmaps.fau[0],
            normalize_time=normalize_time,
            **kwargs,
        )
        _plot_signals_one_axis(
            df=icg_2nd_der.iloc[start_sample : end_sample + 1],
            ax=axs[1],
            color=cmaps.fau[0],
            normalize_time=normalize_time,
            **kwargs,
        )

        _add_icg_c_points(
            icg_data,
            a_point,
            ax=axs[0],
            c_point_color=cmaps.wiso_light[1],
            c_point_label="A Points",
            **kwargs,
        )
        _add_icg_c_points(
            icg_2nd_der,
            a_point,
            ax=axs[1],
            c_point_color=cmaps.wiso_light[1],
            c_point_label="A Points",
            **kwargs,
        )

        _add_icg_b_points(
            icg_2nd_der,
            significant_zero_crossings.squeeze(),
            ax=axs[1],
            b_point_label="$d^2Z/dt^2$ Zero Crossings",
            b_point_color=cmaps.nat_dark[0],
            b_point_marker="X",
            **kwargs,
        )
        _add_icg_b_points(
            icg_2nd_der,
            significant_local_maximums.squeeze(),
            ax=axs[1],
            b_point_label="$d^3Z/dt^3$ Local Max.",
            b_point_color=cmaps.nat_dark[1],
            b_point_marker="X",
            **kwargs,
        )

        _add_icg_b_points(
            icg_data,
            b_point_samples_reference,
            ax=axs[0],
            b_point_label="Reference B Points",
            b_point_color=cmaps.phil_dark[0],
            **kwargs,
        )
        _add_icg_b_points(
            icg_data,
            b_point,
            ax=axs[0],
            b_point_label="Detected B Points",
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


def _get_heartbeats(
    datapoint: BaseUnifiedPepExtractionDataset, heartbeat_subset: Optional[Sequence[int]] = None
) -> pd.DataFrame:
    heartbeats = datapoint.heartbeats.drop(columns="start_time")
    if heartbeat_subset is not None:
        heartbeats = heartbeats.loc[heartbeat_subset]
        heartbeats = (heartbeats - heartbeats.iloc[0]["start_sample"]).astype(int)

    return heartbeats.reset_index(drop=True)


def _get_heartbeat_borders(data: pd.DataFrame, heartbeats: pd.DataFrame) -> pd.DataFrame:
    start_samples = data.index[heartbeats["start_sample"]]
    end_sample = data.index[heartbeats["end_sample"] - 1][-1]
    # combine both into one array
    heartbeat_borders = start_samples.append(pd.Index([end_sample]))
    return heartbeat_borders
