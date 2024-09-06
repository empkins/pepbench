"""Module for visualizing Q-wave onset detection and B-point detection algorithms."""

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
import pandas as pd
from biopsykit.signals.ecg.event_extraction import QWaveOnsetExtractionVanLien2013
from biopsykit.signals.icg.event_extraction import CPointExtractionScipyFindPeaks, BPointExtractionDebski1993
from fau_colors import cmaps
from matplotlib import pyplot as plt

from pepbench.datasets import BaseUnifiedPepExtractionDataset
from pepbench.plotting._base_plotting import _plot_signals_one_axis
from pepbench.plotting._utils import (
    _add_ecg_q_wave_onsets,
    _add_ecg_r_peaks,
    _add_heartbeat_borders,
    _get_annotation_bbox_no_edge,
    _get_data,
    _handle_legend_one_axis,
    _sanitize_heartbeat_subset,
    _get_legend_loc,
    _get_rect,
    _handle_legend_two_axes,
    _add_icg_c_points,
    _add_icg_b_points,
)

__all__ = ["plot_q_wave_onset_extraction_van_lien_2013", "plot_b_point_extraction_debski1993"]


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

    fig, ax = _plot_signals_one_axis(
        datapoint=datapoint,
        use_clean=use_clean,
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        plot_icg=False,
        **kwargs,
    )

    heartbeats = _get_heartbeats(datapoint, heartbeat_subset)
    heartbeat_borders = _get_heartbeat_borders(ecg_data, heartbeats)

    q_wave_onset_algo = QWaveOnsetExtractionVanLien2013(**algo_params)
    q_wave_onset_algo.extract(ecg_data, heartbeats=heartbeats, sampling_rate_hz=datapoint.sampling_rate_ecg)

    r_peak_samples = heartbeats["r_peak_sample"].astype(int)
    q_wave_onset_samples = q_wave_onset_algo.points_["q_wave_onset_sample"].astype(int)

    time_interval_ms = q_wave_onset_algo.get_params()["time_interval_ms"]

    kwargs.setdefault("r_peak_linewidth", 2)
    kwargs.setdefault("r_peak_linestyle", "--")
    kwargs.setdefault("r_peak_marker", "x")
    kwargs.setdefault("q_wave_linewidth", 2)

    _add_heartbeat_borders(heartbeat_borders, ax=ax, **kwargs)
    _add_ecg_r_peaks(ecg_data, r_peak_samples, ax=ax, **kwargs)
    _add_ecg_q_wave_onsets(ecg_data, q_wave_onset_samples, ax=ax, **kwargs)

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

    _handle_legend_one_axis(
        fig=fig,
        ax=ax,
        **kwargs,
    )

    old_ylims = ax.get_ylim()
    ax.set_ylim(old_ylims[0], 1.15 * old_ylims[1])

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
) -> tuple[plt.Figure, plt.Axes]:
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

    # compute ICG derivation
    icg_2nd_der = np.gradient(icg_data.squeeze())
    icg_2nd_der = pd.DataFrame(icg_2nd_der, index=icg_data.index, columns=["ICG Deriv. $(dZ^2/dt^2)$"])

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

    fig, ax = _plot_signals_one_axis(
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
        b_point_label="$dZ^2/dt^2$ Local Min.",
        b_point_marker="x",
        b_point_color=cmaps.phil_dark[0],
        **kwargs,
    )
    _add_icg_b_points(icg_data, b_point_samples, ax=axs[0], b_point_label="B Points", **kwargs)

    for idx, row in search_window.iterrows():
        start = icg_2nd_der.index[row["r_peak_sample"]]
        end = icg_2nd_der.index[row["c_point_sample"]]
        axs[1].axvspan(start, end, color=cmaps.fau_light[1], alpha=0.3, zorder=0, label="B Point Search Windows")

    _handle_legend_two_axes(
        fig=fig,
        axs=axs,
        **kwargs,
    )

    fig.align_ylabels()
    fig.tight_layout(rect=rect)

    return fig, ax


def _get_heartbeats(datapoint: BaseUnifiedPepExtractionDataset, heartbeat_subset: Optional[Sequence[int]] = None):
    heartbeats = datapoint.heartbeats.drop(columns="start_time")
    if heartbeat_subset is not None:
        heartbeats = heartbeats.loc[heartbeat_subset]
        heartbeats = (heartbeats - heartbeats.iloc[0]["start_sample"]).astype(int)

    return heartbeats


def _get_heartbeat_borders(data: pd.DataFrame, heartbeats: pd.DataFrame) -> pd.DataFrame:
    start_samples = data.index[heartbeats["start_sample"]]
    end_sample = data.index[heartbeats["end_sample"] - 1][-1]
    # combine both into one array
    heartbeat_borders = start_samples.append(pd.Index([end_sample]))
    return heartbeat_borders
