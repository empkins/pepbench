"""Module for visualizing Q-wave onset detection and B-point detection algorithms."""

from collections.abc import Sequence
from typing import Any, Optional

import pandas as pd
from biopsykit.signals.ecg.event_extraction import QWaveOnsetExtractionVanLien2013
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
)

__all__ = ["plot_q_wave_onset_extraction_van_lien_2013"]


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
    legend_outside = kwargs.pop("legend_outside", False)
    legend_loc = kwargs.pop("legend_loc", "upper right")
    legend_orientation = kwargs.pop("legend_orientation", "vertical")
    if algo_params is None:
        algo_params = {}

    heartbeat_subset = _sanitize_heartbeat_subset(heartbeat_subset)
    ecg_data, _ = _get_data(
        datapoint, use_clean=use_clean, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset
    )

    fig, ax = _plot_signals_one_axis(
        datapoint, use_clean=use_clean, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset, **kwargs
    )

    # create Q-wave onset extraction algorithm
    heartbeats = datapoint.heartbeats.drop(columns="start_time")
    if heartbeat_subset is not None:
        heartbeats = heartbeats.loc[heartbeat_subset]
        heartbeats = (heartbeats - heartbeats.iloc[0]["start_sample"]).astype(int)
    start_samples = ecg_data.index[heartbeats["start_sample"]]
    end_sample = ecg_data.index[heartbeats["end_sample"] - 1][-1]
    # combine both into one array
    heartbeat_borders = start_samples.append(pd.Index([end_sample]))

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
        legend_outside=legend_outside,
        legend_loc=legend_loc,
        legend_orientation=legend_orientation,
        fig=fig,
        ax=ax,
        **kwargs,
    )

    return fig, ax
