from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from fau_colors import cmaps
from matplotlib import pyplot as plt

from pepbench.datasets import BaseUnifiedPepExtractionDataset


def _get_fig_ax(**kwargs: Any) -> tuple[plt.Figure, plt.Axes]:
    ax: Union[plt.Axes, None] = kwargs.pop("ax", None)

    if ax is not None:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(**kwargs)
    return fig, ax


def _get_fig_axs(**kwargs: Any) -> tuple[plt.Figure, Sequence[plt.Axes]]:
    axs: Union[plt.Axes, None] = kwargs.pop("axs", None)

    if axs is not None:
        fig = axs[0].get_figure()
    else:
        fig, axs = plt.subplots(**kwargs)
    return fig, axs


def _add_ecg_r_peaks(
    ecg_data: pd.DataFrame,
    r_peaks: pd.DataFrame,
    ax: plt.Axes,
    **kwargs: Any,
) -> None:
    label = kwargs.get("r_peak_label", "R-Peaks")
    color = kwargs.get("r_peak_color", cmaps.wiso[0])
    marker = kwargs.get("r_peak_marker", "o")
    linestyle = kwargs.get("r_peak_linestyle", "-")
    linewidth = kwargs.get("r_peak_linewidth", 1)
    alpha = kwargs.get("r_peak_alpha", 0.7)

    print(kwargs)

    r_peaks = r_peaks.astype(int)

    _base_add_scatter(
        x=ecg_data.index[r_peaks], y=ecg_data.iloc[r_peaks], color=color, label=label, marker=marker, ax=ax
    )
    _base_add_vlines(
        x=ecg_data.index[r_peaks],
        color=color,
        alpha=alpha,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
        ax=ax,
    )


def _add_ecg_r_peak_artefacts(
    ecg_data: pd.DataFrame,
    r_peaks: pd.DataFrame,
    ax: plt.Axes,
    **kwargs: Any,
) -> None:
    label = kwargs.get("r_peak_artefact_label", "R-Peak Artefacts")
    color = kwargs.get("r_peak_artefact_color", cmaps.wiso_dark[0])
    marker = kwargs.get("r_peak_artefact_marker", "X")
    linestyle = kwargs.get("r_peak_artefact_linestyle", "-")

    kwargs = kwargs.copy()
    kwargs.update(r_peak_label=label, r_peak_color=color, r_peak_marker=marker, r_peak_linestyle=linestyle)

    _add_ecg_r_peaks(
        ecg_data,
        r_peaks,
        ax,
        **kwargs,
    )


def _add_ecg_q_wave_onsets(
    ecg_data: pd.DataFrame,
    q_wave_onsets: pd.DataFrame,
    ax: plt.Axes,
    **kwargs: Any,
) -> None:
    label = kwargs.get("q_wave_label", "Q-Wave Onsets")
    color = kwargs.get("q_wave_color", cmaps.med[0])
    marker = kwargs.get("q_wave_marker", "o")
    linestyle = kwargs.get("q_wave_linestyle", "-")
    linewidth = kwargs.get("q_wave_linewidth", 1)
    alpha = kwargs.get("q_wave_alpha", 0.7)

    q_wave_onsets = q_wave_onsets.astype(int)

    _base_add_scatter(
        x=ecg_data.index[q_wave_onsets], y=ecg_data.iloc[q_wave_onsets], color=color, label=label, marker=marker, ax=ax
    )
    _base_add_vlines(
        x=ecg_data.index[q_wave_onsets],
        color=color,
        alpha=alpha,
        label=label,
        linestyle=linestyle,
        linewidth=linewidth,
        ax=ax,
    )


def _add_ecg_q_wave_onset_artefacts(
    ecg_data: pd.DataFrame,
    q_wave_artefacts: pd.DataFrame,
    ax: plt.Axes,
    **kwargs: Any,
) -> None:
    label = kwargs.get("q_wave_artefact_label", "Q-Wave Artefacts")
    color = kwargs.get("q_wave_artefact_color", cmaps.med_dark[0])
    marker = kwargs.get("q_wave_artefact_marker", "X")
    linestyle = kwargs.get("q_wave_artefact_linestyle", "-")

    kwargs = kwargs.copy()
    kwargs.update(q_wave_label=label, q_wave_color=color, q_wave_marker=marker, q_wave_linestyle=linestyle)

    _add_ecg_q_wave_onsets(
        ecg_data,
        q_wave_artefacts,
        ax,
        **kwargs,
    )


def _add_icg_b_points(
    icg_data: pd.DataFrame,
    b_points: pd.DataFrame,
    ax: plt.Axes,
    **kwargs: Any,
) -> None:
    color = kwargs.get("b_point_color", cmaps.phil[0])
    b_point_label = kwargs.get("b_point_label", "B Points")
    marker = kwargs.get("b_point_marker", "o")
    linestyle = kwargs.get("b_point_linestyle", "-")
    alpha = kwargs.get("b_point_alpha", 0.7)

    b_points = b_points.astype(int)

    _base_add_scatter(
        x=icg_data.index[b_points], y=icg_data.iloc[b_points], color=color, label=b_point_label, marker=marker, ax=ax
    )
    _base_add_vlines(
        x=icg_data.index[b_points], color=color, alpha=alpha, label=b_point_label, linestyle=linestyle, ax=ax
    )


def _add_icg_b_point_artefacts(
    icg_data: pd.DataFrame,
    b_points: pd.DataFrame,
    ax: plt.Axes,
    **kwargs: Any,
) -> None:
    color = kwargs.get("b_point_artefact_color", cmaps.phil_dark[0])
    label = kwargs.get("b_point_artefact_label", "B Point Artefacts")
    marker = kwargs.get("b_point_artefact_marker", "X")
    linestyle = kwargs.get("b_point_artefact_linestyle", "-")

    kwargs = kwargs.copy()
    kwargs.update(b_point_color=color, b_point_label=label, b_point_marker=marker, b_point_linestyle=linestyle)

    _add_icg_b_points(
        icg_data,
        b_points,
        ax,
        **kwargs,
    )


def _base_add_vlines(
    x: pd.Series,
    color: str,
    alpha: float,
    label: str,
    linestyle: str,
    linewidth: float,
    ax: plt.Axes,
) -> None:

    ax.vlines(
        x=x,
        ymin=0,
        ymax=1,
        color=color,
        transform=ax.get_xaxis_transform(),
        zorder=3,
        alpha=alpha,
        ls=linestyle,
        lw=linewidth,
        label=label,
    )


def _base_add_scatter(
    x: pd.Series,
    y: pd.Series,
    color: str,
    label: str,
    marker: str,
    ax: plt.Axes,
) -> None:

    ax.scatter(
        x=x,
        y=y,
        color=color,
        label=label,
        zorder=3,
        marker=marker,
    )


def _add_pep_from_reference(
    ecg_data: pd.DataFrame,
    icg_data: pd.DataFrame,
    labels: pd.DataFrame,
    ax: plt.Axes,
    **kwargs: Any,
) -> None:
    color = kwargs.get("pep_color", cmaps.nat[0])
    color_artefact = kwargs.get("pep_artefact_color", cmaps.wiso[0])
    edgecolor = kwargs.get("pep_edgecolor", color)
    facecolor = kwargs.get("pep_facecolor", color)
    edgecolor_artefact = kwargs.get("pep_artefact_edgecolor", color_artefact)
    facecolor_artefact = kwargs.get("pep_artefact_facecolor", color_artefact)
    pep_label = kwargs.get("pep_label", "PEP")

    hatch = kwargs.get("pep_hatch", None)
    if hatch is not None:
        facecolor = "none"
        facecolor_artefact = "none"

    for _i, row in labels.iterrows():
        start = ecg_data.index[row[("ecg", "sample_relative")]]
        end = icg_data.index[row[("icg", "sample_relative")]]
        if row[("ecg", "label")] == "Artefact" or row[("icg", "label")] == "Artefact":
            ax.axvspan(
                start,
                end,
                edgecolor=edgecolor_artefact,
                facecolor=facecolor_artefact,
                hatch=hatch,
                alpha=0.3,
                zorder=0,
                label=f"{pep_label} Artefact",
            )
        else:
            ax.axvspan(
                start,
                end,
                edgecolor=edgecolor,
                facecolor=facecolor,
                hatch=hatch,
                alpha=0.3,
                zorder=0,
                label=pep_label,
            )


def _add_pep_from_results(
    ecg_data: pd.DataFrame,
    icg_data: pd.DataFrame,
    labels: pd.DataFrame,
    ax: plt.Axes,
    **kwargs: Any,
) -> None:
    color = kwargs.get("pep_color", cmaps.nat[0])
    edgecolor = kwargs.get("pep_edgecolor", color)
    facecolor = kwargs.get("pep_facecolor", color)
    label = kwargs.get("pep_label", "PEP")

    hatch = kwargs.get("pep_hatch", None)
    if hatch is not None:
        facecolor = "none"

    for _i, row in labels.iterrows():
        start = ecg_data.index[row["ecg"]]
        end = icg_data.index[row["icg"]]
        ax.axvspan(
            start,
            end,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=0.3,
            zorder=0,
            hatch=hatch,
            label=label,
        )


def _add_heartbeat_borders(heartbeats: pd.DataFrame, ax: plt.Axes, **kwargs: Any) -> None:
    color = kwargs.get("heartbeat_border_color", cmaps.tech[2])
    ax.vlines(
        x=heartbeats,
        ymin=0,
        ymax=1,
        zorder=0,
        colors=color,
        transform=ax.get_xaxis_transform(),
        label="Heartbeat Borders",
        ls="--",
    )


def _handle_legend_one_axis(
    legend_orientation: str,
    legend_outside: bool,
    legend_loc: str,
    fig: plt.Figure,
    ax: plt.Axes,
    max_cols: Optional[int] = 5,
    **kwargs: Any,  # noqa: ARG001
) -> None:
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = _remove_duplicate_legend_entries(handles, labels)
    ncols = min(len(handles), max_cols) if legend_orientation == "horizontal" else 1

    if len(fig.legends) > 0:
        fig.legends[0].remove()

    if legend_outside:
        ax.legend().remove()
        fig.legend(handles=handles, labels=labels, ncols=ncols, loc=legend_loc)
    else:
        ax.legend(handles=handles, labels=labels, loc=legend_loc, ncols=ncols)


def _handle_legend_two_axes(
    legend_orientation: str,
    legend_outside: bool,
    legend_loc: str,
    fig: plt.Figure,
    axs: Sequence[plt.Axes],
    max_cols: Optional[int] = 5,
    **kwargs: Any,  # noqa: ARG001
) -> None:

    if len(fig.legends) > 0:
        fig.legends[0].remove()
    if legend_outside:
        handles, labels = axs[0].get_legend_handles_labels()
        handles += axs[1].get_legend_handles_labels()[0]
        labels += axs[1].get_legend_handles_labels()[1]
        for ax in axs:
            ax.legend().remove()

        ncols = min(len(handles), max_cols) if legend_orientation == "horizontal" else 1
        handles, labels = _remove_duplicate_legend_entries(handles, labels)

        fig.legend(handles, labels, ncols=ncols, loc=legend_loc)
    else:
        for ax in axs:
            ax.legend(loc=legend_loc)

    for ax in axs:
        ax.set_xlabel("Time [hh:mm:ss]")
        ax.set_ylabel("Amplitude [a.u.]")


def _remove_duplicate_legend_entries(handles: Sequence[plt.Artist], labels: Sequence[str]) -> tuple[list, list]:
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    return unique_handles, unique_labels


def _sanitize_heartbeat_subset(heartbeat_subsample: Optional[Sequence[int]] = None) -> Optional[Sequence[int]]:
    if heartbeat_subsample is None:
        return None
    if len(heartbeat_subsample) == 1:
        return heartbeat_subsample
    # if heartbeat_subsample is a tuple of two integers, assume it's a range
    if isinstance(heartbeat_subsample, tuple) and len(heartbeat_subsample) == 2:
        return list(range(heartbeat_subsample[0], heartbeat_subsample[1] + 1))
    # if it's a sequence but not incremented by 1, it's an error
    if np.ediff1d(heartbeat_subsample).max() != 1:
        raise ValueError("Heartbeat subsample must be a range or list of indices.")
    return heartbeat_subsample


def _get_data(
    datapoint: BaseUnifiedPepExtractionDataset,
    *,
    use_clean: bool,
    normalize_time: bool,
    heartbeat_subset: Union[Sequence[int], None],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ecg_data = datapoint.ecg_clean if use_clean else datapoint.ecg
    icg_data = datapoint.icg_clean if use_clean else datapoint.icg

    ecg_data.columns = ["ECG"]
    icg_data.columns = ["ICG ($dZ/dt$)"]

    heartbeat_subset = _sanitize_heartbeat_subset(heartbeat_subset)

    if heartbeat_subset is not None:
        heartbeat_borders = datapoint.reference_heartbeats
        start = heartbeat_borders["start_sample"].iloc[heartbeat_subset[0]]
        end = heartbeat_borders["end_sample"].iloc[heartbeat_subset[-1]]

        ecg_data = ecg_data.iloc[start:end]
        icg_data = icg_data.iloc[start:end]

    if normalize_time:
        ecg_data.index = (ecg_data.index - ecg_data.index[0]).total_seconds()
        icg_data.index = (icg_data.index - icg_data.index[0]).total_seconds()

    return ecg_data, icg_data


def _get_reference_labels(
    datapoint: BaseUnifiedPepExtractionDataset,
    heartbeat_subset: Optional[Sequence[int]] = None,
) -> dict[str, pd.DataFrame]:
    reference_heartbeats = datapoint.reference_heartbeats
    reference_labels_ecg = datapoint.reference_labels_ecg
    reference_labels_icg = datapoint.reference_labels_icg
    q_wave_onsets = reference_labels_ecg.xs("ECG", level="channel")["sample_relative"]
    q_wave_artefacts = reference_labels_ecg.reindex(["Artefact"], level="channel")["sample_relative"]
    b_points = reference_labels_icg.xs("ICG", level="channel")["sample_relative"]
    b_point_artefacts = reference_labels_icg.reindex(["Artefact"], level="channel")["sample_relative"]

    heartbeat_subset = _sanitize_heartbeat_subset(heartbeat_subset)

    if heartbeat_subset is not None:
        reference_heartbeats = reference_heartbeats.iloc[heartbeat_subset]
        q_wave_onsets = q_wave_onsets.reindex(heartbeat_subset, level="heartbeat_id").dropna()
        q_wave_artefacts = q_wave_artefacts.reindex(heartbeat_subset, level="heartbeat_id").dropna()
        b_points = b_points.reindex(heartbeat_subset, level="heartbeat_id").dropna()
        b_point_artefacts = b_point_artefacts.reindex(heartbeat_subset, level="heartbeat_id").dropna()

    return_dict = {
        "reference_heartbeats": reference_heartbeats,
        "q_wave_onsets": q_wave_onsets,
        "q_wave_artefacts": q_wave_artefacts,
        "b_points": b_points,
        "b_point_artefacts": b_point_artefacts,
    }

    start_sample = reference_heartbeats["start_sample"].iloc[0]
    # subtract start_sample to get relative sample indices
    return {key: value - start_sample for key, value in return_dict.items()}


def _get_labels_from_challenge_results(
    pep_results_per_sample: pd.DataFrame, heartbeat_subsample: Sequence[int]
) -> dict[str, pd.DataFrame]:
    heartbeat_subsample = _sanitize_heartbeat_subset(heartbeat_subsample)
    pep_results_per_sample = pep_results_per_sample.reindex(heartbeat_subsample, level="heartbeat_id")

    heartbeats_start = pep_results_per_sample[("heartbeat_start_sample", "estimated")]
    heartbeats_end = pep_results_per_sample[("heartbeat_end_sample", "estimated")]

    start_index = heartbeats_start.iloc[0]

    q_wave_labels_reference = pep_results_per_sample[("q_wave_onset_sample", "reference")]
    q_wave_labels_estimated = pep_results_per_sample[("q_wave_onset_sample", "estimated")]
    b_point_labels_reference = pep_results_per_sample[("b_point_sample", "reference")]
    b_point_labels_estimated = pep_results_per_sample[("b_point_sample", "estimated")]

    return_dict = {
        "heartbeats_start": heartbeats_start,
        "heartbeats_end": heartbeats_end,
        "q_wave_labels_reference": q_wave_labels_reference,
        "q_wave_labels_estimated": q_wave_labels_estimated,
        "b_point_labels_reference": b_point_labels_reference,
        "b_point_labels_estimated": b_point_labels_estimated,
    }
    return {key: (val.dropna() - start_index).astype(int) for key, val in return_dict.items()}


def _get_rect(**kwargs: Any) -> tuple[float, ...]:
    rect = kwargs.pop("rect", None)
    legend_outside = kwargs.get("legend_outside", False)
    legend_orientation = kwargs.get("legend_orientation", "vertical")

    if rect is not None:
        return rect
    if legend_outside:
        return (0, 0, 0.80, 1.0) if legend_orientation == "vertical" else (0, 0, 1, 0.90)
    return (0, 0, 1, 1)


def _get_legend_loc(**kwargs: Any) -> str:
    legend_loc = kwargs.get("legend_loc", None)
    legend_outside = kwargs.get("legend_outside", False)
    legend_orientation = kwargs.get("legend_orientation", "vertical")

    if legend_loc is not None:
        return legend_loc
    if legend_outside:
        return "upper right" if legend_orientation == "vertical" else "upper center"
    return "upper right" if legend_orientation == "vertical" else "lower center"


def _get_annotation_bbox() -> dict[str, Any]:
    return {
        "fc": (1, 1, 1, plt.rcParams["legend.framealpha"]),
        "ec": plt.rcParams["legend.edgecolor"],
        "boxstyle": "round",
    }


def _get_annotation_bbox_no_edge() -> dict[str, Any]:
    return {
        "fc": (1, 1, 1, plt.rcParams["legend.framealpha"]),
        "ec": "none",
        "boxstyle": "round",
    }
