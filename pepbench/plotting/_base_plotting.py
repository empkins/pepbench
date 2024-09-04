from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from fau_colors import cmaps
from matplotlib import pyplot as plt

from pepbench.datasets import BaseUnifiedPepExtractionDataset
from pepbench.plotting._utils import _get_fig_ax, _get_fig_axs

__all__ = [
    "plot_signals",
    "plot_signals_with_reference_labels",
    "plot_signals_from_challenge_results",
    "plot_signals_with_reference_pep",
]


def plot_signals(
    datapoint: BaseUnifiedPepExtractionDataset,
    *,
    collapse: Optional[bool] = False,
    use_clean: Optional[bool] = True,
    heartbeat_subsample: Optional[Sequence[int]] = None,
    **kwargs: Any,
) -> tuple[plt.Figure, Union[plt.Axes, Sequence[plt.Axes]]]:
    """Plot ECG and ICG signals."""
    if collapse:
        return _plot_signals_one_axis(datapoint, use_clean=use_clean, heartbeat_subsample=heartbeat_subsample, **kwargs)
    return _plot_signals_two_axes(datapoint, use_clean=use_clean, heartbeat_subsample=heartbeat_subsample, **kwargs)


def plot_signals_with_reference_labels(
    datapoint: BaseUnifiedPepExtractionDataset,
    *,
    collapse: Optional[bool] = False,
    use_clean: Optional[bool] = True,
    heartbeat_subsample: Optional[Sequence[int]] = None,
    **kwargs: Any,
) -> tuple[plt.Figure, Union[plt.Axes, Sequence[plt.Axes]]]:
    legend_orientation = kwargs.get("legend_orientation", "vertical")
    legend_outside = kwargs.get("legend_outside", False)
    legend_max_cols = kwargs.get("legend_max_cols", 6)
    legend_loc = _get_legend_loc(**kwargs)
    rect = _get_rect(**kwargs)

    fig, ax = plot_signals(
        datapoint, collapse=collapse, use_clean=use_clean, heartbeat_subsample=heartbeat_subsample, **kwargs
    )
    ecg_data, icg_data = _get_data(datapoint, use_clean, heartbeat_subsample)

    reference_labels = _get_reference_labels(datapoint, heartbeat_subsample=heartbeat_subsample)
    reference_heartbeats = reference_labels["reference_heartbeats"]
    q_wave_onsets = reference_labels["q_wave_onsets"]
    q_wave_artefacts = reference_labels["q_wave_artefacts"]
    b_points = reference_labels["b_points"]
    b_point_artefacts = reference_labels["b_point_artefacts"]

    # plot q-wave onsets and b-points
    if collapse:
        _add_heartbeat_borders(ecg_data.index[reference_heartbeats["start_sample"]], ax, **kwargs)
        _add_ecg_q_wave_onsets(ecg_data, q_wave_onsets, q_wave_artefacts, ax, **kwargs)
        _add_icg_b_points(icg_data, b_points, b_point_artefacts, ax, **kwargs)
        _handle_legend_one_axis(legend_orientation, legend_outside, legend_loc, fig, ax, max_cols=legend_max_cols)
    else:
        _add_heartbeat_borders(ecg_data.index[reference_heartbeats["start_sample"]], ax[0], **kwargs)
        _add_heartbeat_borders(ecg_data.index[reference_heartbeats["start_sample"]], ax[1], **kwargs)
        _add_ecg_q_wave_onsets(ecg_data, q_wave_onsets, q_wave_artefacts, ax[0], **kwargs)
        _add_icg_b_points(icg_data, b_points, b_point_artefacts, ax[1], **kwargs)
        _handle_legend_two_axes(legend_orientation, legend_outside, legend_loc, fig, ax, max_cols=legend_max_cols)

    fig.tight_layout(rect=rect)
    return fig, ax


def plot_signals_with_reference_pep(
    datapoint: BaseUnifiedPepExtractionDataset,
    *,
    use_clean: Optional[bool] = True,
    heartbeat_subsample: Optional[Sequence[int]] = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    legend_orientation = kwargs.get("legend_orientation", "vertical")
    legend_outside = kwargs.get("legend_outside", False)
    legend_max_cols = kwargs.get("legend_max_cols", 5)
    legend_loc = _get_legend_loc(**kwargs)
    rect = _get_rect(**kwargs)

    fig, ax = plot_signals_with_reference_labels(
        datapoint, use_clean=use_clean, collapse=True, heartbeat_subsample=heartbeat_subsample, **kwargs
    )

    _add_pep(datapoint, use_clean, heartbeat_subsample, ax, **kwargs)

    _handle_legend_one_axis(legend_orientation, legend_outside, legend_loc, fig, ax, max_cols=legend_max_cols)
    fig.tight_layout(rect=rect)
    return fig, ax


def plot_signals_from_challenge_results(
    datapoint: BaseUnifiedPepExtractionDataset,
    pep_results_per_sample: pd.DataFrame,
    *,
    collapse: Optional[bool] = False,
    use_clean: Optional[bool] = True,
    heartbeat_subsample: Optional[Sequence[int]] = None,
    add_pep: Optional[bool] = False,
    **kwargs: Any,
) -> tuple[plt.Figure, Sequence[plt.Axes]]:
    legend_orientation = kwargs.get("legend_orientation", "vertical")
    legend_outside = kwargs.get("legend_outside", False)
    legend_max_cols = kwargs.get("legend_max_cols", 5)
    legend_loc = _get_legend_loc(**kwargs)
    rect = _get_rect(**kwargs)

    fig, axs = plot_signals(
        datapoint, use_clean=use_clean, heartbeat_subsample=heartbeat_subsample, collapse=collapse, **kwargs
    )

    ecg_data, icg_data = _get_data(datapoint, use_clean, heartbeat_subsample)

    labels_from_challenge = _get_labels_from_challenge_results(pep_results_per_sample, heartbeat_subsample)

    heartbeats_start = ecg_data.index[labels_from_challenge["heartbeats_start"]]
    heartbeats_end = ecg_data.index[labels_from_challenge["heartbeats_end"] - 1]
    q_wave_labels_reference = labels_from_challenge["q_wave_labels_reference"]
    q_wave_labels_estimated = labels_from_challenge["q_wave_labels_estimated"]
    b_point_labels_reference = labels_from_challenge["b_point_labels_reference"]
    b_point_labels_estimated = labels_from_challenge["b_point_labels_estimated"]

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

    _add_ecg_q_wave_onsets(
        ecg_data,
        q_wave_labels_reference,
        pd.DataFrame(),
        ax_ecg,
        q_wave_color=cmaps.med[0],
        q_wave_label="Q-Wave Reference",
        plot_artifacts=False,
    )
    _add_ecg_q_wave_onsets(
        ecg_data,
        q_wave_labels_estimated,
        pd.DataFrame(),
        ax_ecg,
        q_wave_color=cmaps.med_dark[0],
        q_wave_label="Q-Wave Estimated",
        plot_artifacts=False,
    )
    _add_icg_b_points(
        icg_data,
        b_point_labels_reference,
        pd.DataFrame(),
        ax_icg,
        b_point_color=cmaps.phil[0],
        b_point_label="B-Point Reference",
        plot_artifacts=False,
    )
    _add_icg_b_points(
        icg_data,
        b_point_labels_estimated,
        pd.DataFrame(),
        ax_icg,
        b_point_color=cmaps.phil_dark[0],
        b_point_label="B-Point Estimated",
        plot_artifacts=False,
    )

    if add_pep:
        _add_pep(datapoint, use_clean, heartbeat_subsample=heartbeat_subsample, ax=ax_icg, **kwargs)
        if not collapse:
            _add_pep(datapoint, use_clean, heartbeat_subsample=heartbeat_subsample, ax=ax_ecg, **kwargs)

    if collapse:
        _handle_legend_one_axis(legend_orientation, legend_outside, legend_loc, fig, axs, max_cols=legend_max_cols)
    else:
        _handle_legend_two_axes(legend_orientation, legend_outside, legend_loc, fig, axs, max_cols=legend_max_cols)

    fig.tight_layout(rect=rect)

    return fig, axs


def _plot_signals_one_axis(
    datapoint: BaseUnifiedPepExtractionDataset,
    *,
    use_clean: Optional[bool] = True,
    heartbeat_subsample: Optional[Sequence[int]] = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    figsize = kwargs.pop("figsize", None)
    legend_outside = kwargs.get("legend_outside", False)
    legend_orientation = kwargs.get("legend_orientation", "vertical")
    legend_max_cols = kwargs.get("legend_max_cols", 5)

    legend_loc = _get_legend_loc(**kwargs)
    rect = _get_rect(**kwargs)

    fig, ax = _get_fig_ax(figsize=figsize)

    ecg_data, icg_data = _get_data(datapoint, use_clean, heartbeat_subsample)

    ecg_data.plot(ax=ax)
    icg_data.plot(ax=ax)

    ax.set_xlabel("Time [hh:mm:ss]")
    ax.set_ylabel("Amplitude [a.u.]")

    _handle_legend_one_axis(legend_orientation, legend_outside, legend_loc, fig, ax, max_cols=legend_max_cols)

    fig.tight_layout(rect=rect)

    return fig, ax


def _plot_signals_two_axes(
    datapoint: BaseUnifiedPepExtractionDataset,
    *,
    use_clean: Optional[bool] = True,
    heartbeat_subsample: Optional[Sequence[int]] = None,
    **kwargs: Any,
) -> tuple[plt.Figure, Sequence[plt.Axes]]:
    figsize = kwargs.pop("figsize", None)
    legend_orientation = kwargs.get("legend_orientation", "vertical")
    legend_outside = kwargs.get("legend_outside", False)
    legend_loc = kwargs.get("legend_loc", None)
    legend_max_cols = kwargs.get("legend_max_cols", 5)
    rect = kwargs.get("rect", None)

    if legend_loc is None:
        legend_loc = _get_legend_loc(**kwargs)
    if rect is None:
        rect = _get_rect(**kwargs)

    fig, axs = _get_fig_axs(figsize=figsize, nrows=2, sharex=True)

    colors = iter(cmaps.faculties)

    ecg_data, icg_data = _get_data(datapoint, use_clean, heartbeat_subsample)
    ecg_data.plot(ax=axs[0], color=next(colors), title="Electrocardiogram (ECG)")
    icg_data.plot(ax=axs[1], color=next(colors), title="Impedance Cardiogram (ICG)")

    _handle_legend_two_axes(legend_orientation, legend_outside, legend_loc, fig, axs, max_cols=legend_max_cols)

    fig.align_ylabels()
    fig.tight_layout(rect=rect)

    return fig, axs


def _add_heartbeat_borders(heartbeats: pd.DataFrame, ax: plt.Axes, **kwargs: Any) -> None:
    color = kwargs.get("heartbeat_color", cmaps.tech[2])
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


def _add_ecg_q_wave_onsets(
    ecg_data: pd.DataFrame,
    q_wave_onsets: pd.DataFrame,
    q_wave_artefacts: pd.DataFrame,
    ax: plt.Axes,
    **kwargs: Any,
) -> None:
    label = kwargs.get("q_wave_label", "Q-Wave Onsets")
    color = kwargs.get("q_wave_color", cmaps.med[0])
    plot_artifacts = kwargs.get("plot_artifacts", True)

    q_wave_onsets = q_wave_onsets.astype(int)
    ax.vlines(
        x=ecg_data.index[q_wave_onsets],
        ymin=0,
        ymax=1,
        color=color,
        transform=ax.get_xaxis_transform(),
        zorder=3,
        alpha=0.7,
    )
    ax.scatter(
        x=ecg_data.index[q_wave_onsets],
        y=ecg_data.iloc[q_wave_onsets],
        color=color,
        label=label,
        zorder=3,
    )
    if plot_artifacts and not q_wave_artefacts.empty:
        ax.vlines(
            x=ecg_data.index[q_wave_artefacts],
            ymin=0,
            ymax=1,
            color=color,
            transform=ax.get_xaxis_transform(),
            zorder=3,
            alpha=0.7,
            ls="dotted",
        )
        ax.scatter(
            x=ecg_data.index[q_wave_artefacts],
            y=ecg_data.iloc[q_wave_artefacts],
            color=color,
            label="Q-Wave Artefacts",
            zorder=3,
            marker="X",
        )


def _add_icg_b_points(
    icg_data: pd.DataFrame,
    b_points: pd.DataFrame,
    b_point_artefacts: pd.DataFrame,
    ax: plt.Axes,
    **kwargs: Any,
) -> None:
    color = kwargs.get("b_point_color", cmaps.phil[0])
    b_point_label = kwargs.get("b_point_label", "B Points")
    plot_artifacts = kwargs.get("plot_artifacts", True)

    b_points = b_points.astype(int)

    ax.vlines(
        x=icg_data.index[b_points],
        ymin=0,
        ymax=1,
        color=color,
        transform=ax.get_xaxis_transform(),
        zorder=3,
        alpha=0.7,
    )
    ax.scatter(
        x=icg_data.index[b_points],
        y=icg_data.iloc[b_points],
        color=color,
        label=b_point_label,
        zorder=3,
    )
    if plot_artifacts and not b_point_artefacts.empty:
        ax.vlines(
            x=icg_data.index[b_point_artefacts],
            ymin=0,
            ymax=1,
            color=color,
            transform=ax.get_xaxis_transform(),
            zorder=3,
            alpha=0.7,
            ls="dotted",
        )
        ax.scatter(
            x=icg_data.index[b_point_artefacts],
            y=icg_data.iloc[b_point_artefacts],
            color=color,
            label="B Point Artefacts",
            zorder=3,
            marker="X",
        )


def _handle_legend_one_axis(
    legend_orientation: str,
    legend_outside: bool,
    legend_loc: str,
    fig: plt.Figure,
    ax: plt.Axes,
    max_cols: Optional[int] = 5,
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


def _add_pep(
    datapoint: BaseUnifiedPepExtractionDataset,
    use_clean: bool,
    heartbeat_subsample: Optional[Sequence[int]],
    ax: plt.Axes,
    **kwargs: Any,
) -> None:
    color = kwargs.get("pep_color", cmaps.nat[0])
    color_artefact = kwargs.get("pep_artefact_color", cmaps.wiso[0])

    ecg_data, icg_data = _get_data(datapoint, use_clean=use_clean, heartbeat_subsample=heartbeat_subsample)
    reference_labels = _get_reference_labels(datapoint, heartbeat_subsample)

    reference_labels_ecg = (
        pd.concat([reference_labels["q_wave_onsets"], reference_labels["q_wave_artefacts"]]).sort_index().reset_index()
    )
    reference_labels_icg = (
        pd.concat([reference_labels["b_points"], reference_labels["b_point_artefacts"]]).sort_index().reset_index()
    )

    labels = pd.concat({"ecg": reference_labels_ecg, "icg": reference_labels_icg}, axis=1)

    for _i, row in labels.iterrows():
        start = ecg_data.index[row[("ecg", "sample_relative")]]
        end = icg_data.index[row[("icg", "sample_relative")]]
        if row[("ecg", "label")] == "Artefact" or row[("icg", "label")] == "Artefact":
            ax.axvspan(start, end, color=color_artefact, alpha=0.3, zorder=0, label="PEP Artefact")
        else:
            ax.axvspan(start, end, color=color, alpha=0.3, zorder=0, label="PEP")


def _handle_legend_two_axes(
    legend_orientation: str,
    legend_outside: bool,
    legend_loc: str,
    fig: plt.Figure,
    axs: Sequence[plt.Axes],
    max_cols: Optional[int] = 5,
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


def _sanitize_heartbeat_subsample(heartbeat_subsample: Optional[Sequence[int]] = None) -> Optional[Sequence[int]]:
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
    datapoint: BaseUnifiedPepExtractionDataset, use_clean: bool, heartbeat_subsample: Union[Sequence[int], None]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ecg_data = datapoint.ecg_clean if use_clean else datapoint.ecg
    icg_data = datapoint.icg_clean if use_clean else datapoint.icg

    ecg_data.columns = ["ECG"]
    icg_data.columns = ["ICG ($dZ/dt$)"]

    heartbeat_subsample = _sanitize_heartbeat_subsample(heartbeat_subsample)

    if heartbeat_subsample is not None:
        heartbeat_borders = datapoint.reference_heartbeats
        start = heartbeat_borders["start_sample"].iloc[heartbeat_subsample[0]]
        end = heartbeat_borders["end_sample"].iloc[heartbeat_subsample[-1]]

        ecg_data = ecg_data.iloc[start:end]
        icg_data = icg_data.iloc[start:end]

    return ecg_data, icg_data


def _get_reference_labels(
    datapoint: BaseUnifiedPepExtractionDataset,
    heartbeat_subsample: Optional[Sequence[int]] = None,
) -> dict[str, pd.DataFrame]:
    reference_heartbeats = datapoint.reference_heartbeats
    reference_labels_ecg = datapoint.reference_labels_ecg
    reference_labels_icg = datapoint.reference_labels_icg
    q_wave_onsets = reference_labels_ecg.xs("ECG", level="channel")["sample_relative"]
    q_wave_artefacts = reference_labels_ecg.reindex(["Artefact"], level="channel")["sample_relative"]
    b_points = reference_labels_icg.xs("ICG", level="channel")["sample_relative"]
    b_point_artefacts = reference_labels_icg.reindex(["Artefact"], level="channel")["sample_relative"]

    heartbeat_subsample = _sanitize_heartbeat_subsample(heartbeat_subsample)

    if heartbeat_subsample is not None:
        reference_heartbeats = reference_heartbeats.iloc[heartbeat_subsample]
        q_wave_onsets = q_wave_onsets.reindex(heartbeat_subsample, level="heartbeat_id").dropna()
        q_wave_artefacts = q_wave_artefacts.reindex(heartbeat_subsample, level="heartbeat_id").dropna()
        b_points = b_points.reindex(heartbeat_subsample, level="heartbeat_id").dropna()
        b_point_artefacts = b_point_artefacts.reindex(heartbeat_subsample, level="heartbeat_id").dropna()

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
    heartbeat_subsample = _sanitize_heartbeat_subsample(heartbeat_subsample)
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
