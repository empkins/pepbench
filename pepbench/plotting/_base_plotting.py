from collections.abc import Sequence
from typing import Any, Optional, Union

import pandas as pd
from fau_colors import cmaps
from matplotlib import pyplot as plt

from pepbench.datasets import BaseUnifiedPepExtractionDataset
from pepbench.plotting._utils import _get_fig_ax, _get_fig_axs

__all__ = [
    "plot_signals",
    "plot_signals_with_reference_labels",
    "plot_signals_from_challenge_result",
    "plot_signals_with_reference_pep",
]


def plot_signals(
    datapoint: BaseUnifiedPepExtractionDataset,
    *,
    collapse: Optional[bool] = False,
    use_clean: Optional[bool] = True,
    **kwargs: Any,
) -> tuple[plt.Figure, Union[plt.Axes, Sequence[plt.Axes]]]:
    """Plot ECG and ICG signals."""
    if collapse:
        return _plot_signals_one_axis(datapoint, use_clean=use_clean, **kwargs)
    return _plot_signals_two_axes(datapoint, use_clean=use_clean, **kwargs)


def plot_signals_with_reference_labels(
    datapoint: BaseUnifiedPepExtractionDataset,
    *,
    collapse: Optional[bool] = False,
    use_clean: Optional[bool] = True,
    **kwargs: Any,
) -> tuple[plt.Figure, Union[plt.Axes, Sequence[plt.Axes]]]:

    legend_orientation = kwargs.get("legend_orientation", "vertical")
    legend_outside = kwargs.get("legend_outside", False)
    legend_loc = _get_legend_loc(**kwargs)
    rect = _get_rect(**kwargs)

    fig, ax = plot_signals(datapoint, collapse=collapse, use_clean=use_clean, **kwargs)
    ecg_data, icg_data = _get_data(datapoint, use_clean)

    reference_heartbeats = datapoint.reference_heartbeats
    reference_labels_ecg = datapoint.reference_labels_ecg
    reference_labels_icg = datapoint.reference_labels_icg
    q_wave_onsets = reference_labels_ecg.xs("ECG", level="channel")["sample_relative"]
    q_wave_artefacts = reference_labels_ecg.reindex(["Artefact"], level="channel")["sample_relative"]
    b_points = reference_labels_icg.xs("ICG", level="channel")["sample_relative"]
    b_point_artefacts = reference_labels_icg.reindex(["Artefact"], level="channel")["sample_relative"]

    # plot q-wave onsets and b-points
    if collapse:
        _plot_heartbeat_borders(ecg_data.index[reference_heartbeats["start_sample"]], ax, **kwargs)
        _plot_ecg_q_wave_onsets(ecg_data, q_wave_onsets, q_wave_artefacts, ax, **kwargs)
        _plot_icg_b_points(icg_data, b_points, b_point_artefacts, ax, **kwargs)
        _handle_legend_one_axis(legend_orientation, legend_outside, legend_loc, fig, ax)
    else:
        _plot_heartbeat_borders(ecg_data.index[reference_heartbeats["start_sample"]], ax[0], **kwargs)
        _plot_heartbeat_borders(ecg_data.index[reference_heartbeats["start_sample"]], ax[1], **kwargs)
        _plot_ecg_q_wave_onsets(ecg_data, q_wave_onsets, q_wave_artefacts, ax[0], **kwargs)
        _plot_icg_b_points(icg_data, b_points, b_point_artefacts, ax[1], **kwargs)
        _handle_legend_two_axes(legend_orientation, legend_outside, legend_loc, fig, ax)

    fig.tight_layout(rect=rect)
    return fig, ax


def plot_signals_with_reference_pep(
    datapoint: BaseUnifiedPepExtractionDataset,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    legend_orientation = kwargs.get("legend_orientation", "vertical")
    legend_outside = kwargs.get("legend_outside", False)
    legend_loc = _get_legend_loc(**kwargs)

    fig, ax = plot_signals_with_reference_labels(datapoint, use_clean=True, collapse=True, **kwargs)

    color = kwargs.get("pep_color", cmaps.nat[0])
    color_artefact = kwargs.get("pep_artefact_color", cmaps.wiso[0])

    ecg_data, icg_data = _get_data(datapoint, use_clean=True)
    reference_labels_ecg = datapoint.reference_labels_ecg
    reference_labels_icg = datapoint.reference_labels_icg
    reference_labels_ecg = reference_labels_ecg.drop("heartbeat", level="channel")["sample_relative"].reset_index()
    reference_labels_icg = reference_labels_icg.drop("heartbeat", level="channel")["sample_relative"].reset_index()
    labels = pd.concat({"ecg": reference_labels_ecg, "icg": reference_labels_icg}, axis=1)

    for _i, row in labels.iterrows():
        start = ecg_data.index[row[("ecg", "sample_relative")]]
        end = icg_data.index[row[("icg", "sample_relative")]]
        if row[("ecg", "label")] == "Artefact" or row[("icg", "label")] == "Artefact":
            ax.axvspan(start, end, color=color_artefact, alpha=0.3, zorder=0, label="PEP Artefact")
        else:
            ax.axvspan(start, end, color=color, alpha=0.3, zorder=0, label="PEP")

    _handle_legend_one_axis(legend_orientation, legend_outside, legend_loc, fig, ax)
    return fig, ax


def _plot_signals_one_axis(
    datapoint: BaseUnifiedPepExtractionDataset, *, use_clean: Optional[bool] = True, **kwargs: Any
) -> tuple[plt.Figure, plt.Axes]:
    figsize = kwargs.pop("figsize", None)
    legend_outside = kwargs.get("legend_outside", False)
    legend_orientation = kwargs.get("legend_orientation", "vertical")

    legend_loc = _get_legend_loc(**kwargs)
    rect = _get_rect(**kwargs)

    fig, ax = _get_fig_ax(figsize=figsize)

    ecg_data, icg_data = _get_data(datapoint, use_clean)
    ecg_data.plot(ax=ax)
    icg_data.plot(ax=ax)

    ax.set_xlabel("Time [hh:mm:ss]")
    ax.set_ylabel("Amplitude [a.u.]")

    _handle_legend_one_axis(legend_orientation, legend_outside, legend_loc, fig, ax)

    fig.tight_layout(rect=rect)

    return fig, ax


def _plot_signals_two_axes(
    datapoint: BaseUnifiedPepExtractionDataset, *, use_clean: Optional[bool] = True, **kwargs: Any
) -> tuple[plt.Figure, Sequence[plt.Axes]]:
    figsize = kwargs.pop("figsize", None)
    legend_orientation = kwargs.get("legend_orientation", "vertical")
    legend_outside = kwargs.get("legend_outside", False)
    legend_loc = kwargs.get("legend_loc", None)
    rect = kwargs.get("rect", None)

    if legend_loc is None:
        legend_loc = _get_legend_loc(**kwargs)
    if rect is None:
        rect = _get_rect(**kwargs)

    fig, axs = _get_fig_axs(figsize=figsize, nrows=2, sharex=True)

    colors = iter(cmaps.faculties)

    ecg_data, icg_data = _get_data(datapoint, use_clean)
    ecg_data.plot(ax=axs[0], color=next(colors), title="Electrocardiogram (ECG)")
    icg_data.plot(ax=axs[1], color=next(colors), title="Impedance Cardiogram (ICG)")

    _handle_legend_two_axes(legend_orientation, legend_outside, legend_loc, fig, axs)

    fig.align_ylabels()
    fig.tight_layout(rect=rect)

    return fig, axs


def _plot_heartbeat_borders(heartbeats: pd.DataFrame, ax: plt.Axes, **kwargs: Any) -> None:
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


def _plot_ecg_q_wave_onsets(
    ecg_data: pd.DataFrame,
    q_wave_onsets: pd.DataFrame,
    q_wave_artefacts: pd.DataFrame,
    ax: plt.Axes,
    **kwargs: Any,
) -> None:
    color = kwargs.get("q_wave_color", cmaps.med[0])
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
        label="Q-Wave Onsets",
        zorder=3,
    )
    if not q_wave_artefacts.empty:
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


def _plot_icg_b_points(
    icg_data: pd.DataFrame,
    b_points: pd.DataFrame,
    b_point_artefacts: pd.DataFrame,
    ax: plt.Axes,
    **kwargs: Any,
) -> None:
    color = kwargs.get("b_point_color", cmaps.phil[0])

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
        label="B Points",
        zorder=3,
    )
    if not b_point_artefacts.empty:
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
    legend_orientation: str, legend_outside: bool, legend_loc: str, fig: plt.Figure, ax: plt.Axes
) -> None:
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = _remove_duplicate_legend_entries(handles, labels)
    ncols = min(len(handles), 6) if legend_orientation == "horizontal" else 1

    if len(fig.legends) > 0:
        fig.legends[0].remove()

    if legend_outside:
        ax.legend().remove()
        fig.legend(handles=handles, labels=labels, ncols=ncols, loc=legend_loc)
    else:
        ax.legend(loc=legend_loc, ncols=ncols)


def _handle_legend_two_axes(
    legend_orientation: str, legend_outside: bool, legend_loc: str, fig: plt.Figure, axs: Sequence[plt.Axes]
) -> None:
    handles, labels = axs[0].get_legend_handles_labels()
    handles += axs[1].get_legend_handles_labels()[0]
    labels += axs[1].get_legend_handles_labels()[1]

    handles, labels = _remove_duplicate_legend_entries(handles, labels)
    ncols = len(handles) if legend_orientation == "horizontal" else 1
    if len(fig.legends) > 0:
        fig.legends[0].remove()
    if legend_outside:
        for ax in axs:
            ax.legend().remove()
        fig.legend(handles, labels, ncols=ncols, loc=legend_loc)
    else:
        for ax in axs:
            ax.legend(loc=legend_loc, ncols=ncols)

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


def plot_signals_from_challenge_result(
    datapoint: BaseUnifiedPepExtractionDataset,
    pep_results_per_sample: pd.DataFrame,
    use_clean: Optional[bool] = True,
    **kwargs: Any,
) -> tuple[plt.Figure, Sequence[plt.Axes]]:
    # legend_orientation = kwargs.get("legend_orientation", "vertical")
    # legend_loc = kwargs.get("legend_loc", "lower right" if legend_orientation == "vertical" else "upper center")
    # rect = kwargs.pop("rect", (0, 0, 0.825, 1.0) if legend_orientation == "vertical" else (0, 0, 1, 0.925))

    fig, axs = plot_signals(datapoint, use_clean=use_clean, **kwargs)

    ecg_data, icg_data = _get_data(datapoint, use_clean)

    fig, ax = plt.subplots()

    start_end_borders = pep_results_per_sample[
        [("heartbeat_start_sample", "estimated"), ("heartbeat_end_sample", "estimated")]
    ]
    q_wave_labels_estimated = pep_results_per_sample[("q_wave_onset_sample", "estimated")]
    b_point_labels_reference = pep_results_per_sample[("b_point_sample", "reference")]
    b_point_labels_estimated = pep_results_per_sample[("b_point_sample", "estimated")]

    ecg_data.plot(ax=ax)
    icg_data.plot(ax=ax)

    ax.vlines(
        x=icg_data.index[start_end_borders[("heartbeat_start_sample", "estimated")]],
        ymin=0,
        ymax=1,
        colors=cmaps.tech[2],
        transform=ax.get_xaxis_transform(),
    )
    ax.vlines(
        x=ecg_data.index[q_wave_labels_estimated],
        ymin=0,
        ymax=1,
        color=cmaps.phil[0],
        transform=ax.get_xaxis_transform(),
        zorder=3,
        label="Q-Wave Estimated",
        ls="--",
        lw=1,
    )
    ax.vlines(
        x=icg_data.index[b_point_labels_reference],
        ymin=0,
        ymax=1,
        color=cmaps.nat[0],
        transform=ax.get_xaxis_transform(),
        zorder=3,
        label="B-Point Reference",
        ls="--",
        lw=1,
    )
    ax.vlines(
        x=icg_data.index[b_point_labels_estimated],
        ymin=0,
        ymax=1,
        color=cmaps.med[0],
        transform=ax.get_xaxis_transform(),
        zorder=3,
        label="B-Point Estimated",
        ls="--",
        lw=1,
    )

    ax.scatter(
        x=ecg_data.index[q_wave_labels_estimated],
        y=ecg_data["ecg"][ecg_data.index[q_wave_labels_estimated]],
        color=cmaps.phil[1],
        zorder=3,
    )
    ax.scatter(
        x=icg_data.index[b_point_labels_reference],
        y=icg_data["icg_der"][icg_data.index[b_point_labels_reference]],
        color=cmaps.nat[1],
        zorder=3,
    )
    ax.scatter(
        x=icg_data.index[b_point_labels_estimated],
        y=icg_data["icg_der"][icg_data.index[b_point_labels_estimated]],
        color=cmaps.med[1],
        zorder=3,
    )

    ax.legend(loc="upper right")

    fig.tight_layout()

    return fig, axs


def _get_data(datapoint: BaseUnifiedPepExtractionDataset, use_clean: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    ecg_data = datapoint.ecg_clean if use_clean else datapoint.ecg
    icg_data = datapoint.icg_clean if use_clean else datapoint.icg

    ecg_data.columns = ["ECG"]
    icg_data.columns = ["ICG ($dZ/dt$)"]

    return ecg_data, icg_data


def _get_rect(**kwargs: Any) -> tuple[float, ...]:
    rect = kwargs.pop("rect", None)
    legend_outside = kwargs.get("legend_outside", False)
    legend_orientation = kwargs.get("legend_orientation", "vertical")

    if rect is not None:
        return rect
    if legend_outside:
        return (0, 0, 0.85, 1.0) if legend_orientation == "vertical" else (0, 0, 1, 0.925)
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
