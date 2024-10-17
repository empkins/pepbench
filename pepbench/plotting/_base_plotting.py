from collections.abc import Sequence
from typing import Any, Optional, Union

import pandas as pd
from biopsykit.signals._base_extraction import BaseExtraction
from biopsykit.signals.ecg.event_extraction import BaseEcgExtraction
from biopsykit.signals.icg.event_extraction import BaseBPointExtraction
from fau_colors import cmaps
from matplotlib import pyplot as plt

from pepbench.datasets import BaseUnifiedPepExtractionDataset
from pepbench.plotting._utils import (
    _add_ecg_q_wave_onset_artefacts,
    _add_ecg_q_wave_onsets,
    _add_heartbeat_borders,
    _add_icg_b_point_artefacts,
    _add_icg_b_points,
    _add_pep_from_reference,
    _add_pep_from_results,
    _get_data,
    _get_fig_ax,
    _get_fig_axs,
    _get_labels_from_challenge_results,
    _get_legend_loc,
    _get_rect,
    _get_reference_labels,
    _handle_legend_one_axis,
    _handle_legend_two_axes,
    _sanitize_heartbeat_subset,
)

__all__ = [
    "plot_signals",
    "plot_signals_with_reference_labels",
    "plot_signals_from_challenge_results",
    "plot_signals_with_reference_pep",
    "plot_signals_with_algorithm_results",
]

from pepbench.plotting.algorithms import _get_heartbeat_borders, _get_heartbeats


def plot_signals(
    datapoint: BaseUnifiedPepExtractionDataset,
    *,
    collapse: Optional[bool] = False,
    use_clean: Optional[bool] = True,
    normalize_time: Optional[bool] = False,
    heartbeat_subset: Optional[Sequence[int]] = None,
    **kwargs: Any,
) -> tuple[plt.Figure, Union[plt.Axes, Sequence[plt.Axes]]]:
    """Plot ECG and ICG signals."""
    if collapse:
        return _plot_signals_one_axis(
            datapoint=datapoint,
            use_clean=use_clean,
            normalize_time=normalize_time,
            heartbeat_subset=heartbeat_subset,
            **kwargs,
        )
    return _plot_signals_two_axes(
        datapoint=datapoint,
        use_clean=use_clean,
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        **kwargs,
    )


def plot_signals_with_reference_labels(
    datapoint: BaseUnifiedPepExtractionDataset,
    *,
    heartbeat_subset: Optional[Sequence[int]] = None,
    collapse: Optional[bool] = False,
    use_clean: Optional[bool] = True,
    normalize_time: Optional[bool] = False,
    **kwargs: Any,
) -> tuple[plt.Figure, Union[plt.Axes, Sequence[plt.Axes]]]:
    kwargs.setdefault("legend_max_cols", 6)
    kwargs.setdefault("legend_loc", _get_legend_loc(**kwargs))
    plot_artefacts = kwargs.get("plot_artefacts", False)
    rect = _get_rect(**kwargs)

    fig, ax = plot_signals(
        datapoint,
        collapse=collapse,
        use_clean=use_clean,
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        **kwargs,
    )
    ecg_data, icg_data = _get_data(
        datapoint, use_clean=use_clean, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset
    )

    reference_labels = _get_reference_labels(datapoint, heartbeat_subset=heartbeat_subset)
    reference_heartbeats = reference_labels["reference_heartbeats"]
    q_wave_onsets = reference_labels["q_wave_onsets"]
    q_wave_artefacts = reference_labels["q_wave_artefacts"]
    b_points = reference_labels["b_points"]
    b_point_artefacts = reference_labels["b_point_artefacts"]

    # plot q-wave onsets and b-points
    if collapse:
        _add_heartbeat_borders(ecg_data.index[reference_heartbeats["start_sample"]], ax, **kwargs)
        _add_ecg_q_wave_onsets(ecg_data, q_wave_onsets, ax, **kwargs)
        _add_icg_b_points(icg_data, b_points, ax, **kwargs)
        if plot_artefacts:
            if not q_wave_artefacts.empty:
                _add_ecg_q_wave_onset_artefacts(ecg_data, q_wave_artefacts, ax, **kwargs)
            if not b_point_artefacts.empty:
                _add_icg_b_point_artefacts(icg_data, b_point_artefacts, ax, **kwargs)

        _handle_legend_one_axis(fig, ax, **kwargs)
    else:
        _add_heartbeat_borders(ecg_data.index[reference_heartbeats["start_sample"]], ax[0], **kwargs)
        _add_heartbeat_borders(ecg_data.index[reference_heartbeats["start_sample"]], ax[1], **kwargs)
        _add_ecg_q_wave_onsets(ecg_data, q_wave_onsets, ax[0], **kwargs)
        _add_icg_b_points(icg_data, b_points, ax[1], **kwargs)
        if plot_artefacts:
            if not q_wave_artefacts.empty:
                _add_ecg_q_wave_onset_artefacts(ecg_data, q_wave_artefacts, ax[0], **kwargs)
            if not b_point_artefacts.empty:
                _add_icg_b_point_artefacts(icg_data, b_point_artefacts, ax[1], **kwargs)

        _handle_legend_two_axes(fig, ax, **kwargs)

    fig.tight_layout(rect=rect)
    return fig, ax


def plot_signals_with_reference_pep(
    datapoint: BaseUnifiedPepExtractionDataset,
    *,
    use_clean: Optional[bool] = True,
    normalize_time: Optional[bool] = False,
    heartbeat_subset: Optional[Sequence[int]] = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    kwargs.setdefault("legend_orientation", "vertical")
    kwargs.setdefault("legend_outside", False)
    kwargs.setdefault("legend_max_cols", 5)
    kwargs.setdefault("legend_loc", _get_legend_loc(**kwargs))
    rect = _get_rect(**kwargs)

    fig, ax = plot_signals_with_reference_labels(
        datapoint,
        use_clean=use_clean,
        collapse=True,
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        **kwargs,
    )

    ecg_data, icg_data = _get_data(
        datapoint, use_clean=use_clean, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset
    )

    reference_labels = _get_reference_labels(datapoint, heartbeat_subset=heartbeat_subset)
    reference_labels_ecg = (
        pd.concat([reference_labels["q_wave_onsets"], reference_labels["q_wave_artefacts"]]).sort_index().reset_index()
    )
    reference_labels_icg = (
        pd.concat([reference_labels["b_points"], reference_labels["b_point_artefacts"]]).sort_index().reset_index()
    )
    reference_labels_combined = pd.concat({"ecg": reference_labels_ecg, "icg": reference_labels_icg}, axis=1)

    _add_pep_from_reference(ecg_data, icg_data, reference_labels_combined, ax, **kwargs)

    _handle_legend_one_axis(fig, ax, **kwargs)
    fig.tight_layout(rect=rect)
    return fig, ax


def plot_signals_with_algorithm_results(
    datapoint: BaseUnifiedPepExtractionDataset,
    *,
    collapse: Optional[bool] = False,
    algorithm: BaseExtraction,
    use_clean: Optional[bool] = True,
    normalize_time: Optional[bool] = False,
    heartbeat_subset: Optional[Sequence[int]] = None,
    **kwargs: Any,
):
    kwargs.setdefault("legend_loc", _get_legend_loc(**kwargs))
    kwargs.setdefault("legend_max_cols", 5)
    rect = _get_rect(**kwargs)

    fig, axs = plot_signals_with_reference_labels(
        datapoint,
        collapse=collapse,
        use_clean=use_clean,
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        **kwargs,
    )

    ecg_data, icg_data = _get_data(
        datapoint, use_clean=use_clean, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset
    )
    heartbeat_subset = _sanitize_heartbeat_subset(heartbeat_subset)
    heartbeats = datapoint.heartbeats.loc[heartbeat_subset]["start_sample"]

    algorithm_results = algorithm.points_

    if isinstance(algorithm, BaseEcgExtraction):
        q_waves = algorithm_results["q_wave_sample"]
        q_waves = q_waves.loc[heartbeat_subset]
        q_waves = q_waves - heartbeats.iloc[0]
        if collapse:
            _add_ecg_q_wave_onsets(
                ecg_data,
                q_waves,
                axs,
                q_wave_label="Detected Q Waves",
                q_wave_color=cmaps.med_dark[0],
                **kwargs,
            )
        else:
            _add_ecg_q_wave_onsets(
                ecg_data,
                q_waves,
                axs[0],
                q_wave_label="Detected Q Waves",
                q_wave_color=cmaps.med_dark[0],
                **kwargs,
            )
    if isinstance(algorithm, BaseBPointExtraction):
        b_points = algorithm_results["b_point_sample"]
        b_points = b_points.loc[heartbeat_subset]
        b_points = b_points - heartbeats.iloc[0]
        if collapse:
            _add_icg_b_points(
                icg_data,
                b_points,
                axs,
                b_point_label="Detected B Points",
                b_point_color=cmaps.phil_dark[0],
                **kwargs,
            )
        else:
            _add_icg_b_points(
                icg_data,
                b_points,
                axs[1],
                b_point_label="Detected B Points",
                b_point_color=cmaps.phil_dark[0],
                **kwargs,
            )

    if collapse:
        _handle_legend_one_axis(fig, axs, **kwargs)
    else:
        _handle_legend_two_axes(fig, axs, **kwargs)

    fig.tight_layout(rect=rect)
    return fig, axs


def plot_signals_from_challenge_results(
    datapoint: BaseUnifiedPepExtractionDataset,
    pep_results_per_sample: pd.DataFrame,
    *,
    collapse: Optional[bool] = False,
    use_clean: Optional[bool] = True,
    normalize_time: Optional[bool] = False,
    heartbeat_subset: Optional[Sequence[int]] = None,
    add_pep: Optional[bool] = False,
    **kwargs: Any,
) -> tuple[plt.Figure, Sequence[plt.Axes]]:
    kwargs.setdefault("legend_loc", _get_legend_loc(**kwargs))
    kwargs.setdefault("legend_max_cols", 5)
    rect = _get_rect(**kwargs)

    fig, axs = plot_signals(
        datapoint,
        use_clean=use_clean,
        normalize_time=normalize_time,
        heartbeat_subset=heartbeat_subset,
        collapse=collapse,
        **kwargs,
    )

    ecg_data, icg_data = _get_data(
        datapoint, use_clean=use_clean, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset
    )

    labels_from_challenge = _get_labels_from_challenge_results(pep_results_per_sample, heartbeat_subset)

    heartbeats_start = ecg_data.index[labels_from_challenge["heartbeats_start"]]
    heartbeats_end = ecg_data.index[labels_from_challenge["heartbeats_end"] - 1]
    q_wave_labels_reference = labels_from_challenge["q_wave_labels_reference"]
    q_wave_labels_estimated = labels_from_challenge["q_wave_labels_estimated"]
    b_point_labels_reference = labels_from_challenge["b_point_labels_reference"]
    b_point_labels_estimated = labels_from_challenge["b_point_labels_estimated"]

    labels_reference = pd.concat({"ecg": q_wave_labels_reference, "icg": b_point_labels_reference}, axis=1)
    labels_estimated = pd.concat({"ecg": q_wave_labels_estimated, "icg": b_point_labels_estimated}, axis=1)

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
        ax_ecg,
        q_wave_color=cmaps.med[0],
        q_wave_label="Q-Wave Reference",
        plot_artifacts=False,
    )
    _add_ecg_q_wave_onsets(
        ecg_data,
        q_wave_labels_estimated,
        ax_ecg,
        q_wave_color=cmaps.med_dark[0],
        q_wave_label="Q-Wave Estimated",
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
    datapoint: Optional[BaseUnifiedPepExtractionDataset] = None,
    df: Optional[pd.DataFrame] = None,
    use_clean: Optional[bool] = True,
    normalize_time: Optional[bool] = False,
    heartbeat_subset: Optional[Sequence[int]] = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    kwargs.setdefault("legend_loc", _get_legend_loc(**kwargs))
    kwargs.setdefault("legend_max_cols", 5)
    plot_ecg = kwargs.get("plot_ecg", True)
    plot_icg = kwargs.get("plot_icg", True)
    color = kwargs.get("color", cmaps.fau[0])

    if datapoint is not None and df is not None:
        raise ValueError("Either `datapoint` or `df` must be provided, but not both.")
    if datapoint is None and df is None:
        raise ValueError("Either `datapoint` or `df` must be provided.")

    rect = _get_rect(**kwargs)

    fig, ax = _get_fig_ax(**kwargs)
    kwargs.pop("ax", None)

    if datapoint is not None:
        ecg_data, icg_data = _get_data(
            datapoint, use_clean=use_clean, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset
        )

        if plot_ecg:
            ecg_data.plot(ax=ax)
        if plot_icg:
            icg_data.plot(ax=ax)
    else:
        df.plot(ax=ax, color=color)

    if normalize_time:
        ax.set_xlabel("Time [s]")
    else:
        ax.set_xlabel("Time [hh:mm:ss]")
    ax.set_ylabel("Amplitude [a.u.]")

    _handle_legend_one_axis(fig, ax, **kwargs)

    fig.tight_layout(rect=rect)

    return fig, ax


def _plot_signals_two_axes(
    *,
    datapoint: BaseUnifiedPepExtractionDataset,
    use_clean: Optional[bool] = True,
    normalize_time: Optional[bool] = False,
    heartbeat_subset: Optional[Sequence[int]] = None,
    **kwargs: Any,
) -> tuple[plt.Figure, Sequence[plt.Axes]]:
    figsize = kwargs.pop("figsize", None)
    kwargs.setdefault("legend_loc", _get_legend_loc(**kwargs))
    kwargs.setdefault("legend_max_cols", 5)

    rect = kwargs.get("rect", _get_rect(**kwargs))

    fig, axs = _get_fig_axs(figsize=figsize, nrows=2, sharex=True)

    colors = iter(cmaps.faculties)

    ecg_data, icg_data = _get_data(
        datapoint, use_clean=use_clean, normalize_time=normalize_time, heartbeat_subset=heartbeat_subset
    )
    ecg_data.plot(ax=axs[0], color=next(colors), title="Electrocardiogram (ECG)")
    icg_data.plot(ax=axs[1], color=next(colors), title="Impedance Cardiogram (ICG)")

    _handle_legend_two_axes(fig, axs, **kwargs)

    fig.align_ylabels()
    fig.tight_layout(rect=rect)

    for ax in axs:
        if normalize_time:
            ax.set_xlabel("Time [s]")
        else:
            ax.set_xlabel("Time [hh:mm:ss]")
        ax.set_ylabel("Amplitude [a.u.]")

    return fig, axs
