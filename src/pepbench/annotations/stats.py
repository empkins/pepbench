__all__ = [
    "add_annotation_agreement_to_results_dataframe",
    "bin_annotation_differences",
    "compute_icc",
    "describe_annotation_differences",
]

from collections.abc import Sequence

import pandas as pd
import pingouin as pg

from pepbench.annotations._annotations import compute_annotation_differences, normalize_annotations_to_heartbeat_start
from pepbench.data_handling import add_unique_id_to_results_dataframe


def describe_annotation_differences(annotation_diffs: pd.DataFrame, include_absolute: bool = True) -> pd.DataFrame:
    annotation_diffs_describe = annotation_diffs.copy()
    if include_absolute:
        annotation_diffs_describe = annotation_diffs_describe.assign(
            difference_ms_absolute=annotation_diffs_describe["difference_ms"].abs()
        )
    return annotation_diffs_describe.describe().T


def bin_annotation_differences(
    annotation_diffs: pd.DataFrame, bins: Sequence[int] | None = None, labels: Sequence[str] | None = None
) -> pd.DataFrame:
    if bins is None:
        bins = [0, 4, 10]
    annotation_bins = pd.cut(
        annotation_diffs.abs().squeeze(),
        bins=[*bins, annotation_diffs.max().squeeze()],
        include_lowest=True,
        labels=labels,
    )
    return annotation_bins.to_frame(name="annotation_bins")


def compute_icc(annotation_diffs: pd.DataFrame, sampling_rate_hz) -> pd.DataFrame:
    annotation_diffs_normalized = normalize_annotations_to_heartbeat_start(
        annotation_diffs, sampling_rate_hz=sampling_rate_hz
    )
    annotation_diffs_normalized = add_unique_id_to_results_dataframe(annotation_diffs_normalized, algo_levels=["rater"])

    return pg.intraclass_corr(
        data=annotation_diffs_normalized.reset_index(),
        targets="id_concat",
        ratings="difference_ms",
        raters="rater",
        nan_policy="omit",
    )


def add_annotation_agreement_to_results_dataframe(
    results_per_sample: pd.DataFrame, annotations: pd.DataFrame, sampling_rate_hz: float
) -> pd.DataFrame:
    annotation_diffs = compute_annotation_differences(annotations, sampling_rate_hz=sampling_rate_hz)
    annotation_bins = bin_annotation_differences(annotation_diffs, labels=["high", "medium", "low"])

    annotation_bins.index = annotation_bins.index.rename({"heartbeat_id": "id"})
    annotation_bins = pd.concat(
        {"Annotation Agreement": pd.concat({"annotation_agreement": annotation_bins}, axis=1)}, axis=1
    )

    results_per_sample = results_per_sample.join(annotation_bins)

    results_per_sample = results_per_sample.reindex(
        ["absolute_error_per_sample_ms", "annotation_agreement"], level=1, axis=1
    ).set_index(("Annotation Agreement", "annotation_agreement", "annotation_bins"), append=True)
    results_per_sample.index = results_per_sample.index.rename(
        {("Annotation Agreement", "annotation_agreement", "annotation_bins"): "agreement_bins"}
    )
    return results_per_sample
