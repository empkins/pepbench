from collections.abc import Sequence

import numpy as np
import pandas as pd
from tpcp.validate import FloatAggregator, no_agg

from pepbench.datasets import BasePepDatasetWithAnnotations
from pepbench.evaluation._error_metrics import abs_error, abs_rel_error, error
from pepbench.evaluation._scoring_aggregator import PerSampleAggregator
from pepbench.heartbeat_matching import match_heartbeat_lists
from pepbench.pipelines import BasePepExtractionPipeline

__all__ = ["mean_and_std", "score_pep_evaluation"]


def score_pep_evaluation(pipeline: BasePepExtractionPipeline, datapoint: BasePepDatasetWithAnnotations) -> dict:
    """Run a PEP extraction pipeline on a single datapoint and compute evaluation metrics.

    # *first* averaged over single datapoint and *then* aggregated (mean, std) on total dataset

    The following evaluation metrics are *first* averaged over a single datapoint and *then* aggregated (mean, std)
    on the total dataset:

        * ``pep_reference_ms``: The mean reference PEP in milliseconds.
        * ``pep_estimated_ms``: The mean estimated PEP in milliseconds.
        * ``error_ms``: The mean error between reference and estimated PEP in milliseconds.
        * ``absolute_error_ms``: The mean absolute error between reference and estimated PEP in milliseconds.
        * ``absolute_relative_error_percent``: The mean absolute relative error (relative to the reference PEP)
          between reference and estimated PEP in percent.


    The following evaluation metrics are passed on (since they are single values per datapoint) and then aggregated
    by summing them up (using the `sum_aggregator`):

        * ``num_pep_total``: The total number of PEPs in this datapoint.
        * ``num_pep_valid``: The number of valid PEPs in this datapoint.
        * ``num_pep_invalid``: The number of invalid PEPs in this datapoint.


    The following evaluation metrics are not aggregated but passed on as they are:

        * ``pearson_r``: The Pearson correlation coefficient between the reference and estimated PEP values of all
          matched heartbeats for a single datapoint (excluding NaN values).

    The following evaluation metrics are not aggregated but passed as per-sample values:

        * ``pep_estimation_per_sample``: The estimated and reference PEPs per sample.


    The following evaluation metrics are directly aggregated (mean, std) over *all* samples *without* intermediate
    aggregation on single datapoint:

        * ``error_per_sample_ms``: The mean error between reference and estimated PEP per sample in milliseconds.
        * ``absolute_error_per_sample_ms``: The mean absolute error between reference and estimated PEP per sample
          in milliseconds.
        * ``absolute_relative_error_per_sample_percent``: The mean absolute relative error (relative to the reference
          PEP) between reference and estimated PEP per sample in percent.


    Parameters
    ----------
    pipeline : :class:`pepbench.pipelines.BasePepExtractionPipeline`
        A PEP extraction pipeline. The pipeline must be a subclass of
        :class:`pepbench.pipelines.BasePepExtractionPipeline`.
    datapoint : :class:`pepbench.datasets.BasePepDatasetWithAnnotations`
        A single datapoint from the dataset. The datapoint must be a subclass of
        :class:`pepbench.datasets.BasePepDatasetWithAnnotations`.

    Returns
    -------
    dict
        A dictionary containing the evaluation metrics.

    """
    # not necessary anymore
    # pipeline = pipeline.clone()
    pipeline = pipeline.safe_run(datapoint)
    pep_estimated = pipeline.pep_results_
    pep_reference = datapoint.reference_pep

    heartbeat_matching = match_heartbeat_lists(
        heartbeats_reference=datapoint.reference_heartbeats,
        heartbeats_extracted=pipeline.heartbeat_segmentation_results_,
        tolerance_ms=100,
        sampling_rate_hz=datapoint.sampling_rate_ecg,
    )

    # todo: handle false negatives and false positives, i.e. heartbeats that are not matched
    tp_matches = heartbeat_matching.query("match_type == 'tp'")

    pep_extracted_tp = pep_estimated.loc[tp_matches["heartbeat_id"]]
    pep_reference_tp = pep_reference.loc[tp_matches["heartbeat_id_reference"]]

    pep_merged = _merge_extracted_and_reference_pep(pep_extracted_tp, pep_reference_tp, tp_matches)

    reference_pep_nan_idxs = pep_merged.index[pep_merged[("pep_ms", "reference")].isna()]

    # drop reference PEPs that are NaN
    pep_merged_filter = pep_merged.drop(index=reference_pep_nan_idxs)

    # determine the estimated PEPs that are NaN
    estimated_pep_nan_idxs = pep_merged_filter.index[pep_merged_filter[("pep_ms", "estimated")].isna()]
    num_peps = len(pep_merged_filter)
    num_invalid_peps = len(estimated_pep_nan_idxs)
    num_valid_peps = len(pep_merged_filter) - num_invalid_peps

    pep_reference = pep_merged_filter[("pep_ms", "reference")].to_numpy(na_value=np.nan).astype(float)
    pep_estimated = pep_merged_filter[("pep_ms", "estimated")].to_numpy(na_value=np.nan).astype(float)

    nan_mask = ~np.logical_or(np.isnan(pep_reference), np.isnan(pep_estimated))

    # compute cross-correlation coefficient
    corr_coeff = np.corrcoef(np.compress(nan_mask, pep_reference), np.compress(nan_mask, pep_estimated))
    corr_coeff = corr_coeff[0][1]

    pep_error = error(pep_merged_filter[("pep_ms", "reference")], pep_merged_filter[("pep_ms", "estimated")]).to_numpy(
        na_value=np.nan
    )
    pep_abs_error = abs_error(
        pep_merged_filter[("pep_ms", "reference")], pep_merged_filter[("pep_ms", "estimated")]
    ).to_numpy(na_value=np.nan)
    pep_abs_rel_error = 100 * abs_rel_error(
        pep_merged_filter[("pep_ms", "reference")], pep_merged_filter[("pep_ms", "estimated")]
    ).to_numpy(na_value=np.nan)

    per_sample_aggregator = PerSampleAggregator(mean_and_std)
    sum_aggregator = FloatAggregator(np.nansum)

    return {
        # *first* averaged over single datapoint and *then* aggregated (mean, std) on total dataset
        "pep_reference_ms": np.nanmean(pep_reference),
        "pep_estimated_ms": np.nanmean(pep_estimated),
        "error_ms": np.nanmean(pep_error),
        "absolute_error_ms": np.nanmean(pep_abs_error),
        "absolute_relative_error_percent": np.nanmean(pep_abs_rel_error),
        # no aggregation here necessary since
        "pearson_r": corr_coeff,
        # single values per datapoint => aggregated on total dataset by summing it up (thus the sum_aggregator)
        "num_pep_total": sum_aggregator(num_peps),
        "num_pep_valid": sum_aggregator(num_valid_peps),
        "num_pep_invalid": sum_aggregator(num_invalid_peps),
        # no aggregation, keep per-sample values
        "pep_estimation_per_sample": no_agg(pep_merged_filter),
        # direct aggregation (mean, std) over all samples *without* intermediate aggregation on single datapoint
        "error_per_sample_ms": per_sample_aggregator(pep_error),
        "absolute_error_per_sample_ms": per_sample_aggregator(pep_abs_error),
        "absolute_relative_error_per_sample_percent": per_sample_aggregator(pep_abs_rel_error),
    }


def mean_and_std(vals: Sequence[float]) -> dict:
    return {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}


def _merge_extracted_and_reference_pep(
    extracted: pd.DataFrame, reference: pd.DataFrame, tp_matches: pd.DataFrame | None = None
) -> pd.DataFrame:
    extracted_original = extracted.copy()
    # if heartbeat_id in index, add it as a column to preserve it in the combined DataFrame
    if "heartbeat_id" in extracted.index.names and "heartbeat_id" in reference.index.names:
        extracted.insert(0, "heartbeat_id", extracted.index.get_level_values("heartbeat_id"))
        reference.insert(0, "heartbeat_id", reference.index.get_level_values("heartbeat_id"))

    common_columns = list(set(reference.columns).intersection(extracted.columns))
    if len(common_columns) == 0:
        raise ValueError("No common columns found in `extracted` and `reference`.")

    extracted = extracted[common_columns]
    reference = reference[common_columns]

    if tp_matches is not None:
        extracted.index = tp_matches.index
        reference.index = tp_matches.index

    matches = extracted.merge(reference, left_index=True, right_index=True, suffixes=("_est", "_ref"))

    # construct MultiIndex columns
    matches.columns = pd.MultiIndex.from_product([["estimated", "reference"], common_columns])
    # make 'metrics' level the uppermost level and sort columns accordingly for readability
    matches = matches.swaplevel(axis=1).sort_index(axis=1, level=0)

    # add rr_intervals to the dataframe; needs to be handled separately since it's not part of the reference data
    rr_interval_extracted = extracted_original[["rr_interval_ms"]].reset_index()
    rr_interval_extracted = pd.concat([rr_interval_extracted, rr_interval_extracted], axis=1)
    rr_interval_extracted.columns = pd.MultiIndex.from_tuples(
        [
            ("heartbeat_id", "estimated"),
            ("rr_interval_ms", "estimated"),
            ("heartbeat_id", "reference"),
            ("rr_interval_ms", "reference"),
        ]
    )
    # drop ("heartbeat_id", "reference")
    rr_interval_extracted = rr_interval_extracted.drop(("heartbeat_id", "reference"), axis=1)

    # add heart_rate_bpm to the dataframe; needs to be handled separately since it's not part of the reference data
    heart_rate_extracted = extracted_original[["heart_rate_bpm"]].reset_index()
    heart_rate_extracted = pd.concat([heart_rate_extracted, heart_rate_extracted], axis=1)
    heart_rate_extracted.columns = pd.MultiIndex.from_tuples(
        [
            ("heartbeat_id", "estimated"),
            ("heart_rate_bpm", "estimated"),
            ("heartbeat_id", "reference"),
            ("heart_rate_bpm", "reference"),
        ]
    )
    # drop ("heartbeat_id", "reference")
    heart_rate_extracted = heart_rate_extracted.drop(("heartbeat_id", "reference"), axis=1)

    # set index to join on the MultiIndex
    matches = matches.set_index(("heartbeat_id", "estimated"))
    rr_interval_extracted = rr_interval_extracted.set_index(("heartbeat_id", "estimated"))
    heart_rate_extracted = heart_rate_extracted.set_index(("heartbeat_id", "estimated"))

    # join the rr_intervals to the DataFrame and reset the index
    matches = matches.join(rr_interval_extracted).join(heart_rate_extracted).reset_index()

    matches = matches.reindex(
        [
            "heartbeat_id",
            "heartbeat_start_sample",
            "heartbeat_end_sample",
            "q_peak_sample",
            "b_point_sample",
            "rr_interval_ms",
            "heart_rate_bpm",
            "pep_sample",
            "pep_ms",
            "nan_reason",
        ],
        axis=1,
        level=0,
    )

    return matches.astype(
        {
            ("heartbeat_id", "estimated"): "Int64",
            ("heartbeat_start_sample", "reference"): "Int64",
            ("heartbeat_start_sample", "estimated"): "Int64",
            ("heartbeat_end_sample", "reference"): "Int64",
            ("heartbeat_end_sample", "estimated"): "Int64",
            ("q_peak_sample", "reference"): "Int64",
            ("q_peak_sample", "estimated"): "Int64",
            ("b_point_sample", "reference"): "Int64",
            ("b_point_sample", "estimated"): "Int64",
            ("rr_interval_ms", "reference"): "Float64",
            ("rr_interval_ms", "estimated"): "Float64",
            ("heart_rate_bpm", "reference"): "Float64",
            ("heart_rate_bpm", "estimated"): "Float64",
            ("pep_sample", "reference"): "Int64",
            ("pep_sample", "estimated"): "Int64",
            ("pep_ms", "reference"): "Float64",
            ("pep_ms", "estimated"): "Float64",
            ("nan_reason", "reference"): "object",
            ("nan_reason", "estimated"): "object",
        }
    )
