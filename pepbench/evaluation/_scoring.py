from typing import Union, Sequence, Tuple

import numpy as np
import pandas as pd
from tpcp.validate import no_agg, FloatAggregator, Scorer, validate

from pepbench.datasets import BaseUnifiedPepExtractionDataset
from pepbench.evaluation import match_heartbeat_lists
from pepbench.evaluation._error_metrics import error, abs_error, abs_rel_error
from pepbench.evaluation._scoring_aggregator import SingleValueAggregator
from pepbench.pipelines._base_pipeline import _BasePepExtractionPipeline

__all__ = ["validate_pep_pipeline", "convert_validate_result_to_dataframe", "score"]


def validate_pep_pipeline(pipeline: _BasePepExtractionPipeline, dataset: BaseUnifiedPepExtractionDataset) -> dict:
    mean_std_agg = FloatAggregator(mean_and_std)
    scorer = Scorer(score, default_aggregator=mean_std_agg)
    return validate(pipeline=pipeline, dataset=dataset, scoring=scorer)


def convert_validate_result_to_dataframe(
    *, dataset: BaseUnifiedPepExtractionDataset, results: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_labels = results["data_labels"]
    subset = dataset.get_subset(group_labels=data_labels[0])

    results_subset_single = {
        key.replace("single__", ""): val[0]
        for key, val in results.items()
        if key.startswith("single__") and "per_sample" not in key
    }
    result_df_single = pd.DataFrame.from_dict(results_subset_single)
    result_df_single.index = pd.MultiIndex.from_frame(subset.index)

    results_subset_agg = {key.replace("agg__", ""): val[0] for key, val in results.items() if key.startswith("agg__")}
    results_subset_agg = {
        agg_type: {
            key.replace(f"__{agg_type}", ""): val
            for key, val in results_subset_agg.items()
            if key.endswith(f"__{agg_type}")
        }
        for agg_type in ["mean", "std"]
    }
    result_df_agg = pd.DataFrame.from_dict(results_subset_agg)

    results_subset_per_sample = {
        key.replace("single__", ""): val[0]
        for key, val in results.items()
        if key.startswith("single__") and "per_sample" in key
    }
    # concatenate the per_sample results
    pep_estimation = results_subset_per_sample.pop("pep_estimation_per_sample")
    pep_estimation = {tuple(key): test_idx for key, test_idx in zip(subset.index.to_numpy(), pep_estimation)}
    pep_estimation = pd.concat(pep_estimation)
    pep_estimation.index.names = list(subset.index.columns) + [""]
    results_subset_per_sample = {key: np.concatenate(val, axis=0) for key, val in results_subset_per_sample.items()}

    # heartbeat_ids = heartbeat_ids.set_index([("heartbeat_id", "estimated"), ("heartbeat_id", "reference")])
    result_df_per_sample = pd.DataFrame.from_dict(results_subset_per_sample)
    result_df_per_sample.columns = pd.MultiIndex.from_product([list(result_df_per_sample.columns), ["metric"]])
    result_df_per_sample.index = pep_estimation.index
    result_df_per_sample = pd.concat([pep_estimation, result_df_per_sample], axis=1)

    # result_df_agg
    return result_df_agg, result_df_single, result_df_per_sample


def score(pipeline: _BasePepExtractionPipeline, datapoint: BaseUnifiedPepExtractionDataset) -> dict:
    pipeline = pipeline.clone()
    # TODO: change to pipeline.save_run(datapoint) later
    pipeline = pipeline.run(datapoint)
    pep_estimated = pipeline.pep_results_
    pep_reference = datapoint.reference_pep

    heartbeat_matching = match_heartbeat_lists(
        heartbeats_reference=datapoint.reference_heartbeats,
        heartbeats_extracted=pipeline.heartbeat_segmentation_results_,
        tolerance_ms=20,
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

    pep_reference = pep_merged_filter[("pep_ms", "reference")].to_numpy(na_value=np.nan)
    pep_estimated = pep_merged_filter[("pep_ms", "estimated")].to_numpy(na_value=np.nan)

    pep_error = error(pep_merged_filter[("pep_ms", "reference")], pep_merged_filter[("pep_ms", "estimated")]).to_numpy(
        na_value=np.nan
    )
    pep_abs_error = abs_error(
        pep_merged_filter[("pep_ms", "reference")], pep_merged_filter[("pep_ms", "estimated")]
    ).to_numpy(na_value=np.nan)
    pep_abs_rel_error = 100 * abs_rel_error(
        pep_merged_filter[("pep_ms", "reference")], pep_merged_filter[("pep_ms", "estimated")]
    ).to_numpy(na_value=np.nan)

    single_value_aggregator = SingleValueAggregator(mean_and_std)

    return {
        "pep_reference_ms": np.nanmean(pep_reference),
        "pep_estimated_ms": np.nanmean(pep_estimated),
        "error_ms": np.nanmean(pep_error),
        "absolute_error_ms": np.nanmean(pep_abs_error),
        "absolute_relative_error_percent": np.nanmean(pep_abs_rel_error),
        "num_peps": no_agg(num_peps),
        "num_valid_peps": no_agg(num_valid_peps),
        "num_invalid_peps": no_agg(num_invalid_peps),
        "pep_estimation_per_sample": no_agg(pep_merged_filter),
        "error_per_sample_ms": single_value_aggregator(pep_error),
        "absolute_error_per_sample_ms": single_value_aggregator(pep_abs_error),
        "absolute_relative_error_per_sample_percent": single_value_aggregator(pep_abs_rel_error),
    }


def mean_and_std(vals: Sequence[float]):
    return {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}


def _merge_extracted_and_reference_pep(
    extracted: pd.DataFrame, reference: pd.DataFrame, tp_matches: Union[pd.DataFrame, None] = None
) -> pd.DataFrame:
    extracted_original = extracted.copy()
    # if wb_id in index, add it as a column to preserve it in the combined DataFrame
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
    rr_interval_extracted.columns = pd.MultiIndex.from_product([["heartbeat_id", "rr_interval_ms"], ["estimated"]])
    # set index to join on the MultiIndex
    matches = matches.set_index(("heartbeat_id", "estimated"))
    rr_interval_extracted = rr_interval_extracted.set_index(("heartbeat_id", "estimated"))
    # join the rr_intervals to the DataFrame and reset the index
    matches = matches.join(rr_interval_extracted).reset_index()

    matches = matches.reindex(
        [
            "heartbeat_id",
            "heartbeat_start_sample",
            "heartbeat_end_sample",
            "q_wave_onset_sample",
            "b_point_sample",
            "rr_interval_ms",
            "pep_sample",
            "pep_ms",
            "nan_reason",
        ],
        axis=1,
        level=0,
    )

    return matches.convert_dtypes(infer_objects=True)
