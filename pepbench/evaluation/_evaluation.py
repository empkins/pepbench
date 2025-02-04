import json
import warnings
from collections import namedtuple
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from tpcp import Algorithm
from tpcp.validate import FloatAggregator, Scorer, validate
from typing_extensions import Self

from pepbench.datasets import BaseUnifiedPepExtractionDataset
from pepbench.evaluation._scoring import mean_and_std, score_pep_evaluation
from pepbench.pipelines import BasePepExtractionPipeline
from pepbench.utils._timing import measure_time
from pepbench.utils._types import path_t

__all__ = ["PepEvaluationChallenge", "ChallengeResults"]

ChallengeResults = namedtuple("ChallengeResults", ["agg_mean_std", "agg_total", "single", "per_sample"])


# TODO add CrossValidateChallenge
class PepEvaluationChallenge(Algorithm):
    _action_methods = "run"

    dataset: BaseUnifiedPepExtractionDataset
    scoring: Optional[Callable]

    results_: dict
    results_agg_mean_std_: pd.DataFrame
    results_agg_total_: pd.DataFrame
    results_single_: pd.DataFrame
    results_per_sample_: pd.DataFrame

    timing_information_: dict
    # timing information
    start_time_utc_timestamp_: float
    start_time_utc: str
    end_time_utc_timestamp_: float
    end_time_: str
    runtime_s_: float

    def __init__(
        self,
        *,
        dataset: BaseUnifiedPepExtractionDataset,
        scoring: Optional[Callable] = score_pep_evaluation,
        validate_kwargs: Optional[dict] = None,
    ) -> None:
        self.dataset = dataset
        self.scoring = scoring
        self.validate_kwargs = validate_kwargs or {}

    def run(self, pipeline: BasePepExtractionPipeline) -> Self:
        with measure_time() as timing_results, warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # better specify the RuntimeWarning using regex
            mean_std_agg = FloatAggregator(mean_and_std)
            scorer = Scorer(score_pep_evaluation, default_aggregator=mean_std_agg, **self.validate_kwargs)
            self.results_ = validate(pipeline=pipeline, dataset=self.dataset, scoring=scorer)
            self.results_as_df()

        self.timing_information_ = timing_results
        self._set_attrs_from_dict(timing_results)
        return self

    def save_results(self, folder_path: path_t, filename_stub: str) -> None:
        """Save the results of the evaluation to disk.

        Parameters
        ----------
        folder_path
            The folder path to save the results to.
        filename_stub
            The filename stub to use for the results file.

        """
        timing_information_path = folder_path.joinpath(f"{filename_stub}_timing_information.json")
        # save timing information as json
        with timing_information_path.open("w") as fp:
            json.dump(self.timing_information_, fp)

        # save agg results as csv
        results_agg_mean_std_path = folder_path.joinpath(f"{filename_stub}_results_agg_mean_std.csv")
        self.results_agg_mean_std_.to_csv(results_agg_mean_std_path)

        results_agg_total_path = folder_path.joinpath(f"{filename_stub}_results_agg_total.csv")
        self.results_agg_total_.to_csv(results_agg_total_path)

        # save single results as csv
        results_single_path = folder_path.joinpath(f"{filename_stub}_results_single.csv")
        self.results_single_.to_csv(results_single_path)

        # save per sample results as csv
        results_per_sample_path = folder_path.joinpath(f"{filename_stub}_results_per-sample.csv")
        self.results_per_sample_.to_csv(results_per_sample_path)

    def _set_attrs_from_dict(self, attr_dict: dict[str, Any]) -> None:
        """Set attributes of an object from a dictionary.

        Parameters
        ----------
        obj
            The object to set the attributes on.
        attr_dict
            The dictionary with the attributes to set.
        """
        for key, value in attr_dict.items():
            setattr(self, f"{key}_", value)

    def results_as_df(self) -> Self:
        results = self.results_.copy()

        data_labels = results["data_labels"]
        subset = self.dataset.get_subset(group_labels=data_labels[0])

        results_subset_single = {
            key.replace("single__", ""): val[0]
            for key, val in self.results_.items()
            if key.startswith("single__") and "per_sample" not in key
        }
        result_df_single = pd.DataFrame.from_dict(results_subset_single)
        result_df_single.index = pd.MultiIndex.from_frame(subset.index)

        results_subset_agg = {
            key.replace("agg__", ""): val[0] for key, val in results.items() if key.startswith("agg__")
        }
        # mean and std aggregations
        results_subset_agg_mean_std = {
            agg_type: {
                key.replace(f"__{agg_type}", ""): val
                for key, val in results_subset_agg.items()
                if key.endswith(f"__{agg_type}")
            }
            for agg_type in ["mean", "std"]
        }
        # add "total" aggregation
        results_subset_agg_total = {"total": {key: val for key, val in results_subset_agg.items() if "num_" in key}}

        result_df_agg_mean_std = pd.DataFrame.from_dict(results_subset_agg_mean_std)
        result_df_agg_mean_std.index.name = "metrics"

        result_df_agg_total = pd.DataFrame.from_dict(results_subset_agg_total)
        result_df_agg_total = result_df_agg_total.astype(int)
        result_df_agg_total = result_df_agg_total.reindex(["num_pep_total", "num_pep_valid", "num_pep_invalid"])
        result_df_agg_total.index.name = "metrics"

        results_subset_per_sample = {
            key.replace("single__", ""): val[0]
            for key, val in results.items()
            if key.startswith("single__") and "per_sample" in key
        }
        # concatenate the per_sample results
        pep_estimation = results_subset_per_sample.pop("pep_estimation_per_sample")
        pep_estimation = {tuple(key): test_idx for key, test_idx in zip(subset.index.to_numpy(), pep_estimation)}
        pep_estimation = pd.concat(pep_estimation)
        pep_estimation.index.names = [*list(subset.index.columns), ""]
        results_subset_per_sample = {key: np.concatenate(val, axis=0) for key, val in results_subset_per_sample.items()}

        result_df_per_sample = pd.DataFrame.from_dict(results_subset_per_sample)
        result_df_per_sample.columns = pd.MultiIndex.from_product([list(result_df_per_sample.columns), ["metric"]])
        result_df_per_sample.index = pep_estimation.index
        result_df_per_sample = pd.concat([pep_estimation, result_df_per_sample], axis=1)

        self._set_attrs_from_dict(
            {
                "results_agg_mean_std": result_df_agg_mean_std,
                "results_agg_total": result_df_agg_total,
                "results_single": result_df_single,
                "results_per_sample": result_df_per_sample,
            }
        )
        return self
