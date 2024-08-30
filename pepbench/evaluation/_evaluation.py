import warnings
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

__all__ = ["PepEvaluationChallenge"]


class PepEvaluationChallenge(Algorithm):

    _action_methods = "run"

    dataset: BaseUnifiedPepExtractionDataset
    scoring: Optional[Callable]

    results_: dict
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
        with measure_time() as timing_results:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_std_agg = FloatAggregator(mean_and_std)
                scorer = Scorer(score_pep_evaluation, default_aggregator=mean_std_agg, **self.validate_kwargs)
                self.results_ = validate(pipeline=pipeline, dataset=self.dataset, scoring=scorer)
                self.results_as_df()

        self._set_attrs_from_dict(timing_results)
        return self

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
        pep_estimation.index.names = [*list(subset.index.columns), ""]
        results_subset_per_sample = {key: np.concatenate(val, axis=0) for key, val in results_subset_per_sample.items()}

        # heartbeat_ids = heartbeat_ids.set_index([("heartbeat_id", "estimated"), ("heartbeat_id", "reference")])
        result_df_per_sample = pd.DataFrame.from_dict(results_subset_per_sample)
        result_df_per_sample.columns = pd.MultiIndex.from_product([list(result_df_per_sample.columns), ["metric"]])
        result_df_per_sample.index = pep_estimation.index
        result_df_per_sample = pd.concat([pep_estimation, result_df_per_sample], axis=1)

        self._set_attrs_from_dict(
            {
                "results_agg": result_df_agg,
                "results_single": result_df_single,
                "results_per_sample": result_df_per_sample,
            }
        )
        return self
