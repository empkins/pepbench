r"""Evaluation utilities for PEP extraction pipelines.

Provides a ``tpcp`` Algorithm implementation to evaluate PEP extraction pipelines on datasets.
The module exposes a namedtuple for packaging results and the ``PepEvaluationChallenge`` class
that runs the evaluation, aggregates results and saves them to disk.

Components
----------
:class:\~pepbench.evaluation._evaluation.ChallengeResults``
    Named tuple with fields ``agg_mean_std``, ``agg_total``, ``single``, ``per_sample``.
:class:`~pepbench.evaluation._evaluation.PepEvaluationChallenge`
    The main challenge class that runs and aggregates evaluations.

Notes
-----
- Uses :class:`tpcp.validate.Scorer` together with :class:`tpcp.validate.FloatAggregator`.
- Results are stored as pandas DataFrames on the challenge instance with attribute names ending
  in an underscore (``_``).
"""

import json
import warnings
from collections import namedtuple
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from tpcp import Algorithm
from tpcp.validate import FloatAggregator, Scorer, validate
from typing_extensions import Self

from pepbench.datasets import BasePepDatasetWithAnnotations
from pepbench.evaluation._scoring import mean_and_std, score_pep_evaluation
from pepbench.pipelines import BasePepExtractionPipeline
from pepbench.utils._timing import measure_time
from pepbench.utils._types import path_t

__all__ = ["ChallengeResults", "PepEvaluationChallenge"]

ChallengeResults = namedtuple("ChallengeResults", ["agg_mean_std", "agg_total", "single", "per_sample"])


# TODO add CrossValidateChallenge
class PepEvaluationChallenge(Algorithm):
    """Evaluation challenge for PEP extraction pipelines.

    This is the ``tpcp`` implementation of the evaluation challenge for PEP extraction pipelines.
    It evaluates a given PEP extraction pipeline on a dataset and produces aggregated and per-sample
    evaluation results stored as pandas DataFrames on the challenge instance.

    Parameters
    ----------
    dataset : :class:`BasePepDatasetWithAnnotations`
        The dataset to evaluate. The dataset must implement the unified interface required by the
        evaluation utilities.
    scoring : Callable, optional
        The scoring function to use for evaluation. The scoring function should accept the pipeline and a
        datapoint and return a dictionary with evaluation outputs. Default is
        :func:`pepbench.evaluation._scoring.score_pep_evaluation`.
    validate_kwargs : dict, optional
        Additional keyword arguments passed to :class:`tpcp.validate.Scorer`.

    Attributes
    ----------
    results_ : dict
        Raw results returned by :func:`tpcp.validate.validate`.
    results_agg_mean_std_ : :class:`pandas.DataFrame`
        Mean and standard deviation aggregated results.
    results_agg_total_ : :class:`pandas.DataFrame`
        Total counts aggregated results.
    results_single_ : :class:`pandas.DataFrame`
        Single (non-aggregated) results for each datapoint.
    results_per_sample_ : :class:`pandas.DataFrame`
        Per-sample flattened results.

    """

    _action_methods = "run"

    dataset: BasePepDatasetWithAnnotations
    scoring: Callable | None

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
        dataset: BasePepDatasetWithAnnotations,
        scoring: Callable = score_pep_evaluation,
        validate_kwargs: dict | None = None,
    ) -> None:
        """Initialize a new evaluation challenge.

        To initialize a new evaluation challenge, you need to provide a dataset and a scoring function. Afterwards,
        you can challenge a specific PEP extraction pipeline by passing it to the ``run`` method.

        Parameters
        ----------
        dataset : :class:`~pepbench.datasets._base_pep_extraction_dataset.BasePepDatasetWithAnnotations`
            The dataset to evaluate the pipeline on. The dataset needs to be a subclass of
            ``BaseUnifiedPepExtractionDataset``, which provides the necessary unified interface to access the data.
        scoring : Callable, optional
            The scoring function to use for the evaluation. The scoring function should take the pipeline and a
            datapoint from the dataset as input and return a dictionary with the evaluation results. The default
            scoring function is :func:``pepbench.evaluation._scoring.score_pep_evaluation``.
        validate_kwargs : dict, optional
            Additional keyword arguments to pass to the :class:``tpcp.validate.Scorer`` class.

        """
        self.dataset = dataset
        self.scoring = scoring
        self.validate_kwargs = validate_kwargs or {}

    def run(self, pipeline: BasePepExtractionPipeline) -> Self:
        """Run the evaluation challenge for a given pipeline.

        Executes validation using :func:`tpcp.validate.validate` with a :class:`tpcp.validate.Scorer`
        and aggregates timing information.

        Parameters
        ----------
        pipeline : :class:`~pepbench.pipelines._base_pipeline.BasePepExtractionPipeline`
            The PEP extraction pipeline to evaluate. The pipeline needs to be a subclass of
            :class:``pepbench.pipelines.BasePepExtractionPipeline`` and should be able to process the dataset.

        Returns
        -------
        Self
        The challenge instance with results stored as attributes (see class docstring).
        """
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

        Saves timing information as JSON and :class:`~pandas.DataFrame` results as CSV
        files using the provided filename stub.

        Parameters
        ----------
        folder_path : :class:`pathlib.Path` or str
            Folder path to save the results to.
        filename_stub : str
            Filename stub to prefix saved files.


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

        Each key in ``attr_dict`` is appended with an underscore and set as an attribute on the instance.

        Parameters
        ----------
        obj
            The object to set the attributes on.
        attr_dict
            Dictionary of attribute names (keys) and values to set on the instance.
        """
        for key, value in attr_dict.items():
            setattr(self, f"{key}_", value)

    def results_as_df(self) -> Self:
        """Convert the raw validation results to pandas DataFrames and attach them to the instance.

        The method builds the following DataFrames and stores them as instance attributes:
            * ``results_agg_mean_std_``: Mean and standard deviation of aggregated metrics.
            * ``results_agg_total_``: Total counts (e.g. total/valid/invalid PEPs).
            * ``results_single_``: Single (non-aggregated) results per datapoint.
            * ``results_per_sample_``: Per-sample flattened results with multiindex columns.

        Returns
        -------
        Self
            The challenge instance with :class:`~pandas.DataFrame` attributes populated.

        """
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
        pep_estimation = {
            tuple(key): test_idx for key, test_idx in zip(subset.index.to_numpy(), pep_estimation, strict=False)
        }
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
