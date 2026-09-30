.. _user_guide_evaluation:

Running Evaluation Challenges
=============================

pepbench provides a standardized **evaluation framework** for PEP
extraction pipelines via :class:`pepbench.evaluation.PepEvaluationChallenge`.
Each **challenge** is defined by a (pipeline, dataset) pair and yields
metrics aggregated at different levels.

Key classes and functions
-------------------------

* :class:`pepbench.evaluation.PepEvaluationChallenge` – runs evaluation
  across a dataset of annotated samples.
* :class:`pepbench.evaluation.ChallengeResults` – tuple-like container
  for aggregated and per-sample results.
* :func:`pepbench.evaluation.score_pep_evaluation` – default scoring
  function.

Initialising a challenge
------------------------

You need:

* a dataset subclassing
  :class:`pepbench.datasets.BasePepDatasetWithAnnotations`
  (e.g., :class:`EmpkinsDataset` with ``only_labeled=True``), and
* a scoring function (usually :func:`score_pep_evaluation`).

.. code-block:: python

   from pepbench.datasets import EmpkinsDataset
   from pepbench.evaluation import PepEvaluationChallenge, score_pep_evaluation

   ds = EmpkinsDataset(
       base_path="/path/to/empkins",
       only_labeled=True,
       exclude_missing_data=True,
       label_type="average",
   )

   challenge = PepEvaluationChallenge(
       dataset=ds,
       scoring=score_pep_evaluation,
   )

Running the challenge on a pipeline
-----------------------------------

.. code-block:: python

   from pepbench.pipelines import PepExtractionPipeline
   from pepbench.algorithms.heartbeat_segmentation import HeartbeatSegmentationNeurokit
   from pepbench.algorithms.ecg import QPeakExtractionVanLien2013
   from pepbench.algorithms.icg import (
       BPointExtractionLozano2007LinearRegression,
       CPointExtractionScipyFindPeaks,
   )
   from pepbench.algorithms.outlier_correction import OutlierCorrectionLinearInterpolation

   pipeline = PepExtractionPipeline(
       heartbeat_segmentation_algo=HeartbeatSegmentationNeurokit(),
       q_peak_algo=QPeakExtractionVanLien2013(),
       b_point_algo=BPointExtractionLozano2007LinearRegression(),
       c_point_algo=CPointExtractionScipyFindPeaks(),
       outlier_correction_algo=OutlierCorrectionLinearInterpolation(),
   )

   # Run the evaluation (internally loops over all datapoints)
   challenge = challenge.run(pipeline)

   # Convert internal results to DataFrames
   challenge = challenge.results_as_df()

After calling :meth:`results_as_df`, the challenge instance carries
four main result attributes:

* ``results_agg_mean_std_`` – mean and standard deviation across datapoints
* ``results_agg_total_`` – overall counts (e.g. valid vs invalid PEP)
* ``results_single_`` – one row per datapoint
* ``results_per_sample_`` – per-sample / per-beat results

Each attribute is a pandas DataFrame.

Example: inspecting per-datapoint performance
---------------------------------------------

.. code-block:: python

   single = challenge.results_single_
   print(single.head())

   # Sort by RMSE against reference PEP (column name depends on scoring)
   single_sorted = single.sort_values("rmse_pep")
   print(single_sorted[["participant", "condition", "rmse_pep"]].head())

Example: using ChallengeResults directly
----------------------------------------

If you call :func:`score_pep_evaluation` manually or in custom workflows,
it returns a :class:`ChallengeResults` object:

.. code-block:: python

   from pepbench.evaluation import score_pep_evaluation

   results: ChallengeResults = score_pep_evaluation(
       pipeline=pipeline,
       datapoint=datapoint,
   )

   agg_mean_std = results.agg_mean_std
   agg_total = results.agg_total
   per_sample = results.per_sample

Saving results to disk
----------------------

The challenge can write its results to disk:

.. code-block:: python

   challenge.save_results(
       folder_path="results/2025-01-01",
       filename_stub="lozano_qvanlien",
   )

This creates files (e.g. CSVs) with aggregated and per-sample metrics,
which is convenient for papers or further statistical analysis.

Plotting signals and results
----------------------------

pepbench provides helper plotting functions, e.g.
:func:`pepbench.plotting.plot_signals_from_challenge_results`, which can
visualize ECG/ICG signals together with algorithmic and reference PEP:

.. code-block:: python

   from pepbench.plotting import plot_signals_from_challenge_results

   datapoint = next(iter(ds))
   pep_per_sample = challenge.results_per_sample_.loc[datapoint.index_as_tuples()[0]]

   fig, axes = plot_signals_from_challenge_results(
       datapoint=datapoint,
       pep_results_per_sample=pep_per_sample,
       normalize_time=True,
       add_pep=True,
   )

   fig.suptitle("Example PEP extraction vs reference")

For more complex plotting options see the Plotting API reference.
