.. _user_guide_pipelines:

Building and Customizing PEP Extraction Pipelines
=================================================

A **PEP extraction pipeline** in pepbench is a
:tpcp:`tpcp <tpcp>` pipeline that chains heartbeat segmentation, Q-peak
detection, C- and B-point detection, optional outlier correction, and
finally PEP computation.

The main class is :class:`pepbench.pipelines.PepExtractionPipeline`.

Conceptual structure
--------------------

A pipeline is configured by *choosing algorithms* for each step:

* ``heartbeat_segmentation_algo`` – ECG heartbeat boundaries
* ``q_peak_algo`` – Q-peaks on ECG
* ``b_point_algo`` – B-points on ICG
* ``c_point_algo`` – C-points on ICG (optional but required for some B algorithms)
* ``outlier_correction_algo`` – optional B-point post-processing

The pipeline then provides:

* methods: :meth:`run`, :meth:`safe_run`
* result attributes: ``heartbeat_segmentation_results_``, ``q_peak_results_``,
  ``c_point_results_``, ``b_point_results_``,
  ``b_point_after_outlier_correction_results_``, ``pep_results_``

A minimal pipeline
------------------

.. code-block:: python

   from pepbench.algorithms.heartbeat_segmentation import HeartbeatSegmentationNeurokit
   from pepbench.algorithms.ecg import QPeakExtractionVanLien2013
   from pepbench.algorithms.icg import (
       BPointExtractionLozano2007LinearRegression,
       CPointExtractionScipyFindPeaks,
   )
   from pepbench.algorithms.outlier_correction import OutlierCorrectionLinearInterpolation
   from pepbench.pipelines import PepExtractionPipeline
   from pepbench.datasets import EmpkinsDataset

   # 1. Load a dataset
   ds = EmpkinsDataset(
       base_path="/path/to/empkins",
       only_labeled=True,
       exclude_missing_data=True,
   )

   datapoint = next(iter(ds))  # single datapoint

   # 2. Configure algorithms
   heartbeat_algo = HeartbeatSegmentationNeurokit()
   q_algo = QPeakExtractionVanLien2013(time_interval_ms=40)
   c_algo = CPointExtractionScipyFindPeaks()
   b_algo = BPointExtractionLozano2007LinearRegression()
   outlier_algo = OutlierCorrectionLinearInterpolation()

   # 3. Build the pipeline
   pipeline = PepExtractionPipeline(
       heartbeat_segmentation_algo=heartbeat_algo,
       q_peak_algo=q_algo,
       b_point_algo=b_algo,
       c_point_algo=c_algo,
       outlier_correction_algo=outlier_algo,
       handle_negative_pep="nan",
       handle_missing_events="warn",
   )

   # 4. Run on a single datapoint
   pipeline = pipeline.safe_run(datapoint)

   pep_df = pipeline.pep_results_
   print(pep_df.head())

Why ``safe_run``?
-----------------

:meth:`PepExtractionPipeline.safe_run` wraps :meth:`run` with additional
sanity checks:

* verifies that ``run`` returns ``self``
* checks that result attributes are set and follow the ``*_`` naming convention
* checks that input parameters are not mutated

When experimenting with custom pipelines or algorithms, prefer
``safe_run``; once things are stable, ``run`` can be used directly if
you need slightly less overhead.

Inspecting intermediate results
-------------------------------

Because the pipeline exposes results per step, you can inspect or plot
intermediate stages:

.. code-block:: python

   hb = pipeline.heartbeat_segmentation_results_
   q = pipeline.q_peak_results_
   c = pipeline.c_point_results_
   b_raw = pipeline.b_point_results_
   b_corr = pipeline.b_point_after_outlier_correction_results_
   pep = pipeline.pep_results_

   # Example: join PEP with heartbeat table
   pep_with_hb = pep.join(hb, how="left")

Configuring parameters
----------------------

All algorithms and the pipeline follow the tpcp ``get_params``
/ ``set_params`` convention.

.. code-block:: python

   # Inspect all parameters (including nested algorithms)
   print(pipeline.get_params())

   # Change the Q-peak offset
   pipeline = pipeline.set_params(q_peak_algo__time_interval_ms=50)

   # Change outlier correction behavior
   pipeline = pipeline.set_params(
       outlier_correction_algo__max_gap_beats=3,
   )

After changing parameters, simply call ``safe_run`` again.

Working with multiple datapoints
--------------------------------

Pipelines are **stateless** with respect to the dataset: you reuse the
same pipeline instance (or cloned copies) for each datapoint.

.. code-block:: python

   for dp in ds:
       res = pipeline.clone().safe_run(dp)
       # store res.pep_results_ somewhere

For large evaluation runs, you will normally let
:class:`PepEvaluationChallenge` handle this loop (see
:ref:`user_guide_evaluation`).

``PepExtractionPipelineReferenceQPeaks`` and ``PepExtractionPipelineReferenceBPoints``
--------------------------------------------------------------------------------------

pepbench also includes convenience pipelines that use **reference
annotations** instead of algorithmic detection for either Q-peaks or
B-points.

These are useful for:

* upper-bound comparisons (algorithm vs “perfect” reference)
* sanity checks on datasets and scoring

Refer to the API reference for these specialized classes.
