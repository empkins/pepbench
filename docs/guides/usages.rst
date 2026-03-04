Why use pepbench?
=================

What you'll need
----------------

- A dataset with ECG and ICG signals (see :class:`~pepbench.datasets.ExampleDataset`, :class:`~pepbench.datasets.EmpkinsDataset` or :class:`~pepbench.datasets.GuardianDataset`).
- Familiarity with NumPy / pandas. We assume :mod:`pepbench` is already installed—see the project's :doc:`Installation instructions <../README>` for installation and environment hints.
- An algorithm selection for heartbeat segmentation, Q-peak extraction and B-point extraction (any algorithm listed in :doc:`../modules/algorithms/index`).
- (Optional) Reference labels for Q-peaks / B-points to evaluate accuracy.

Introduction
------------

This short guide shows three common usages of pepbench:

1. Extract pre-ejection period (PEP) by detecting Q-peaks in the ECG and B-points in the ICG.
2. Compare how different algorithms perform (when reference labels are available).
3. Insert and benchmark your own algorithms against the built-in implementations.

Each section contains a concise explanation, expected inputs/outputs, a minimal example using concrete algorithm classes, and pointers to related documentation.

1) Compute PEP
---------------

Goal
  Extract Q-peaks from ECG and B-points from ICG and compute PEP for each heartbeat.

Inputs
  - ECG and ICG signals (see :attr:`~pepbench.datasets.BasePepDataset.ecg`, :attr:`~pepbench.datasets.BasePepDataset.icg`).
  - Optional: reference heartbeats or labels when available (:attr:`~pepbench.datasets.BasePepDatasetWithAnnotations.reference_pep`, :attr:`~pepbench.datasets.BasePepDatasetWithAnnotations.reference_heartbeats`, :attr:`~pepbench.datasets.BasePepDatasetWithAnnotations.reference_labels_ecg` / :attr:`~pepbench.datasets.BasePepDatasetWithAnnotations.reference_labels_icg`)

Typical pipeline
: Use the provided pipelines to run a sequence of algorithms in the right order:
  - heartbeat segmentation -> Q-peak extraction -> C-point (optional) -> B-point extraction -> outlier correction -> PEP computation.

Minimal example

.. code-block:: python

    from pepbench.pipelines import PepExtractionPipeline
    from pepbench.algorithms import heartbeat_segmentation, ecg, icg

    # concrete algorithm choices (you can replace these with any algorithm from the algorithms reference)
    heartbeat_algo = heartbeat_segmentation.HeartbeatSegmentationNeurokit()
    q_algo = ecg.QPeakExtractionVanLien2013()
    b_algo = icg.BPointExtractionForouzanfar2018()

    pipeline = PepExtractionPipeline(
        heartbeat_segmentation_algo=heartbeat_algo,
        q_peak_algo=q_algo,
        b_point_algo=b_algo,
    )

    # datapoint should implement the dataset interface (ECG/ICG signals + sampling rates)
    pipeline.run(datapoint)
    pep_results = pipeline.pep_results_

Notes and tuning
: - Many B-point algorithms accept parameters for smoothing, derivative thresholds or minimum peak prominence — tune these to your signal quality.
  - If ICG or ECG are noisy, consider applying preprocessing from :mod:`pepbench.algorithms.preprocessing`.
  - Use the reference pipelines (see section 2) to validate outputs when labels exist.

2) Compare algorithm performance (what detects what better/worse)
----------------------------------------------------------------

Goal
: Evaluate and compare algorithm performance using datasets that provide reference labels (Q-peaks, B-points, or PEP).

Inputs
: - A labeled dataset (example datasets: `example_data/Empkins_Dataset` or any dataset implementing the project's dataset interface).
  - A list of algorithms to evaluate for the same detection task.

Recommended metrics
: - True positives / false positives / false negatives per heartbeat (match heartbeats first).
  - Mean absolute error (ms) of detected event timing vs. reference.
  - Coverage (how many heartbeats could be processed) and failure modes (NaNs / rejected beats).

Evaluation workflow (concise)
: 1. Use a reference pipeline that plugs either reference B-points or reference Q-peaks (see :class:`pepbench.pipelines.PepExtractionPipelineReferenceBPoints` and :class:`pepbench.pipelines.PepExtractionPipelineReferenceQPeaks`).
  2. For each algorithm, run the pipeline on all labeled datapoints and collect per-heartbeat differences.
  3. Aggregate results across subjects and report distributions (boxplots, mean/median error).

Minimal example

.. code-block:: python

    from pepbench.pipelines import PepExtractionPipelineReferenceBPoints
    from pepbench.algorithms import heartbeat_segmentation, ecg, icg

    heartbeat_algo = heartbeat_segmentation.HeartbeatSegmentationNeurokit()
    q_algo = ecg.QPeakExtractionVanLien2013()
    b_algo = icg.BPointExtractionDrost2022()

    ref_pipeline = PepExtractionPipelineReferenceBPoints(
        heartbeat_segmentation_algo=heartbeat_algo,
        q_peak_algo=q_algo,
        b_point_algo=b_algo,
    )

    # loop over labeled datapoints and collect ref_pipeline.pep_results_

Interpretation tips
: - Inspect matched vs. unmatched heartbeats to identify whether mismatches arise from heartbeat segmentation, Q-peak detection, or B-point localization.
  - Use visualization helpers in :doc:`../modules/plotting/index` (or :mod:`pepbench.plotting.algorithms`) to inspect per-beat overlays.

3) Insert your own algorithms and benchmark (plug-in / benchmarking)
------------------------------------------------------------------

Goal
: Add a custom algorithm implementation and compare it to existing algorithms using the project's pipelines and benchmarking notebooks.

Contract (simple)
: - Input: signals (ECG or ICG as required) and sampling_rate_hz.
  - Output: an algorithm class exposing a public method like ``extract(...)`` and an attribute ``points_`` (consistent with existing algorithm implementations).
  - Error modes: should raise a clear exception on invalid input and return empty/NaN results if no events found.

How to implement
: 1. Follow the interface of the existing base classes in :mod:`pepbench.algorithms` (see source under ``src/pepbench/algorithms``). Implement ``extract(...)`` and set ``points_`` to a DataFrame with the expected columns (e.g. ``q_peak_sample`` or ``b_point_sample``).
  2. Optionally provide a ``clone()`` or follow the pattern used by other algorithms so pipelines can safely copy instances.
  3. Add a minimal unit test under ``tests/`` exercising its happy path and one edge case.

Minimal example (skeleton)

.. code-block:: python

    from pepbench.algorithms import BaseEcgExtraction  # or BaseBPointExtraction
    import pandas as pd

    class MyQPeakAlgo(BaseEcgExtraction):
        def extract(self, ecg, heartbeats, sampling_rate_hz: float):
            # implement detection and set self.points_ to a DataFrame
            self.points_ = pd.DataFrame({"q_peak_sample": []})

    # use MyQPeakAlgo in the same pipelines as built-ins to benchmark

Benchmarking tips
: - Reuse the existing benchmarking notebooks under ``experiments/pep_algorithm_benchmarking`` for batch runs and plotting.
  - Compare across algorithms using the same heartbeat segmentation and dataset to isolate differences to the extraction step.
  - Automate parameter sweeps (grid search) when tuning algorithm hyperparameters and record results per parameter set.

See also
--------

- :doc:`../modules/algorithms/index` (algorithm reference and available implementations)
- :doc:`../guides/pipelines` (pipeline usage and parameter reference)
- :doc:`../guides/evaluation` (evaluation strategies and metrics)
- Source: ``src/pepbench/algorithms`` and ``src/pepbench/pipelines`` for implementation patterns and interfaces

Acknowledgements and next steps
------------------------------

This guide is a compact starting point. Consider adding:

- A short worked example notebook that runs a full benchmark on a small example dataset.
- A troubleshooting section for common failure modes (e.g., mismatched sampling rates, noisy signals).
