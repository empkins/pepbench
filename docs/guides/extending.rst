.. _user_guide_extending:

Extending pepbench with Your Own Algorithm
==========================================

pepbench is designed to be **extensible**: you can add new PEP extraction
algorithms and still reuse the existing pipelines and evaluation
infrastructure.

This guide outlines the main steps:

1. Implement a new algorithm class in the appropriate module.
2. Document it with a clear docstring and literature reference.
3. (Optionally) expose it in the public API.
4. Use it in :class:`PepExtractionPipeline` and
   :class:`PepEvaluationChallenge`.

Where to put new algorithms
---------------------------

The algorithm modules follow a structure similar to the
`mobgap <https://mobgap.readthedocs.io/>`_ toolbox: base classes are in
“public” modules, concrete algorithms live in dedicated files and are
re-exported via ``__init__.py``.

pepbench has submodules such as:

* ``pepbench.algorithms.ecg`` – Q-peak algorithms
* ``pepbench.algorithms.icg`` – B- and C-point algorithms
* ``pepbench.algorithms.heartbeat_segmentation``
* ``pepbench.algorithms.outlier_correction``

Choose the module corresponding to your algorithm’s task. For example, a
new B-point strategy would go into ``pepbench/algorithms/icg/``.

Base classes and interface
--------------------------

Algorithms usually subclass *biopsykit* / tpcp-style base classes, e.g.:

* :class:`biopsykit.signals.ecg.BaseEcgExtraction`
* :class:`biopsykit.signals.icg.BaseBPointExtraction`
* :class:`biopsykit.signals.icg.BaseCPointExtraction`

The crucial requirements are:

* implement an :meth:`extract` method with signature compatible with the
  base class (e.g., ``extract(ecg, heartbeats, sampling_rate_hz)`` for
  Q-peak methods) and
* store results in a ``points_`` or similar attribute, as the existing
  algorithms do.

A minimal example (Q-peak)
--------------------------

.. code-block:: python

   # pepbench/algorithms/ecg/_my_q_peak_algo.py

   from biopsykit.signals.ecg import BaseEcgExtraction
   import pandas as pd

   class QPeakExtractionMyMethod(BaseEcgExtraction):
       """Q-peak extraction algorithm based on <Your Paper>.

       This algorithm implements the Q-peak detection method described by
       <Author et al., Year>, adapted to the pepbench / tpcp interface.

       Parameters
       ----------
       some_parameter : float
           Method-specific parameter controlling ...
       handle_missing_events : {"warn", "raise", "ignore"}, optional
           How to handle missing events in the input dataframes.
       """

       def __init__(self, some_parameter=0.1, handle_missing_events="warn"):
           self.some_parameter = some_parameter
           self.handle_missing_events = handle_missing_events

       def extract(self, *, ecg, heartbeats, sampling_rate_hz):
           # Implement your detection here, using `ecg` and `heartbeats`
           # Return self and populate `self.points_` similar to existing algorithms
           q_points = self._detect_q_points(ecg, heartbeats, sampling_rate_hz)
           self.points_ = pd.DataFrame({"q_peak": q_points}, index=heartbeats.index)
           return self

       def _detect_q_points(self, ecg, heartbeats, sampling_rate_hz):
           # Your algorithm implementation here
           ...

Re-export the class in ``pepbench/algorithms/ecg/__init__.py`` so users
can import it as:

.. code-block:: python

   from pepbench.algorithms.ecg import QPeakExtractionMyMethod

Docstring and references
------------------------

For scientific algorithms, it is essential to include:

* a short description of the method,
* the citation of the original paper (Journal, year, DOI), and
* any assumptions (e.g., sampling rate, filtering).

The existing algorithm docstrings (e.g.
:class:`QPeakExtractionVanLien2013`, :class:`BPointExtractionStern1985`)
are good examples to follow.

Integrating your algorithm into pipelines
-----------------------------------------

Once the class is implemented and importable, you can use it in
:class:`PepExtractionPipeline` just like the built-in algorithms:

.. code-block:: python

   from pepbench.pipelines import PepExtractionPipeline
   from pepbench.algorithms.heartbeat_segmentation import HeartbeatSegmentationNeurokit
   from pepbench.algorithms.ecg import QPeakExtractionMyMethod
   from pepbench.algorithms.icg import BPointExtractionLozano2007LinearRegression

   pipeline = PepExtractionPipeline(
       heartbeat_segmentation_algo=HeartbeatSegmentationNeurokit(),
       q_peak_algo=QPeakExtractionMyMethod(some_parameter=0.2),
       b_point_algo=BPointExtractionLozano2007LinearRegression(),
   )

   dp = next(iter(ds))
   pipeline = pipeline.safe_run(dp)

   pep_df = pipeline.pep_results_

Using the evaluation framework
------------------------------

To benchmark your new algorithm against existing ones, simply plug your
pipeline into :class:`PepEvaluationChallenge`:

.. code-block:: python

   from pepbench.evaluation import PepEvaluationChallenge, score_pep_evaluation

   challenge = PepEvaluationChallenge(
       dataset=ds,
       scoring=score_pep_evaluation,
   )

   challenge = challenge.run(pipeline).results_as_df()

   print(challenge.results_agg_mean_std_)

This reproduces the **systematic benchmarking** idea of the PEPbench
paper, now including your custom algorithm.

Testing and quality checks
--------------------------

Following best practices from API usability and developer-tool
documentation, new algorithms should be accompanied by:

* unit tests verifying that ``extract`` runs and populates result
  attributes,
* basic sanity checks on PEP distributions (no massive negative values,
  reasonable ranges),
* clear docstrings and examples.

You can mirror the approach from the mobgap `Developer Guide` for
testing new algorithms, adapting tpcp’s
``TestAlgorithmMixin`` pattern to your needs.
