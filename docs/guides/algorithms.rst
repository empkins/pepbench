.. _user_guide_algorithms:

Choosing PEP Extraction Algorithms
==================================

pepbench implements a collection of ECG and ICG algorithms from the
psychophysiology literature for detecting Q-peaks, B-points, and C-points,
as well as heartbeat segmentation and outlier correction.

The goal of this guide is to help you **choose algorithms** for a given
study design and signal quality — not to replace the API reference.

Overview of algorithm families
------------------------------

The algorithms fall into the following categories:

* ``heartbeat_segmentation`` – find individual heartbeats in the ECG
* ``ecg`` – Q-peak detection (fiducial point on ECG)
* ``icg`` – B-point and C-point detection on ICG
* ``outlier_correction`` – post-hoc correction of implausible B-points

They are combined into a pipeline via
:class:`pepbench.pipelines.PepExtractionPipeline`. See the
:ref:`user_guide_pipelines` for details.

Q-peak extraction algorithms
----------------------------

pepbench currently includes (see Algorithms API page for full details):

* :class:`pepbench.algorithms.ecg.QPeakExtractionForouzanfar2018`
* :class:`pepbench.algorithms.ecg.QPeakExtractionVanLien2013`
* :class:`pepbench.algorithms.ecg.QPeakExtractionMartinez2004Neurokit`


All are based on published ECG methods:

* Forouzanfar et al. (2018) – joint ECG/ICG approach for PEP
* Van Lien et al. (2013) – estimate Q from R-peak and fixed offset
* Martinez et al. (2004) – wavelet-based QRS delineation


Very roughly:

* **Forouzanfar2018** – tuned for a specific PEP extraction workflow;
  best used when reproducing or extending that approach.
* **VanLien2013** – simple and robust when R-peaks are reliable.
* **Martinez2004Neurokit** – more sophisticated delineation, useful if
  you already rely on NeuroKit2 preprocessing.


A simple example:

.. code-block:: python

   from pepbench.algorithms.ecg import QPeakExtractionVanLien2013

   q_algo = QPeakExtractionVanLien2013(time_interval_ms=40)

   q_algo = q_algo.extract(
       ecg=datapoint.ecg,
       heartbeats=datapoint.heartbeats,
       sampling_rate_hz=datapoint.sampling_rate_ecg,
   )

   q_peaks = q_algo.points_   # DataFrame with Q-peak locations

B-point extraction algorithms
-----------------------------

B-point extraction on ICG is more challenging and multiple heuristics
exist. pepbench includes algorithms such as:

* :class:`pepbench.algorithms.icg.BPointExtractionDrost2022`
* :class:`pepbench.algorithms.icg.BPointExtractionForouzanfar2018`
* :class:`pepbench.algorithms.icg.BPointExtractionArbol2017IsoelectricCrossings`
* :class:`pepbench.algorithms.icg.BPointExtractionArbol2017SecondDerivative`
* :class:`pepbench.algorithms.icg.BPointExtractionArbol2017ThirdDerivative`
* :class:`pepbench.algorithms.icg.BPointExtractionLozano2007LinearRegression`
* :class:`pepbench.algorithms.icg.BPointExtractionLozano2007QuadraticRegression`
* :class:`pepbench.algorithms.icg.BPointExtractionDebski1993SecondDerivative`
* :class:`pepbench.algorithms.icg.BPointExtractionStern1985`

The underlying papers differ in how they define the B-point on the
dZ/dt curve (local minima, isoelectric crossings, derivative extremes,
regression-based estimates).

Very coarse guidance (always check the paper for your use case):

* **Drost2022** – derivative-based method that may be more sensitive to noise but better pinpoints rapid upstrokes.
* **Forouzanfar2018** – tailored to the PEPbench-style joint ECG/ICG workflow.
* **Arbol2017** – derivative-based method that may be more sensitive to noise but better pinpoints rapid upstrokes.
* **Lozano2007 variants** – regression-based methods; often suggested for noisy data and ambulatory settings.
* **Debski1993** – early derivative-based method; among the first to define B-point via second-derivative zero crossings.
* **Stern1985** – classic local-minimum approach in dZ/dt.


C-point extraction and heartbeat segmentation
---------------------------------------------

Several B-point algorithms require the ICG **C-point** as a reference.
pepbench provides:

* :class:`pepbench.algorithms.icg.CPointExtractionScipyFindPeaks`

and for heartbeats:

* :class:`pepbench.algorithms.heartbeat_segmentation.HeartbeatSegmentationNeurokit`

These are used internally by :class:`PepExtractionPipeline` and usually
do not need to be changed unless you are experimenting with alternative
segmentations.

Outlier correction
------------------

After detecting B-points, outlier correction can stabilize PEP values:

* :class:`pepbench.algorithms.outlier_correction.OutlierCorrectionDummy`
* :class:`pepbench.algorithms.outlier_correction.OutlierCorrectionForouzanfar2018`
* :class:`pepbench.algorithms.outlier_correction.OutlierCorrectionLinearInterpolation`

Use cases:

* **No post-processing** – use ``OutlierCorrectionDummy``.
* **Local corrections based on algorithm logic** – use the Forouzanfar
  variant.
* **Gap-filling for sporadic failures** – use linear interpolation.

Putting it together: choosing algorithms
----------------------------------------

For many studies, a reasonable starting point is:

.. code-block:: python

   from pepbench.algorithms.heartbeat_segmentation import HeartbeatSegmentationNeurokit
   from pepbench.algorithms.ecg import QPeakExtractionVanLien2013
   from pepbench.algorithms.icg import BPointExtractionLozano2007LinearRegression
   from pepbench.algorithms.outlier_correction import OutlierCorrectionLinearInterpolation

   heartbeat_algo = HeartbeatSegmentationNeurokit()
   q_algo = QPeakExtractionVanLien2013()
   b_algo = BPointExtractionLozano2007LinearRegression()
   outlier_algo = OutlierCorrectionLinearInterpolation()

   # Plug these into a PepExtractionPipeline (see pipelines guide)

You can then compare algorithm combinations systematically using the
evaluation challenge (see :ref:`user_guide_evaluation`), which is
precisely the use case the PEPbench framework was designed for.
