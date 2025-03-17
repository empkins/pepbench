.. _api_ref_algorithms:

Algorithms
==========

.. automodule:: pepbench.algorithms
    :no-members:
    :no-inherited-members:

Q-Peak Extraction Algorithms
++++++++++++++++++++++++++++

.. currentmodule:: pepbench.algorithms.ecg

.. autosummary::
   :toctree: generated/ecg
   :template: class.rst

    QPeakExtractionForouzanfar2018
    QPeakExtractionMartinez2004Neurokit
    QPeakExtractionVanLien2013


B-Point Extraction Algorithms
+++++++++++++++++++++++++++++

.. currentmodule:: pepbench.algorithms.icg

.. autosummary::
   :toctree: generated/icg
   :template: class.rst

    BPointExtractionArbol2017IsoelectricCrossings
    BPointExtractionArbol2017SecondDerivative
    BPointExtractionArbol2017ThirdDerivative
    BPointExtractionDebski1993SecondDerivative
    BPointExtractionDrost2022
    BPointExtractionForouzanfar2018
    BPointExtractionLozano2007LinearRegression
    BPointExtractionLozano2007QuadraticRegression
    BPointExtractionSherwood1990
    BPointExtractionStern1985


Outlier Correction Algorithms
+++++++++++++++++++++++++++++
.. currentmodule:: pepbench.algorithms.outlier_correction

.. autosummary::
   :toctree: generated/outlier_correction
   :template: class.rst

    OutlierCorrectionDummy
    OutlierCorrectionForouzanfar2018
    OutlierCorrectionLinearInterpolation


C-Point Extraction Algorithms
+++++++++++++++++++++++++++++

.. currentmodule:: pepbench.algorithms.icg

.. autosummary::
   :toctree: generated/icg
   :template: class.rst

    CPointExtractionScipyFindPeaks

Heartbeat Segmentation Algorithms
+++++++++++++++++++++++++++++++++

.. currentmodule:: pepbench.algorithms.heartbeat_segmentation

.. autosummary::
   :toctree: generated/heartbeat_segmentation
   :template: class.rst

    HeartbeatSegmentationNeurokit