"""Individual algorithms for preprocessing signals and extracting fiducial points from ECG and ICG signals.

Together, these algorithms can be combined into a PEP extraction pipeline
(e.g., :class:`~pepbench.pipelines.PepExtractionPipeline`).

The following categories of algorithms are available:
    * :mod:`~pepbench.algorithms.heartbeat_segmentation` - Algorithms for heartbeat segmentation
    * :mod:`~pepbench.algorithms.preprocessing` - Algorithms for preprocessing of ECG and ICG signals
    * :mod:`~pepbench.algorithms.ecg` - Algorithms for ECG fiducial point extraction
    * :mod:`~pepbench.algorithms.icg` - Algorithms for ICG fiducial point extraction
    * :mod:`~pepbench.algorithms.outlier_correction` - Algorithms for outlier correction of ICG fiducial points

"""

from pepbench.algorithms import ecg, heartbeat_segmentation, icg, outlier_correction, preprocessing

__all__ = ["ecg", "heartbeat_segmentation", "icg", "outlier_correction", "preprocessing"]
