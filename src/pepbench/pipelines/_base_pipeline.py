"""Base PEP extraction pipeline utilities.

This module provides the base pipeline building block used to extract PEP-related signals
from synchronized ECG and ICG recordings. The primary export is
:class:`pepbench.pipelines._base_pipeline.BasePepExtractionPipeline`, a lightweight wrapper
around :class:`tpcp.Pipeline` that bundles heartbeat segmentation, Q-peak, B-point and
C-point extraction algorithms together with optional outlier correction and PEP post-processing.

The module also exposes a small docstring filler helper
(:data:`base_pep_pipeline_docfiller`) used to inject common parameter and attribute
documentation into subclasses of :class:`BasePepExtractionPipeline`.

See Also
--------
:mod:`pepbench.datasets`
    Dataset interfaces expected by the pipelines (e.g. :class:`~pepbench.datasets.BasePepDataset`).
:mod:`biopsykit.signals.ecg.segmentation`
    Heartbeat segmentation algorithms used by the pipeline.
:mod:`biopsykit.signals.ecg.event_extraction`
    Q-peak extraction interfaces used by the pipeline.
:mod:`biopsykit.signals.icg.event_extraction`
    B- and C-point extraction interfaces used by the pipeline.

Public classes
--------------
:class:`~pepbench.pipelines._base_pipeline.BasePepExtractionPipeline`
    Pipeline class orchestrating heartbeat segmentation, event extraction, outlier correction
    and PEP computation.

Notes
-----
- The class is intended to be subclassed or configured with concrete algorithm implementations.
- The module follows the project-wide NumPy/Sphinx docstring conventions used across the package.
"""

from typing import TYPE_CHECKING, Literal, TypeVar

import pandas as pd
from biopsykit.signals._base_extraction import HANDLE_MISSING_EVENTS
from biopsykit.signals.ecg.event_extraction import BaseEcgExtraction
from biopsykit.signals.ecg.segmentation import BaseHeartbeatSegmentation
from biopsykit.signals.icg.event_extraction import (
    BaseBPointExtraction,
    BaseCPointExtraction,
    CPointExtractionScipyFindPeaks,
)
from biopsykit.signals.icg.outlier_correction import BaseBPointOutlierCorrection, OutlierCorrectionDummy
from biopsykit.signals.pep import PepExtraction
from biopsykit.signals.pep._pep_extraction import NEGATIVE_PEP_HANDLING
from biopsykit.utils.dtypes import (
    BPointDataFrame,
    CPointDataFrame,
    HeartbeatSegmentationDataFrame,
    PepResultDataFrame,
    QPeakDataFrame,
    is_pep_result_dataframe,
)
from tpcp import CloneFactory, Parameter, Pipeline

from pepbench._docutils import make_filldoc

if TYPE_CHECKING:
    from pepbench.datasets import BasePepDataset, BasePepDatasetWithAnnotations

__all__ = ["BasePepExtractionPipeline"]


BasePepDatasetT = TypeVar("BasePepDatasetT", bound="BasePepDataset")
BasePepDatasetWithAnnotationsT = TypeVar("BasePepDatasetWithAnnotationsT", bound="BasePepDatasetWithAnnotations")

base_pep_pipeline_docfiller = make_filldoc(
    {
        "base_parameters": """
        heartbeat_segmentation_algo : :class:`~biopsykit.signals.ecg.segmentation.BaseHeartbeatSegmentation`
            Algorithm for heartbeat segmentation.
        q_peak_algo : :class:`~biopsykit.signals.ecg.event_extraction.BaseEcgExtraction`
            Algorithm for Q-peak extraction.
        b_point_algo : :class:`~biopsykit.signals.icg.event_extraction.BaseBPointExtraction`
            Algorithm for B-point extraction.
        c_point_algo : :class:`~biopsykit.signals.icg.event_extraction.BaseCPointExtraction`
            Algorithm for C-point extraction, necessary for most subsequent B-point extraction algorithms.
        outlier_correction_algo : :class:`~biopsykit.signals.icg.outlier_correction.BaseOutlierCorrection`
            Algorithm for outlier correction of B-point data (optional).
        handle_negative_pep : one of {`"nan"`, `"zero"`, `"keep"`}
            How to handle negative PEP values. Possible values are:
                - `"nan"`: Set negative PEP values to NaN
                - `"zero"`: Set negative PEP values to 0
                - `"keep"`: Keep negative PEP values as is
        handle_missing_events : one of {`"warn"`, `"ignore"`, `"raise"`}
            How to handle missing events. Possible values are:
                - `"warn"`: Issue a warning if missing events are detected
                - `"ignore"`: Ignore missing events
                - `"raise"`: Raise an error if missing events are detected
        """,
        "datapoint_pipeline": """
        datapoint : :class:`~pepbench.datasets._base_pep_extraction_dataset.BasePepDataset`
            The data to run the pipeline on. This needs to be a valid datapoint (i.e. a dataset with just a single row).
            The Dataset should be a child class of
            :class:`~pepbench.datasets._base_pep_extraction_dataset.BasePepDataset` or implement all the same
            parameters and methods.
        """,
        "datapoint_pipeline_labeled": """
        datapoint : :class:`~pepbench.datasets._base_pep_extraction_dataset.BaseUnifiedPepExtractionDataset`
            The data to run the pipeline on. This needs to be a valid datapoint (i.e. a dataset with just a single row).
            The Dataset should be a child class of
            :class:`~pepbench.datasets._base_pep_extraction_dataset.BaseUnifiedPepExtractionDataset` or implement all
            the same parameters and methods. This means that it must *also* implement methods to get the reference
            heartbeats and reference PEP.
            """,
        "attributes": """
        heartbeat_segmentation_results_ : :class:`~biopsykit.signals.ecg.segmentation.HeartbeatSegmentationDataFrame`
            Results from the heartbeat segmentation step.
        q_peak_results_ : :class:`~biopsykit.signals.ecg.event_extraction.QPeakDataFrame`
            Results from the Q-peak extraction step.
        c_point_results_ : :class:`~biopsykit.signals.icg.event_extraction.CPointDataFrame`
            Results from the C-point extraction step.
        b_point_results_ : :class:`~biopsykit.signals.icg.event_extraction.BPointDataFrame`
            Results from the B-point extraction step.
        b_point_after_outlier_correction_results_ : :class:`~biopsykit.signals.icg.event_extraction.BPointDataFrame`
            Results from the B-point extraction step after outlier correction.
        pep_results_ : :class:`~biopsykit.signals.pep.PepResultDataFrame`
            Results from the PEP extraction step.
        """,
    },
    doc_summary="Decorator to fill common parts of the docstring for subclasses of :class:`BasePepExtractionPipeline`.",
)


@base_pep_pipeline_docfiller
class BasePepExtractionPipeline(Pipeline):
    """Base class for PEP extraction pipelines.

    This class provides all the necessary methods to extract PEP from ECG and ICG data using the specified algorithms.
    For usage, it is recommended to use the derived pipelines
    (e.g., :class:`~pepbench.pipelines.PepExtractionPipeline`).

    %(base_parameters)s

    %(attributes)s

    """

    heartbeat_segmentation_algo: Parameter[BaseHeartbeatSegmentation]
    q_peak_algo: Parameter[BaseEcgExtraction]
    b_point_algo: Parameter[BaseBPointExtraction]
    c_point_algo: Parameter[BaseCPointExtraction]
    outlier_correction_algo: Parameter[BaseBPointOutlierCorrection]
    handle_negative_pep: NEGATIVE_PEP_HANDLING
    handle_missing_events: HANDLE_MISSING_EVENTS

    heartbeat_segmentation_results_: HeartbeatSegmentationDataFrame
    q_peak_results_: QPeakDataFrame
    c_point_results_: CPointDataFrame | None
    b_point_results_: BPointDataFrame
    b_point_after_outlier_correction_results_: BPointDataFrame
    pep_results_: PepResultDataFrame

    def __init__(
        self,
        *,
        heartbeat_segmentation_algo: BaseHeartbeatSegmentation,
        q_peak_algo: BaseEcgExtraction,
        b_point_algo: BaseBPointExtraction,
        c_point_algo: BaseCPointExtraction = CloneFactory(CPointExtractionScipyFindPeaks()),
        outlier_correction_algo: BaseBPointOutlierCorrection | None = None,
        handle_negative_pep: Literal[NEGATIVE_PEP_HANDLING] = "nan",
        handle_missing_events: Literal[HANDLE_MISSING_EVENTS] | None = None,
    ) -> None:
        """Initialize a :class:`~pepbench.pipelines._base_pipeline.BasePepExtractionPipeline`.

        Parameters
        ----------
        heartbeat_segmentation_algo : BaseHeartbeatSegmentation
            Algorithm instance used to segment ECG into heartbeats.
        q_peak_algo : BaseEcgExtraction
            Algorithm instance used to detect Q-peaks in the ECG.
        b_point_algo : BaseBPointExtraction
            Algorithm instance used to detect B-points in the ICG.
        c_point_algo : BaseCPointExtraction, optional
            Algorithm used to detect C-points in the ICG. Required by many B-point extractors.
            Defaults to a scipy-based peak finder clone.
        outlier_correction_algo : BaseBPointOutlierCorrection or None, optional
            Algorithm for outlier correction applied to B-point results. If ``None``, a dummy no-op
            outlier corrector is used.
        handle_negative_pep : {'nan', 'zero', 'keep'}, optional
            Strategy to handle negative PEP values:
                - ``'nan'``: set negative PEP to NaN
                - ``'zero'``: set negative PEP to 0
                - ``'keep'``: keep negative values as-is
            Default is ``'nan'``.
        handle_missing_events : {'warn', 'ignore', 'raise'} or None, optional
            Strategy to handle missing events during extraction. If ``None``, defaults to ``'warn'``.

        """
        self.heartbeat_segmentation_algo = heartbeat_segmentation_algo
        self.q_peak_algo = q_peak_algo
        self.b_point_algo = b_point_algo
        self.c_point_algo = c_point_algo
        if outlier_correction_algo is None:
            outlier_correction_algo = OutlierCorrectionDummy()
        self.outlier_correction_algo = outlier_correction_algo
        self.pep_extraction_algo = PepExtraction()
        self.handle_negative_pep = handle_negative_pep

        if handle_missing_events is None:
            handle_missing_events = "warn"
        self.handle_missing_events = handle_missing_events

    def _compute_pep(
        self,
        *,
        heartbeats: HeartbeatSegmentationDataFrame,
        q_peak_samples: QPeakDataFrame,
        b_point_samples: BPointDataFrame,
        sampling_rate_hz: float,
    ) -> pd.DataFrame:
        """Compute PEP results from heartbeat segmentation and event detections.

        This helper runs an internal :class:`biopsykit.signals.pep.PepExtraction` instance to
        compute PEP-related metrics from the provided heartbeat, Q-peak and B-point data and
        returns a well-typed :class:`pandas.DataFrame` compatible with the package's expected
        ``PepResultDataFrame`` schema.

        Parameters
        ----------
        heartbeats : HeartbeatSegmentationDataFrame
            Segmented heartbeat frames with at least ``start_sample`` and ``end_sample`` indices.
        q_peak_samples : QPeakDataFrame
            Detected Q-peak sample indices.
        b_point_samples : BPointDataFrame
            Detected B-point sample indices.
        sampling_rate_hz : float
            Sampling frequency of the recordings in Hz.

        Returns
        -------
        :class:`~pandas.DataFrame`
            PEP result dataframe containing columns such as ``q_peak_sample``, ``b_point_sample``,
            ``pep_sample``, ``pep_ms`` and supporting metadata. The returned dataframe will have
            pandas nullable dtypes applied (e.g., ``Int64``, ``Float64``) and be validated with
            :func:`pepbench.utils.dtypes.is_pep_result_dataframe`.
        """
        pep_extraction_algo = PepExtraction(handle_negative_pep=self.handle_negative_pep)
        pep_extraction_algo.extract(
            heartbeats=heartbeats,
            q_peak_samples=q_peak_samples,
            b_point_samples=b_point_samples,
            sampling_rate_hz=sampling_rate_hz,
        )

        pep_results = pep_extraction_algo.pep_results_.copy()
        pep_results = pep_results.astype(
            {
                "heartbeat_start_sample": "Int64",
                "heartbeat_end_sample": "Int64",
                "r_peak_sample": "Int64",
                "rr_interval_sample": "Int64",
                "rr_interval_ms": "Float64",
                "heart_rate_bpm": "Float64",
                "q_peak_sample": "Int64",
                "b_point_sample": "Int64",
                "pep_sample": "Int64",
                "pep_ms": "Float64",
                "nan_reason": "object",
            }
        )

        is_pep_result_dataframe(pep_results)

        return pep_results
