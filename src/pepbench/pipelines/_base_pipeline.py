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
        Parameters
        ----------
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
        Attributes
        ----------
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
