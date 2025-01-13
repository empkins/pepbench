from typing import Literal, Optional

import pandas as pd
from biopsykit.signals._base_extraction import HANDLE_MISSING_EVENTS
from biopsykit.signals.ecg.event_extraction import BaseEcgExtraction
from biopsykit.signals.ecg.segmentation import BaseHeartbeatSegmentation
from biopsykit.signals.icg.event_extraction import (
    BaseBPointExtraction,
    BaseCPointExtraction,
    CPointExtractionScipyFindPeaks,
)
from biopsykit.signals.icg.outlier_correction import BaseOutlierCorrection, OutlierCorrectionDummy
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
from tpcp import Parameter, Pipeline

__all__ = ["BasePepExtractionPipeline"]


class BasePepExtractionPipeline(Pipeline):
    heartbeat_segmentation_algo: Parameter[BaseHeartbeatSegmentation]
    q_peak_algo: Parameter[BaseEcgExtraction]
    b_point_algo: Parameter[BaseBPointExtraction]
    c_point_algo: Parameter[BaseCPointExtraction]
    outlier_correction_algo: Parameter[BaseOutlierCorrection]
    handle_negative_pep: NEGATIVE_PEP_HANDLING
    handle_missing_events: HANDLE_MISSING_EVENTS

    heartbeat_segmentation_results_: HeartbeatSegmentationDataFrame
    q_peak_results_: QPeakDataFrame
    c_point_results_: Optional[CPointDataFrame]
    b_point_results_: BPointDataFrame
    b_point_after_outlier_correction_results_: BPointDataFrame
    pep_results_: PepResultDataFrame

    def __init__(
        self,
        *,
        heartbeat_segmentation_algo: BaseHeartbeatSegmentation,
        q_peak_algo: BaseEcgExtraction,
        b_point_algo: BaseBPointExtraction,
        c_point_algo: Optional[BaseCPointExtraction] = CPointExtractionScipyFindPeaks(),
        outlier_correction_algo: Optional[BaseOutlierCorrection] = None,
        handle_negative_pep: Literal[NEGATIVE_PEP_HANDLING] = "nan",
        handle_missing_events: Optional[Literal[HANDLE_MISSING_EVENTS]] = None,
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
