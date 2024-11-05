from typing import Literal, Optional

import pandas as pd
from biopsykit.signals._base_extraction import HANDLE_MISSING_EVENTS
from biopsykit.signals.ecg.event_extraction import BaseEcgExtraction
from biopsykit.signals.ecg.segmentation import BaseHeartbeatSegmentation
from biopsykit.signals.icg.event_extraction import BaseBPointExtraction, CPointExtractionScipyFindPeaks
from biopsykit.signals.icg.outlier_correction import BaseOutlierCorrection
from biopsykit.signals.pep import PepExtraction
from biopsykit.signals.pep._pep_extraction import NEGATIVE_PEP_HANDLING
from tpcp import Parameter, Pipeline

__all__ = ["BasePepExtractionPipeline"]


class BasePepExtractionPipeline(Pipeline):
    heartbeat_segmentation_algo: Parameter[BaseHeartbeatSegmentation]
    q_wave_algo: Parameter[BaseEcgExtraction]
    b_point_algo: Parameter[BaseBPointExtraction]
    outlier_correction_algo: Parameter[BaseOutlierCorrection]
    handle_negative_pep: NEGATIVE_PEP_HANDLING
    handle_missing_events: HANDLE_MISSING_EVENTS

    heartbeat_segmentation_results_: pd.DataFrame
    q_wave_results_: pd.DataFrame
    c_point_results_: Optional[pd.DataFrame]
    b_point_results_: pd.DataFrame
    b_point_after_outlier_correction_results_: pd.DataFrame
    pep_results_: pd.DataFrame

    def __init__(
        self,
        *,
        heartbeat_segmentation_algo: BaseHeartbeatSegmentation,
        q_wave_algo: BaseEcgExtraction,
        b_point_algo: BaseBPointExtraction,
        outlier_correction_algo: BaseOutlierCorrection,
        handle_negative_pep: Literal[NEGATIVE_PEP_HANDLING] = "nan",
        handle_missing_events: Optional[Literal[HANDLE_MISSING_EVENTS]] = None,
    ) -> None:
        self.heartbeat_segmentation_algo = heartbeat_segmentation_algo
        self.q_wave_algo = q_wave_algo
        self.b_point_algo = b_point_algo
        self.c_point_algo = CPointExtractionScipyFindPeaks()
        self.outlier_correction_algo = outlier_correction_algo
        self.pep_extraction_algo = PepExtraction()
        self.handle_negative_pep = handle_negative_pep

        if handle_missing_events is None:
            handle_missing_events = "warn"
        self.handle_missing_events = handle_missing_events

    def _compute_pep(
        self,
        *,
        heartbeats: pd.DataFrame,
        q_wave_onset_samples: pd.DataFrame,
        b_point_samples: pd.DataFrame,
        sampling_rate_hz: int,
    ) -> pd.DataFrame:
        pep_extraction_algo = PepExtraction(handle_negative_pep=self.handle_negative_pep)
        pep_extraction_algo.extract(
            heartbeats=heartbeats,
            q_wave_onset_samples=q_wave_onset_samples,
            b_point_samples=b_point_samples,
            sampling_rate_hz=sampling_rate_hz,
        )

        pep_results = pep_extraction_algo.pep_results_.copy()
        pep_results = pep_results.convert_dtypes(infer_objects=True)

        return pep_results
