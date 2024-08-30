from typing import Literal, Optional, get_args

import numpy as np
import pandas as pd
from biopsykit.signals._base_extraction import HANDLE_MISSING_EVENTS
from biopsykit.signals.ecg.event_extraction import BaseEcgExtraction
from biopsykit.signals.ecg.segmentation import BaseHeartbeatSegmentation
from biopsykit.signals.icg.event_extraction import BaseBPointExtraction, CPointExtractionScipyFindPeaks
from biopsykit.signals.icg.outlier_correction import BaseOutlierCorrection
from biopsykit.signals.pep import PepExtraction
from tpcp import Parameter, Pipeline

NEGATIVE_PEP_HANDLING = Literal["nan", "zero", "keep"]

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
    c_point_results_: pd.DataFrame
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
        handle_negative_pep: Optional[Literal[NEGATIVE_PEP_HANDLING]] = "nan",
        handle_missing_events: Optional[Literal[HANDLE_MISSING_EVENTS]] = "warn",
    ) -> None:
        self.heartbeat_segmentation_algo = heartbeat_segmentation_algo
        self.q_wave_algo = q_wave_algo
        self.b_point_algo = b_point_algo
        self.c_point_algo = CPointExtractionScipyFindPeaks()
        self.outlier_correction_algo = outlier_correction_algo
        self.pep_extraction_algo = PepExtraction()
        if handle_negative_pep not in get_args(NEGATIVE_PEP_HANDLING):
            raise ValueError(
                f"Invalid value for 'handle_negative_pep': {handle_negative_pep}. "
                f"Must be one of {NEGATIVE_PEP_HANDLING}"
            )
        self.handle_negative_pep = handle_negative_pep
        if handle_missing_events not in get_args(HANDLE_MISSING_EVENTS):
            raise ValueError(
                f"Invalid value for 'handle_missing_events': {handle_missing_events}. "
                f"Must be one of {HANDLE_MISSING_EVENTS}"
            )
        self.handle_missing_events = handle_missing_events

    def _compute_pep(
        self,
        *,
        heartbeats: pd.DataFrame,
        q_wave_onset_samples: pd.DataFrame,
        b_point_samples: pd.DataFrame,
        sampling_rate_hz: int,
    ) -> pd.DataFrame:
        pep_extraction_algo = PepExtraction()
        pep_extraction_algo.extract(
            heartbeats=heartbeats,
            q_wave_onset_samples=q_wave_onset_samples,
            b_point_samples=b_point_samples,
            sampling_rate_hz=sampling_rate_hz,
        )

        pep_results = pep_extraction_algo.pep_results_.copy()
        pep_results = self._add_invalid_pep_reason(pep_results, b_point_samples)
        pep_results = pep_results.convert_dtypes(infer_objects=True)

        return pep_results

    def _add_invalid_pep_reason(
        self,
        pep_results: pd.DataFrame,
        b_point_samples: pd.DataFrame,
    ) -> pd.DataFrame:
        # TODO add option to store multiple nan_reasons in one column?
        # add new column named "nan_reason" to pep_results to store the reason for the error
        pep_results = pep_results.assign(nan_reason=b_point_samples["nan_reason"].astype(object))

        neg_pep_idx = pep_results["pep_ms"] < 0
        if self.handle_negative_pep == "zero":
            pep_results.loc[neg_pep_idx, ["pep_sample", "pep_ms"]] = 0
            pep_results.loc[neg_pep_idx, "nan_reason"] = "negative_pep"
        elif self.handle_negative_pep == "nan":
            pep_results.loc[neg_pep_idx, ["pep_sample", "pep_ms"]] = np.nan
            pep_results.loc[neg_pep_idx, "nan_reason"] = "negative_pep"

        return pep_results
