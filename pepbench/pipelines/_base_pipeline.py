from typing import Optional

import numpy as np
import pandas as pd
from biopsykit.signals.ecg.event_extraction import BaseEcgExtraction
from biopsykit.signals.ecg.segmentation import BaseHeartbeatSegmentation
from biopsykit.signals.icg.event_extraction import BaseBPointExtraction, CPointExtractionScipyFindPeaks
from biopsykit.signals.icg.outlier_correction import BaseOutlierCorrection
from biopsykit.signals.pep import PepExtraction
from tpcp import Pipeline, Parameter

from pepbench.pipelines._helper import merge_pep_with_reference


class _BasePepExtractionPipeline(Pipeline):

    heartbeat_segmentation_algo: Parameter[BaseHeartbeatSegmentation]
    q_wave_algo: Parameter[BaseEcgExtraction]
    b_point_algo: Parameter[BaseBPointExtraction]
    outlier_correction_algo: Parameter[BaseOutlierCorrection]
    snap_negative_pep_to_nan: Parameter[Optional[bool]]

    pep_results_: pd.DataFrame

    def __init__(
        self,
        *,
        heartbeat_segmentation_algo: BaseHeartbeatSegmentation,
        q_wave_algo: BaseEcgExtraction,
        b_point_algo: BaseBPointExtraction,
        outlier_correction_algo: BaseOutlierCorrection,
        snap_negative_pep_to_nan: Optional[bool] = False,
    ):
        self.heartbeat_segmentation_algo = heartbeat_segmentation_algo
        self.q_wave_algo = q_wave_algo
        self.b_point_algo = b_point_algo
        self.c_point_algo = CPointExtractionScipyFindPeaks()
        self.outlier_correction_algo = outlier_correction_algo
        self.pep_extraction_algo = PepExtraction()
        self.snap_negative_pep_to_nan = snap_negative_pep_to_nan

    def _compute_pep(
        self,
        *,
        reference_pep: pd.DataFrame,
        heartbeats: pd.DataFrame,
        q_wave_onset_samples: pd.DataFrame,
        b_point_samples: pd.DataFrame,
        sampling_rate_hz: int,
    ):
        pep_extraction_algo = PepExtraction()
        pep_extraction_algo.extract(
            heartbeats=heartbeats,
            q_wave_onset_samples=q_wave_onset_samples,
            b_point_samples=b_point_samples,
            sampling_rate_hz=sampling_rate_hz,
        )

        pep_results = pep_extraction_algo.pep_results_.copy()
        # add new column named "invalid_reason" to pep_results to store the reason for the error
        pep_results = pep_results.assign(invalid_reason=np.nan)
        if self.snap_negative_pep_to_nan:
            neg_pep_idx = pep_results["pep_ms"] < 0
            pep_results.loc[neg_pep_idx, ["pep_sample", "pep_ms"]] = np.nan
            pep_results.loc[neg_pep_idx, "invalid_reason"] = "negative_pep"

        pep_results = merge_pep_with_reference(pep_results, reference_pep)
        pep_results = pep_results.convert_dtypes(infer_objects=True)
        return pep_results
