from typing_extensions import Self

import pandas as pd
from tpcp import Pipeline, Parameter
from tpcp._dataset import DatasetT

from biopsykit.signals.ecg.event_extraction import BaseEcgExtraction
from biopsykit.signals.ecg.segmentation import BaseHeartbeatSegmentation
from biopsykit.signals.icg.event_extraction import BaseBPointExtraction, CPointExtractionScipyFindPeaks
from biopsykit.signals.icg.outlier_correction import BaseOutlierCorrection
from biopsykit.signals.pep import PepExtraction

__all__ = ["PepExtractionPipeline"]


class PepExtractionPipeline(Pipeline):
    """Pipeline for PEP extraction"""

    heartbeat_segmentation_algo: Parameter[BaseHeartbeatSegmentation]
    q_wave_algo: Parameter[BaseEcgExtraction]
    b_point_algo: Parameter[BaseBPointExtraction]
    outlier_correction_algo: Parameter[BaseOutlierCorrection]
    pep_extraction_algo: Parameter[PepExtraction]

    pep_results_: pd.DataFrame

    def __init__(
        self,
        heartbeat_segmentation_algo: BaseHeartbeatSegmentation,
        q_wave_algo: BaseEcgExtraction,
        b_point_algo: BaseBPointExtraction,
        outlier_correction_algo: BaseOutlierCorrection,
    ):
        self.heartbeat_segmentation_algo = heartbeat_segmentation_algo
        self.q_wave_algo = q_wave_algo
        self.b_point_algo = b_point_algo
        self.c_point_algo = CPointExtractionScipyFindPeaks()
        self.outlier_correction_algo = outlier_correction_algo
        self.pep_extraction_algo = PepExtraction()

    def run(self, datapoint: DatasetT) -> Self:
        heartbeat_algo = self.heartbeat_segmentation_algo.clone()
        q_wave_algo = self.q_wave_algo.clone()
        c_point_algo = self.c_point_algo.clone()
        b_point_algo = self.b_point_algo.clone()
        outlier_algo = self.outlier_correction_algo.clone()
        pep_extraction_algo = self.pep_extraction_algo.clone()

        fs_ecg = datapoint.sampling_rate_ecg
        fs_icg = datapoint.sampling_rate_icg

        ecg_data = datapoint.ecg_clean
        icg_data = datapoint.icg_clean

        # extract heartbeats
        heartbeat_algo.extract(ecg=ecg_data, sampling_rate_hz=fs_ecg)
        heartbeats = heartbeat_algo.heartbeat_list_

        reference_pep = datapoint.reference_pep

        # run Q-wave extraction
        q_wave_algo.extract(ecg=ecg_data, heartbeats=heartbeats, sampling_rate_hz=fs_ecg)

        # run C-point extraction
        c_point_algo.extract(icg=icg_data, heartbeats=heartbeats, sampling_rate_hz=fs_icg)

        # run B-point extraction
        b_point_algo.extract(
            icg=icg_data, heartbeats=heartbeats, c_points=c_point_algo.points_, sampling_rate_hz=fs_icg
        )

        outlier_algo.correct_outlier(
            b_points=b_point_algo.points_, c_points=c_point_algo.points_, sampling_rate_hz=fs_icg
        )

        pep_extraction_algo.extract(
            heartbeats=heartbeats,
            q_wave_onset_samples=q_wave_algo.points_,
            b_point_samples=b_point_algo.points_,
            sampling_rate_hz=fs_icg,
        )
        pep_results = pep_extraction_algo.pep_results_

        self.pep_results_ = pep_results

        return self
