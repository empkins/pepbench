import numpy as np
from tpcp._dataset import DatasetT
from typing_extensions import Self

from pepbench.pipelines._base_pipeline import BasePepExtractionPipeline

__all__ = ["PepExtractionPipelineReferenceBPoints"]


class PepExtractionPipelineReferenceBPoints(BasePepExtractionPipeline):
    """tpcp Pipeline for PEP extraction that uses reference B-points for B-point detection."""

    def run(self, datapoint: DatasetT) -> Self:
        heartbeat_algo = self.heartbeat_segmentation_algo.clone()
        q_wave_algo = self.q_wave_algo.clone()
        outlier_algo = self.outlier_correction_algo.clone()

        reference_pep = datapoint.reference_pep
        fs_ecg = datapoint.sampling_rate_ecg

        ecg_data = datapoint.ecg_clean

        # extract heartbeats
        heartbeat_algo.extract(ecg=ecg_data, sampling_rate_hz=fs_ecg, handle_missing=self.handle_missing_events)
        heartbeats = heartbeat_algo.heartbeat_list_

        # run Q-wave extraction
        q_wave_algo.extract(
            ecg=ecg_data, heartbeats=heartbeats, sampling_rate_hz=fs_ecg, handle_missing=self.handle_missing_events
        )
        q_wave_onset_samples = q_wave_algo.points_

        # run Q-wave extraction
        b_point_samples = reference_pep[["b_point_sample"]].copy()
        # add nan_reason column to match extracted b-points
        b_point_samples["nan_reason"] = np.nan

        outlier_algo.correct_outlier(
            b_points=b_point_samples,
            c_points=None,
            sampling_rate_hz=0,
            handle_missing=self.handle_missing_events,
            handle_negative=self.handle_negative_pep,
        )
        b_point_samples_after_outlier = outlier_algo.points_

        pep_results = self._compute_pep(
            heartbeats=heartbeats,
            q_wave_onset_samples=q_wave_onset_samples,
            b_point_samples=b_point_samples,
            sampling_rate_hz=fs_ecg,
        )

        self.heartbeat_segmentation_results_ = heartbeats
        self.q_wave_results_ = q_wave_onset_samples
        self.c_point_results_ = None
        self.b_point_results_ = b_point_samples
        self.b_point_after_outlier_correction_results_ = b_point_samples_after_outlier
        self.pep_results_ = pep_results
        return self
