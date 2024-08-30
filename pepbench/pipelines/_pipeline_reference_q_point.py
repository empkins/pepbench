from tpcp._dataset import DatasetT
from typing_extensions import Self

from pepbench.pipelines._base_pipeline import BasePepExtractionPipeline

__all__ = ["PepExtractionPipelineReferenceQPoints"]


class PepExtractionPipelineReferenceQPoints(BasePepExtractionPipeline):
    """tpcp Pipeline for PEP extraction that uses reference Q-points for Q-wave onset detection."""

    def run(self, datapoint: DatasetT) -> Self:
        heartbeat_algo = self.heartbeat_segmentation_algo.clone()
        c_point_algo = self.c_point_algo.clone()
        b_point_algo = self.b_point_algo.clone()
        outlier_algo = self.outlier_correction_algo.clone()

        reference_pep = datapoint.reference_pep
        fs_ecg = datapoint.sampling_rate_ecg
        fs_icg = datapoint.sampling_rate_icg

        ecg_data = datapoint.ecg_clean
        icg_data = datapoint.icg_clean

        # extract heartbeats
        heartbeat_algo.extract(ecg=ecg_data, sampling_rate_hz=fs_ecg)
        heartbeats = heartbeat_algo.heartbeat_list_

        # run Q-wave extraction
        q_wave_onset_samples = reference_pep[["q_wave_onset_sample"]].copy()

        # run C-point extraction
        c_point_algo.extract(icg=icg_data, heartbeats=heartbeats, sampling_rate_hz=fs_icg)

        # run B-point extraction
        b_point_algo.extract(
            icg=icg_data, heartbeats=heartbeats, c_points=c_point_algo.points_, sampling_rate_hz=fs_icg
        )

        outlier_algo.correct_outlier(
            b_points=b_point_algo.points_, c_points=c_point_algo.points_, sampling_rate_hz=fs_icg
        )
        b_point_samples = outlier_algo.points_

        pep_results = self._compute_pep(
            reference_pep=reference_pep,
            heartbeats=heartbeats,
            q_wave_onset_samples=q_wave_onset_samples,
            b_point_samples=b_point_samples,
            sampling_rate_hz=fs_icg,
        )

        self.pep_results_ = pep_results
        return self
