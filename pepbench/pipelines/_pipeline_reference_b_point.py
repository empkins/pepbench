from tpcp._dataset import DatasetT
from typing_extensions import Self

from pepbench.pipelines._base_pipeline import BasePepExtractionPipeline

__all__ = ["PepExtractionPipelineReferenceBPoints"]


class PepExtractionPipelineReferenceBPoints(BasePepExtractionPipeline):
    """tpcp Pipeline for PEP extraction that uses reference B-points for B-point detection."""

    def run(self, datapoint: DatasetT) -> Self:
        heartbeat_algo = self.heartbeat_segmentation_algo.clone()
        q_wave_algo = self.q_wave_algo.clone()

        reference_pep = datapoint.reference_pep
        fs_ecg = datapoint.sampling_rate_ecg

        ecg_data = datapoint.ecg_clean

        # extract heartbeats
        heartbeat_algo.extract(ecg=ecg_data, sampling_rate_hz=fs_ecg)
        heartbeats = heartbeat_algo.heartbeat_list_

        # run Q-wave extraction
        q_wave_algo.extract(ecg=ecg_data, heartbeats=heartbeats, sampling_rate_hz=fs_ecg)
        q_wave_onset_samples = q_wave_algo.points_

        # run Q-wave extraction
        b_point_samples = reference_pep[["b_point_sample"]].copy()

        pep_results = self._compute_pep(
            reference_pep=reference_pep,
            heartbeats=heartbeats,
            q_wave_onset_samples=q_wave_onset_samples,
            b_point_samples=b_point_samples,
            sampling_rate_hz=fs_ecg,
        )

        self.pep_results_ = pep_results
        return self
