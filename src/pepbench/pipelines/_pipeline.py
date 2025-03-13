from typing import Self, get_args

from biopsykit.signals._base_extraction import CanHandleMissingEventsMixin
from biopsykit.signals.pep._pep_extraction import NEGATIVE_PEP_HANDLING
from tpcp._dataset import DatasetT

__all__ = ["PepExtractionPipeline"]

from pepbench.pipelines._base_pipeline import BasePepExtractionPipeline


class PepExtractionPipeline(BasePepExtractionPipeline):
    """tpcp Pipeline for PEP extraction."""

    def run(self, datapoint: DatasetT) -> Self:
        if self.handle_negative_pep not in get_args(NEGATIVE_PEP_HANDLING):
            raise ValueError(
                f"Invalid value for 'handle_negative_pep': {self.handle_negative_pep}. "
                f"Must be one of {NEGATIVE_PEP_HANDLING}"
            )

        heartbeat_algo = self.heartbeat_segmentation_algo.clone()
        q_peak_algo = self.q_peak_algo.clone()
        c_point_algo = self.c_point_algo.clone()
        b_point_algo = self.b_point_algo.clone()
        outlier_algo = self.outlier_correction_algo.clone()

        fs_ecg = datapoint.sampling_rate_ecg
        fs_icg = datapoint.sampling_rate_icg

        ecg_data = datapoint.ecg_clean
        icg_data = datapoint.icg_clean

        # set handle_missing parameter for all algorithms
        if self.handle_missing_events is not None:
            for algo in (heartbeat_algo, q_peak_algo, c_point_algo, b_point_algo, outlier_algo):
                if isinstance(algo, CanHandleMissingEventsMixin):
                    # this overwrites the default value of the handle_missing parameter
                    algo.set_params(handle_missing_events=self.handle_missing_events)

        # extract heartbeats
        heartbeat_algo.extract(ecg=ecg_data, sampling_rate_hz=fs_ecg)
        heartbeats = heartbeat_algo.heartbeat_list_

        # run Q-peak extraction
        q_peak_algo.extract(ecg=ecg_data, heartbeats=heartbeats, sampling_rate_hz=fs_ecg)
        q_peak_samples = q_peak_algo.points_

        # run C-point extraction
        c_point_algo.extract(icg=icg_data, heartbeats=heartbeats, sampling_rate_hz=fs_icg)

        # run B-point extraction
        b_point_algo.extract(
            icg=icg_data,
            heartbeats=heartbeats,
            c_points=c_point_algo.points_,
            sampling_rate_hz=fs_icg,
        )

        outlier_algo.correct_outlier(
            b_points=b_point_algo.points_,
            c_points=c_point_algo.points_,
            sampling_rate_hz=fs_icg,
            handle_negative=self.handle_negative_pep,
        )
        b_point_samples_after_outlier = outlier_algo.points_

        pep_results = self._compute_pep(
            heartbeats=heartbeats,
            q_peak_samples=q_peak_samples,
            b_point_samples=b_point_samples_after_outlier,
            sampling_rate_hz=fs_icg,
        )

        self.heartbeat_segmentation_results_ = heartbeats
        self.q_peak_results_ = q_peak_samples
        self.c_point_results_ = c_point_algo.points_
        self.b_point_results_ = b_point_algo.points_
        self.b_point_after_outlier_correction_results_ = b_point_samples_after_outlier
        self.pep_results_ = pep_results
        return self
