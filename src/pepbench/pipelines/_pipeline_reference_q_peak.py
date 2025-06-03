from typing import get_args

import pandas as pd
from biopsykit.signals._base_extraction import CanHandleMissingEventsMixin
from biopsykit.signals.pep._pep_extraction import NEGATIVE_PEP_HANDLING
from tpcp._dataset import DatasetT
from typing_extensions import Self

from pepbench.heartbeat_matching import match_heartbeat_lists
from pepbench.pipelines._base_pipeline import BasePepExtractionPipeline, base_pep_pipeline_docfiller

__all__ = ["PepExtractionPipelineReferenceQPeaks"]


@base_pep_pipeline_docfiller
class PepExtractionPipelineReferenceQPeaks(BasePepExtractionPipeline):
    """`tpcp` Pipeline for PEP extraction that uses reference Q-peaks for Q-peak detection.

    This pipeline is used to validate different B-point extraction algorithms and computing the PEP using reference
    Q-peaks.

    %(base_parameters)s

    Other Parameters
    ----------------
    %(datapoint_pipeline_labeled)s

    %(attributes)s

    """

    def run(self, datapoint: DatasetT) -> Self:
        """Run the pipeline on the given datapoint.

        The pipeline will extract PEP from the given datapoint using the specified algorithms. The results will be
        stored in the attributes of the class.

        Parameters
        ----------
        %(datapoint_pipeline_labeled)s

        """
        if self.handle_negative_pep not in get_args(NEGATIVE_PEP_HANDLING):
            raise ValueError(
                f"Invalid value for 'handle_negative_pep': {self.handle_negative_pep}. "
                f"Must be one of {NEGATIVE_PEP_HANDLING}"
            )

        heartbeat_algo = self.heartbeat_segmentation_algo.clone()
        c_point_algo = self.c_point_algo.clone()
        b_point_algo = self.b_point_algo.clone()
        outlier_algo = self.outlier_correction_algo.clone()

        reference_pep = datapoint.reference_pep
        fs_ecg = datapoint.sampling_rate_ecg
        fs_icg = datapoint.sampling_rate_icg

        ecg_data = datapoint.ecg
        icg_data = datapoint.icg

        # set handle_missing parameter for all algorithms
        if self.handle_missing_events is not None:
            for algo in (heartbeat_algo, c_point_algo, b_point_algo, outlier_algo):
                if isinstance(algo, CanHandleMissingEventsMixin):
                    # this overwrites the default value of the handle_missing parameter
                    algo.set_params(handle_missing_events=self.handle_missing_events)

        # extract heartbeats
        heartbeat_algo.extract(ecg=ecg_data, sampling_rate_hz=fs_ecg)
        heartbeats = heartbeat_algo.heartbeat_list_

        heartbeat_matching = match_heartbeat_lists(
            heartbeats_reference=datapoint.reference_heartbeats,
            heartbeats_extracted=heartbeats,
            tolerance_ms=100,
            sampling_rate_hz=datapoint.sampling_rate_ecg,
        )

        # run C-point extraction
        c_point_algo.extract(icg=icg_data, heartbeats=heartbeats, sampling_rate_hz=fs_icg)

        # run B-point extraction
        b_point_algo.extract(
            icg=icg_data, heartbeats=heartbeats, c_points=c_point_algo.points_, sampling_rate_hz=fs_icg
        )

        # TODO: handle false negatives and false positives, i.e. heartbeats that are not matched
        tp_matches = heartbeat_matching.query("match_type == 'tp'")

        # run Q-Peak extraction
        q_peak_samples = reference_pep[["q_peak_sample"]].copy()
        q_peak_samples_tp = q_peak_samples.loc[tp_matches["heartbeat_id_reference"]]

        b_point_samples = b_point_algo.points_
        b_point_samples_tp = b_point_samples.loc[tp_matches["heartbeat_id"]]
        c_point_samples = c_point_algo.points_
        c_point_samples_tp = c_point_samples.loc[tp_matches["heartbeat_id"]]

        tp_matches = tp_matches.set_index("heartbeat_id")

        q_peak_samples_tp.index = tp_matches.index
        c_point_samples_tp.index = tp_matches.index
        b_point_samples_tp.index = tp_matches.index

        # add nan_reason column to match extracted b-points
        q_peak_samples_tp["nan_reason"] = pd.NA

        outlier_algo.correct_outlier(b_points=b_point_samples_tp, c_points=c_point_samples_tp, sampling_rate_hz=fs_icg)
        b_point_samples_after_outlier = outlier_algo.points_

        pep_results = self._compute_pep(
            heartbeats=heartbeats,
            q_peak_samples=q_peak_samples_tp,
            b_point_samples=b_point_samples_after_outlier,
            sampling_rate_hz=fs_icg,
        )

        self.heartbeat_segmentation_results_ = heartbeats
        self.q_peak_results_ = q_peak_samples_tp
        self.c_point_results_ = c_point_samples_tp
        self.b_point_results_ = b_point_samples_tp
        self.b_point_after_outlier_correction_results_ = b_point_samples_after_outlier
        self.pep_results_ = pep_results
        return self
