from typing import get_args

import pandas as pd
from biopsykit.signals._base_extraction import CanHandleMissingEventsMixin
from biopsykit.signals.pep._pep_extraction import NEGATIVE_PEP_HANDLING
from typing_extensions import Self

from pepbench.datasets import BasePepDatasetWithAnnotations
from pepbench.heartbeat_matching import match_heartbeat_lists
from pepbench.pipelines._base_pipeline import BasePepExtractionPipeline, base_pep_pipeline_docfiller

__all__ = ["PepExtractionPipelineReferenceBPoints"]


@base_pep_pipeline_docfiller
class PepExtractionPipelineReferenceBPoints(BasePepExtractionPipeline):
    """`tpcp` Pipeline for PEP extraction that uses reference B-points for B-point detection.

    This pipeline is used to validate different Q-peak extraction algorithms and computing the PEP using reference
    B-points.

    %(base_parameters)s

    Other Parameters
    ----------------
    %(datapoint_pipeline_labeled)s

    %(attributes)s

    """

    @base_pep_pipeline_docfiller
    def run(self, datapoint: BasePepDatasetWithAnnotations) -> Self:
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
        q_peak_algo = self.q_peak_algo.clone()
        outlier_algo = self.outlier_correction_algo.clone()

        reference_pep = datapoint.reference_pep
        fs_ecg = datapoint.sampling_rate_ecg

        ecg_data = datapoint.ecg

        # set handle_missing parameter for all algorithms
        if self.handle_missing_events is not None:
            for algo in (heartbeat_algo, q_peak_algo, outlier_algo):
                if isinstance(algo, CanHandleMissingEventsMixin):
                    # this overwrites the default value of the handle_missing parameter
                    algo.set_params(handle_missing_events=self.handle_missing_events)

        # extract heartbeats
        heartbeat_algo.extract(ecg=ecg_data, sampling_rate_hz=fs_ecg)
        heartbeats = heartbeat_algo.heartbeat_list_

        # run Q-peak extraction
        q_peak_algo.extract(ecg=ecg_data, heartbeats=heartbeats, sampling_rate_hz=fs_ecg)
        q_peak_samples = q_peak_algo.points_

        heartbeat_matching = match_heartbeat_lists(
            heartbeats_reference=datapoint.reference_heartbeats,
            heartbeats_extracted=heartbeats,
            tolerance_ms=100,
            sampling_rate_hz=datapoint.sampling_rate_ecg,
        )

        # TODO: handle false negatives and false positives, i.e. heartbeats that are not matched
        tp_matches = heartbeat_matching.query("match_type == 'tp'")
        q_peak_samples_tp = q_peak_samples.loc[tp_matches["heartbeat_id"]]
        b_point_samples = reference_pep[["b_point_sample", "nan_reason"]].copy()
        b_point_samples_tp = b_point_samples.loc[tp_matches["heartbeat_id_reference"]]

        tp_matches = tp_matches.set_index("heartbeat_id")

        q_peak_samples_tp.index = tp_matches.index
        b_point_samples_tp.index = tp_matches.index

        # add nan_reason column to match extracted b-points
        b_point_samples_tp["nan_reason"] = pd.NA

        outlier_algo.correct_outlier(
            b_points=b_point_samples_tp,
            c_points=None,
            sampling_rate_hz=0,
            handle_missing=self.handle_missing_events,
            handle_negative=self.handle_negative_pep,
        )
        b_point_samples_after_outlier = outlier_algo.points_

        pep_results = self._compute_pep(
            heartbeats=heartbeats,
            q_peak_samples=q_peak_samples_tp,
            b_point_samples=b_point_samples_after_outlier,
            sampling_rate_hz=fs_ecg,
        )

        self.heartbeat_segmentation_results_ = heartbeats
        self.q_peak_results_ = q_peak_samples
        self.c_point_results_ = None
        self.b_point_results_ = b_point_samples
        self.b_point_after_outlier_correction_results_ = b_point_samples_after_outlier
        self.pep_results_ = pep_results
        return self
