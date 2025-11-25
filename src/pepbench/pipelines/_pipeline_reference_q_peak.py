"""
PEP extraction pipeline using reference Q-peaks.

This module implements a validation pipeline that computes the pre-ejection period (PEP)
by using reference Q-peak annotations. The pipeline extracts heartbeats from ECG,
runs C- and B-point extraction on ICG, aligns extracted heartbeats to reference annotations,
applies outlier correction to B-point samples, and computes per-beat PEP values.

The pipeline is intended for controlled evaluation of B-point extraction and outlier
correction algorithms when reference Q-peaks are available in the dataset.
"""

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

    This pipeline validates different B-point extraction algorithms by computing the
    pre-ejection period (PEP) using reference Q-peak annotations from labeled datasets.
    It performs heartbeat segmentation, heartbeat matching against reference annotations,
    Q-peak alignment using reference annotations, runs C-point and B-point extraction,
    applies outlier correction for B-point samples, and computes final PEP values.

    Parameters
    ----------
    %(base_parameters)s

    Other Parameters
    ----------------
    %(datapoint_pipeline_labeled)s

    Attributes
    ----------
    %(attributes)s

    """

    def run(self, datapoint: DatasetT) -> Self:
        """Run the pipeline on the given datapoint.

        The pipeline executes the following steps:
        - Validate configuration values (for example, that `handle_negative_pep` is valid).
        - Clone configured algorithms and optionally set `handle_missing_events` on algorithms
          that support missing-event handling.
        - Extract heartbeats from the ECG signal.
        - Match extracted heartbeats to the reference heartbeat annotations (tolerance 100 ms).
        - Select true-positive matches and align reference Q-peaks with extracted heartbeat indices.
        - Run C-point and B-point extraction for matched heartbeats, add `nan_reason` column to
          aligned Q-peak table.
        - Mark and prepare B-point samples for outlier correction, then apply the outlier
          correction algorithm.
        - Compute PEP values using aligned Q- and B-points and store results on the pipeline.

        Parameters
        ----------
        datapoint : BasePepDatasetWithAnnotations
            Labeled datapoint that provides ECG data, sampling rates, reference heartbeat
            annotations, and reference PEP/Q-peak annotations.

        Returns
        -------
        Self
            The pipeline instance with result attributes populated (see class attributes).

        Raises
        ------
        ValueError
            If `handle_negative_pep` is not a valid value from the allowed options.

        Notes
        -----
        - Heartbeat matching uses a default tolerance of 100 ms.
        - False positives and negatives in heartbeat matching are currently not corrected
          automatically by this pipeline; only true-positive matches are used to compute PEP.
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
