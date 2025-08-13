"""Pipelines to extract PEPs from a given dataset.

This module contains different pipelines to extract PEPs from a given dataset. The pipelines are based on the
`tpcp` library and provide a standardized interface for PEP extraction.

The standard pipeline for most use cases is the :class:`~pepbench.pipelines.PepExtractionPipeline`. This pipeline
uses a combination of Q-peak and B-point detection algorithms (with an optional outlier correction step) to extract
PEPs from ECG and ICG data.

The other pipelines (:class:`~pepbench.pipelines.PepExtractionPipelineReferenceBPoints` and
:class:`~pepbench.pipelines.PepExtractionPipelineReferenceQPeaks`) are specialized pipelines that use reference
B-points or Q-peaks, respectively, to extract PEPs. These pipelines are primarily useful for benchmarking and
validation purposes.

"""

from pepbench.pipelines._base_pipeline import BasePepExtractionPipeline
from pepbench.pipelines._pipeline import PepExtractionPipeline
from pepbench.pipelines._pipeline_reference_b_point import PepExtractionPipelineReferenceBPoints
from pepbench.pipelines._pipeline_reference_q_peak import PepExtractionPipelineReferenceQPeaks

__all__ = [
    "BasePepExtractionPipeline",
    "PepExtractionPipeline",
    "PepExtractionPipelineReferenceBPoints",
    "PepExtractionPipelineReferenceQPeaks",
]
