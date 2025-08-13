"""Tools for handling and analyzing annotation data.

This package includes functions for:
- Computing annotation differences,
- Loading and matching annotations from datasets,
- Normalizing annotations to heartbeat start times,
- Statistical analysis of annotation differences, including binning and computing descriptive statistics,
- Calculating the Intraclass Correlation Coefficient (ICC), and
- Integrating annotation agreement information into results dataframes.

"""

from pepbench.annotations._annotations import (
    compute_annotation_differences,
    load_annotations_from_dataset,
    match_annotations,
    normalize_annotations_to_heartbeat_start,
)

__all__ = [
    "compute_annotation_differences",
    "load_annotations_from_dataset",
    "match_annotations",
    "normalize_annotations_to_heartbeat_start",
]
