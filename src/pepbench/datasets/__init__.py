"""The datasets used for PEP extraction.

Currently, the following datasets are available in `pepbench`:
- `GuardianDataset`: Dataset from the Guardian project
- `EmpkinsDataset`: Dataset from the EmpkinS project

Both datasets provide a unified interface to the underlying data and enable easy use with the PEP extraction
algorithms and pipelines.

The PEP extraction pipelines require the datasets to provide the following attributes:
- `


Both are implemented as subclasses of `BaseUnifiedPepExtractionDataset` and provide the necessary methods to extract

"""

from pepbench.datasets._base_pep_extraction_dataset import (
    BasePepExtractionMixin,
    BaseUnifiedPepExtractionDataset,
    MetadataMixin,
    PepLabelMixin,
)
from pepbench.datasets.empkins import EmpkinsDataset
from pepbench.datasets.guardian import GuardianDataset

__all__ = [
    "BasePepExtractionMixin",
    "PepLabelMixin",
    "MetadataMixin",
    "BaseUnifiedPepExtractionDataset",
    "GuardianDataset",
    "EmpkinsDataset",
]
