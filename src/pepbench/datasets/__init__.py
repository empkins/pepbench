"""Datasets for beat-to-beat PEP extraction.

Currently, two following datasets are available in `pepbench` (see below). Both datasets provide a unified interface
to the underlying data and enable easy use with the PEP extraction algorithms and pipelines.

The PEP extraction pipelines require the datasets to provide the following attributes:

    - `ecg`: Access to the ECG signal data
    - `icg`: Access to the ICG signal data
    - `sampling_rate_ecg`: Sampling rate of the ECG signal
    - `sampling_rate_icg`: Sampling rate of the ICG signal

The provided dataset classes are implemented as subclasses of :class:`pepbench.datasets.BasePepDataset`.
This interface provides these necessary attributes and methods to extract PEP data from the underlying signals.

If the datasets should be used with labeled reference data, i.e., for evaluating the PEP extraction algorithms,
the datasets should also provide the following attributes:

    - `reference_pep`: Reference PEP data, which should have the following columns:
        - `heartbeat_start_sample`: Index of the heartbeat start sample, relative to the beginning of the signal
        - `heartbeat_end_sample`: Index of the heartbeat end sample, relative to the beginning of the signal
        - `q_peak_sample`: Index of the Q-peak sample, relative to the beginning of the signal
        - `b_point_sample`: Index of the B-point sample, relative to the beginning of the signal
    - `reference_heartbeats`: Reference heartbeats, which should have the following columns:
        - `heartbeat_start_sample`: Index of the heartbeat start sample, relative to the beginning of the signal
        - `heartbeat_end_sample`: Index of the heartbeat end sample, relative to the beginning of the signal
    - `reference_labels_ecg`: Reference labels for the ECG data
    - `reference_labels_icg`: Reference labels for the ICG data


"""

from pepbench.datasets._base_pep_extraction_dataset import (
    BasePepDataset,
    BasePepDatasetWithAnnotations,
    MetadataMixin,
    PepLabelMixin,
)
from pepbench.datasets.empkins import EmpkinsDataset
from pepbench.datasets.guardian import GuardianDataset
from pepbench.datasets.rebeat_icg import ReBeatIcgDataset
from pepbench.datasets.time_window_icg import TimeWindowIcgDataset

__all__ = [
    "BasePepDataset",
    "BasePepDatasetWithAnnotations",
    "EmpkinsDataset",
    "GuardianDataset",
    "MetadataMixin",
    "PepLabelMixin",
    "ReBeatIcgDataset",
    "TimeWindowIcgDataset",
]
