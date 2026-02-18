"""Example Dataset for Pepbench.

Provides access to ECG and ICG signals, reference labels and reference heartbeats
for evaluation purposes.
"""
import zipfile
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from biopsykit.signals.ecg.preprocessing import EcgPreprocessingNeurokit
from biopsykit.signals.icg.preprocessing import IcgPreprocessingBandpass
from biopsykit.utils.dtypes import EcgRawDataFrame, IcgRawDataFrame, HeartbeatSegmentationDataFrame
from biopsykit.utils.file_handling import get_subject_dirs

from pepbench.datasets import BasePepDatasetWithAnnotations, BasePepDataset
from pepbench.datasets._helper import compute_reference_heartbeats
from pepbench.utils._types import path_t
from biopsykit.signals.ecg.segmentation import HeartbeatSegmentationNeurokit

#TODO: explain how to use this dataset, e.g., how to load the data, how to access the signals and reference labels, etc.

class WrapperDataset(BasePepDataset):
    """Example Dataset for Pepbench.

    This dataset serves as an example implementation of a dataset for the `pepbench` package.
    It provides access to ECG and ICG signals, along with reference labels and heartbeats for
    evaluation purposes.

    Parameters
    ----------
    example_file_path : str or Path
        Path to the example dataset zip file.
    groupby_cols : Sequence[str], optional
        Columns to group the dataset by. Default is None.
    subset_index : Sequence[str], optional
        Subset of the dataset index to use. Default is None.
    return_clean : bool, optional
        Whether to return cleaned signals. Default is True.

    Attributes
    ----------
    example_file_path : str or Path
        Path to the example dataset zip file.
    """

    _ecg: EcgRawDataFrame
    _icg: IcgRawDataFrame
    _sampling_rate_ecg: int
    _sampling_rate_icg: int

    _heartbeats: HeartbeatSegmentationDataFrame

    def __init__(
        self,
        ecg: EcgRawDataFrame,
        icg: IcgRawDataFrame,
        sampling_rate_ecg: int,
        sampling_rate_icg: int,
        groupby_cols: Sequence[str] | None = None,
        subset_index: Sequence[str] | None = None,
    ) -> None:
        """Initialize the :class:`~pepbench.datasets.ExampleDataset`.

        Parameters
        ----------
        ecg : EcgRawDataFrame
            ECG signal data as a BiopsyKit EcgRawDataFrame.
        icg : IcgRawDataFrame
            ICG signal data as a BiopsyKit IcgRawDataFrame.
        sampling_rate_ecg : int
            Sampling rate of the ECG signal in Hz.
        sampling_rate_icg : int
            Sampling rate of the ICG signal in Hz.
        groupby_cols : Sequence[str], optional
            Columns to group the dataset by. Default is None.
        subset_index : Sequence[str], optional
            Subset of the dataset index to use. Default is None.
        """
        self._ecg = ecg
        self._icg = icg
        self._sampling_rate_ecg = sampling_rate_ecg
        self._sampling_rate_icg = sampling_rate_icg
        super().__init__(
            groupby_cols=groupby_cols,
            subset_index=subset_index,
        )

    def create_index(self) -> pd.DataFrame:
        """Create the dataset index.

        The index contains one row per participant.

        Returns
        -------
        :class:`~pandas.DataFrame`
            Dataset index as a pandas DataFrame
        """


        index = pd.DataFrame(["data"], columns=["index"])
        return index

    @property
    def sampling_rate_ecg(self) -> int:
        """
        Return the sampling rate of the ECG signal.

        Returns
        -------
        int
            Sampling rate of the ECG signal in Hz.

        """
        return self._sampling_rate_ecg

    @property
    def sampling_rate_icg(self) -> int:
        """
        Return the sampling rate of the ICG signal.

        Returns
        -------
        int
            Sampling rate of the ICG signal in Hz.

        """
        return self._sampling_rate_icg

    @property
    def ecg(self) -> EcgRawDataFrame:
        """
        Return the ECG data for a single participant and phase.

        The property returns ECG data for a single participant and a single phase.
        If ``return_clean`` is True, the signal is cleaned before being returned.

        Returns
        -------
        :class:`~biopsykit.utils.dtypes.EcgRawDataFrame`
            ECG data as a BiopsyKit EcgRawDataFrame.

        Raises
        ------
        ValueError
            If accessed for multiple participants or phases.
        """
        return self._ecg

    @property
    def icg(self) -> IcgRawDataFrame:
        """
        Return the ICG data for a single participant and phase.

        The property returns ICG data for a single participant and a single phase.
        If ``return_clean`` is True, the signal is cleaned before being returned.

        Returns
        -------
        :class:`~biopsykit.utils.dtypes.IcgRawDataFrame`
            ICG data as a BiopsyKit IcgRawDataFrame.

        Raises
        ------
        ValueError
            If accessed for multiple participants or phases.
        """
        return self._icg

    @property
    def heartbeats(self) -> HeartbeatSegmentationDataFrame:
        """Heartbeat segmentation computed from the ECG signal.

        Uses :class:`~biopsykit.signals.ecg.segmentation.HeartbeatSegmentationNeurokit`
        to extract heartbeat borders and returns the algorithm's heartbeat list.

        Returns
        -------
        :class:`~pandas.DataFrame` or :class:`~biopsykit.utils.dtypes.HeartbeatSegmentationDataFrame`
            Heartbeat borders and related metadata (one row per heartbeat).
        """
        if self._heartbeats is not None:
            return self._heartbeats
        hb_algo = HeartbeatSegmentationNeurokit(variable_length=True)
        hb_algo.extract(ecg=self.ecg, sampling_rate_hz=self.sampling_rate_ecg)
        self._heartbeats = hb_algo.heartbeat_list_
        return self._heartbeats
