import zipfile
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from biopsykit.signals.ecg.preprocessing import EcgPreprocessingNeurokit
from biopsykit.signals.icg.preprocessing import IcgPreprocessingBandpass
from biopsykit.utils.dtypes import EcgRawDataFrame, IcgRawDataFrame
from biopsykit.utils.file_handling import get_subject_dirs

from pepbench.datasets import BasePepDatasetWithAnnotations
from pepbench.datasets._helper import compute_reference_heartbeats
from pepbench.utils._types import path_t

HERE = Path(__file__).parent
EXAMPLE_DATA_PATH = HERE.joinpath("../../../example_data")


class ExampleDataset(BasePepDatasetWithAnnotations):
    example_file_path: path_t

    def __init__(
        self,
        example_file_path: path_t,
        groupby_cols: Sequence[str] | None = None,
        subset_index: Sequence[str] | None = None,
        *,
        return_clean: bool = True,
    ) -> None:
        self.example_file_path = example_file_path
        # unzip the example dataset
        with zipfile.ZipFile(str(self.example_file_path)) as zf:
            zf.extractall(EXAMPLE_DATA_PATH)
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index, return_clean=return_clean)

    def create_index(self) -> pd.DataFrame:
        participant_ids = [
            participant_dir.name for participant_dir in get_subject_dirs(EXAMPLE_DATA_PATH, "VP_[0-9]{3}")
        ]

        index = pd.DataFrame(participant_ids, columns=["participant"])
        return index

    @property
    def sampling_rate_ecg(self) -> int:
        return 500

    @property
    def sampling_rate_icg(self) -> int:
        return 500

    @property
    def ecg(self) -> EcgRawDataFrame:
        if not self.is_single(None):
            raise ValueError("ECG data can only be accessed for a single participant and a single phase!")
        data = self._load_data("ecg")
        if self.return_clean:
            algo = EcgPreprocessingNeurokit()
            algo.clean(ecg=data, sampling_rate_hz=self.sampling_rate_ecg)
            return algo.ecg_clean_
        return data

    @property
    def icg(self) -> IcgRawDataFrame:
        if not self.is_single(None):
            raise ValueError("ICG data can only be accessed for a single participant and a single phase!")
        data = self._load_data("icg")
        if self.return_clean:
            algo = IcgPreprocessingBandpass()
            algo.clean(icg=data, sampling_rate_hz=self.sampling_rate_icg)
            return algo.icg_clean_
        return data

    def _load_data(self, data_type: str) -> pd.DataFrame:
        p_id = self.index["participant"][0]
        data = pd.read_csv(
            EXAMPLE_DATA_PATH.joinpath(f"{p_id}/{p_id.lower()}_{data_type}_data.csv"),
            index_col=0,
        )
        data.index = pd.DatetimeIndex(data.index)
        return data

    @property
    def reference_labels_ecg(self) -> pd.DataFrame:
        """Return the reference labels for the ECG signal.

        Returns
        -------
        :class:`~pandas.DataFrame`
            Reference labels for the ECG signal as a pandas DataFrame

        """
        return self._load_reference_labels("ECG")

    @property
    def reference_labels_icg(self) -> pd.DataFrame:
        """Return the reference labels for the ICG signal.

        Returns
        -------
        :class:`~pandas.DataFrame`
            Reference labels for the ICG signal as a pandas DataFrame

        """
        return self._load_reference_labels("ICG")

    def _load_reference_labels(self, channel: str) -> pd.DataFrame:
        participant = self.index["participant"][0]

        if not (self.is_single(None)):
            raise ValueError("Reference data can only be accessed for a single participant.")

        file_path = EXAMPLE_DATA_PATH.joinpath(
            f"{participant}/reference_labels/{participant}_reference_labels_{channel}.csv"
        )
        reference_data = pd.read_csv(file_path)
        reference_data = reference_data.set_index(["heartbeat_id", "channel", "label"])
        # absolute and relative sample are the same in the example data
        reference_data = reference_data.assign(sample_absolute=reference_data["sample_relative"])
        return reference_data

    @property
    def reference_heartbeats(self) -> pd.DataFrame:
        """Return the reference heartbeats.

        Returns
        -------
        :class:`~pandas.DataFrame`
            Reference heartbeats as a pandas DataFrame

        """
        return self._load_reference_heartbeats()

    def _load_reference_heartbeats(self) -> pd.DataFrame:
        reference_ecg = self.reference_labels_ecg
        reference_heartbeats = reference_ecg.reindex(["heartbeat"], level="channel")
        reference_heartbeats = compute_reference_heartbeats(reference_heartbeats)
        return reference_heartbeats
