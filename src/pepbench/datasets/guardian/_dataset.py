from collections.abc import Sequence
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import ClassVar

import pandas as pd
from biopsykit.metadata import bmi
from biopsykit.signals.ecg.preprocessing import EcgPreprocessingNeurokit
from biopsykit.signals.ecg.segmentation import HeartbeatSegmentationNeurokit
from biopsykit.signals.icg.preprocessing import IcgPreprocessingBandpass

from pepbench.datasets import BasePepDatasetWithAnnotations
from pepbench.datasets._base_pep_extraction_dataset import MetadataMixin, base_pep_extraction_docfiller
from pepbench.datasets._helper import compute_reference_heartbeats, load_labeling_borders
from pepbench.datasets.guardian._helper import _load_tfm_data
from pepbench.utils._types import path_t

__all__ = ["GuardianDataset"]


_cached_get_tfm_data = lru_cache(maxsize=4)(_load_tfm_data)
# cache_dir = "./cachedir"
# memory = Memory(location=cache_dir, verbose=0)
# _cached_get_tfm_data = memory.cache(_load_tfm_data)


@base_pep_extraction_docfiller
class GuardianDataset(BasePepDatasetWithAnnotations, MetadataMixin):
    """Dataset class for the Guardian Dataset.

    This class is the ``tpcp`` dataset class for the Guardian dataset. It provides access to the Task Force Monitor
    (TFM) data (for ECG and ICG), the reference annotations for the ECG and ICG annotations, as well as metadata
    like age, gender, and BMI.

    Attributes
    ----------
    %(base_attributes_pep)s
    %(base_attributes_pep_label)s
    %(base_attributes_metadata)s
    timelog : :class:`~pandas.DataFrame`
        Timelog data, indicating the start and end of each experimental phase, as a pandas DataFrame.

    """

    base_path: Path
    use_cache: bool

    SAMPLING_RATES: ClassVar[dict[str, int]] = {"ecg_1": 500, "ecg_2": 500, "icg_der": 500}
    PHASES: ClassVar[tuple[str, ...]] = ["Pause", "Valsalva", "HoldingBreath", "TiltUp", "TiltLevel"]

    GENDER_MAPPING: ClassVar[dict[str, str]] = {"M": "Male", "F": "Female"}

    SUBSET_NO_RECORDED_DATA = (
        ("GDN0006", "HoldingBreath"),
        ("GDN0009", "HoldingBreath"),
        ("GDN0010", "Valsalva"),
        ("GDN0017", "Pause"),
        ("GDN0018", "TiltLevel"),
        ("GDN0020", "TiltUp"),
        ("GDN0022", "TiltUp"),
        ("GDN0024", "TiltLevel"),
        ("GDN0025", "Valsalva"),
        ("GDN0028", "TiltUp"),
        ("GDN0030", "Pause"),
        ("GDN0030", "TiltUp"),
    )
    SUBSET_NOISY_DATA = (
        ("GDN0025", "TiltUp"),
        ("GDN0025", "TiltLevel"),
    )

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Sequence[str] | None = None,
        subset_index: Sequence[str] | None = None,
        *,
        return_clean: bool = True,
        exclude_no_recorded_data: bool = True,
        exclude_noisy_data: bool = True,
        use_cache: bool = True,
        only_labeled: bool = False,
        label_type: str = "rater_01",
    ) -> None:
        """Initialize a new ``GuardianDataset`` instance.

        Parameters
        ----------
        base_path : :class:`~pathlib.Path` or str
            Path to the root directory of the Guardian dataset.
        return_clean : bool
            Whether to return the preprocessed/cleaned ECG and ICG data when accessing the respective properties.
            Default: ``True``.
        exclude_no_recorded_data : bool, optional
            Whether to exclude participants with no recorded data. Default: ``True``.
        exclude_noisy_data : bool, optional
            Whether to exclude participants with noisy data. Default: ``True``.
        use_cache : bool, optional
            Whether to use caching for loading TFM data. Default: ``True``.
        only_labeled : bool, optional
            Whether to only return segments that are labeled (i.e., cut the data to the labeling borders).
            This is necessary when using the dataset for evaluating the performance of PEP extraction algorithms or for
            training ML-based PEP extraction algorithms. Default: ``False``.
        label_type: str, optional
            Which annotations to use. Can be either "rater_01", "rater_02", or "average". Default: "rater_01".
        """
        self.base_path = base_path
        self.exclude_no_recorded_data = exclude_no_recorded_data
        self.exclude_noisy_data = exclude_noisy_data
        self.data_to_exclude = self._find_data_to_exclude()
        self.use_cache = use_cache
        self.label_type = label_type
        super().__init__(
            groupby_cols=groupby_cols, subset_index=subset_index, return_clean=return_clean, only_labeled=only_labeled
        )

    def _sanitize_params(self) -> None:
        # ensure pathlib
        self.base_path = Path(self.base_path)

    def create_index(self) -> pd.DataFrame:
        self._sanitize_params()
        overview_df = pd.read_csv(self.base_path.joinpath("metadata/dataset_overview.csv"), sep=";")
        pids = list(overview_df["participant"])
        index = list(product(pids, self.PHASES))
        index = pd.DataFrame(index, columns=["participant", "phase"])
        for item in self.data_to_exclude:
            index = index.drop(index[(index["participant"] == item[0]) & (index["phase"] == item[1])].index)
        index = index.reset_index(drop=True)

        return index

    def _find_data_to_exclude(self) -> Sequence[tuple[str, str]]:
        data_to_exclude = []
        if self.exclude_no_recorded_data:
            data_to_exclude.extend(self.SUBSET_NO_RECORDED_DATA)
        if self.exclude_noisy_data:
            data_to_exclude.extend(self.SUBSET_NOISY_DATA)

        return data_to_exclude

    @property
    def sampling_rates(self) -> dict[str, int]:
        """Return the sampling rates of the ECG and ICG signals.

        Returns
        -------
        dict
            Dictionary with the sampling rates of the ECG and ICG signals in Hz.

        """
        return self.SAMPLING_RATES

    @property
    def sampling_rate_ecg(self) -> int:
        """Return the sampling rate of the ECG signal.

        Returns
        -------
        int
            Sampling rate of the ECG signal in Hz.

        """
        return self.SAMPLING_RATES["ecg_2"]

    @property
    def sampling_rate_icg(self) -> int:
        """Return the sampling rate of the ICG signal.

        Returns
        -------
        int
            Sampling rate of the ICG signal in Hz.

        """
        return self.SAMPLING_RATES["icg_der"]

    @property
    def date(self) -> pd.Series | pd.Timestamp:
        """Return the recording date of the participant(s).

        Returns
        -------
        pd.Series or pd.Timestamp
            Recording date of the participant(s). If only a single participant is selected, a single timestamp is
            returned. If multiple participants are selected, a :class:`~pandas.Series` with the recording dates for
            each participant is returned.

        """
        metadata_path = self.base_path.joinpath("metadata/recording_timestamps.xlsx")
        metadata = pd.read_excel(metadata_path)
        metadata = metadata.set_index("participant")["date"]
        if self.is_single("participant"):
            return metadata[self.index["participant"][0]]
        return metadata

    @property
    def tfm_data(self) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Return the Task Force Monitor (TFM) data.

        The data is loaded from the raw TFM data files and returned as :class:`~pandas.DataFrame`. The data can only be
        accessed for a single participant and a single phase or for a single participant and all phases.


        Returns
        -------
        :class:`~pandas.DataFrame` or dict
            TFM data as :class:`~pandas.DataFrame` or dictionary of :class:`~pandas.DataFrame` if multiple phases are
            selected.

        Raises
        ------
        ValueError
            If the data is accessed for multiple participants or multiple phases (that are not all phases).

        """
        participant = self.index["participant"][0]
        phases = self.index["phase"]

        if not (self.is_single(None) or len(phases) == len(self.PHASES)):
            raise ValueError(
                "TFM data can only be accessed for a single participant and ALL phases or "
                "for a single participant and a SINGLE phase."
            )

        tfm_path = self.base_path.joinpath(f"data_raw/{participant}/tfm_data/{participant.lower()}_no01.mat")

        if self.use_cache:
            tfm_data_dict = _cached_get_tfm_data(tfm_path, self.date)
        else:
            tfm_data_dict = _load_tfm_data(tfm_path, self.date)

        if self.is_single(None):
            tfm_data_dict = tfm_data_dict[phases[0]]

        if self.only_labeled:
            labeling_borders = self.labeling_borders

            if self.is_single(None):
                tfm_data_dict = self._cut_to_labeling_borders(tfm_data_dict, labeling_borders)
            else:
                for phase in phases:
                    borders = labeling_borders[labeling_borders["description"].apply(lambda x, ph=phase: ph in x)]
                    tfm_data = tfm_data_dict[phase]
                    tfm_data_dict[phase] = self._cut_to_labeling_borders(tfm_data, borders)

        return tfm_data_dict

    @property
    def icg(self) -> pd.DataFrame:
        """Return the ICG signal.

        If ``return_clean`` is set to ``True`` in the ``__init__``, the ICG signal is preprocessed and cleaned using
        the :class:`~biopsykit.signals.icg.preprocessing.IcgPreprocessingBandpass` algorithm before returning it.

        Returns
        -------
        :class:`~pandas.DataFrame`
            ICG signal as :class:`~pandas.DataFrame`.

        """
        if not self.is_single(None):
            raise ValueError("ICG data can only be accessed for a single participant and a single phase!")
        icg = self.tfm_data[["icg_der"]]
        if self.return_clean:
            algo = IcgPreprocessingBandpass()
            algo.clean(icg=icg, sampling_rate_hz=self.sampling_rate_icg)
            return algo.icg_clean_
        return icg

    @property
    def ecg(self) -> pd.DataFrame:
        """Return the ECG signal.

        If ``return_clean`` is set to ``True`` in the ``__init__``, the ECG signal is preprocessed and cleaned using the
        :class:`~biopsykit.signals.ecg.preprocessing.EcgPreprocessingNeurokit` algorithm before returning it.

        Returns
        -------
        :class:`~pandas.DataFrame`
            ECG signal as :class:`~pandas.DataFrame`.

        """
        if not self.is_single(None):
            raise ValueError("ECG data can only be accessed for a single participant and a single phase!")
        ecg = self.tfm_data[["ecg_2"]]
        ecg.columns = ["ecg"]

        if self.return_clean:
            algo = EcgPreprocessingNeurokit()
            algo.clean(ecg=ecg, sampling_rate_hz=self.sampling_rate_ecg)
            return algo.ecg_clean_
        return ecg

    @property
    def labeling_borders(self) -> pd.DataFrame:
        """Return the labeling borders for a selected participant and phase(s).

        Returns
        -------
        :class:`~pandas.DataFrame`
            Labeling borders as :class:`~pandas.DataFrame`.

        """
        participant = self.index["participant"][0]

        if not self.is_single("participant"):
            raise ValueError("Labeling borders can only be accessed for a single participant.")

        file_path = self.base_path.joinpath(
            f"data_raw/{participant}/tfm_data/reference_labels/labeling_borders_{participant}.csv"
        )
        data = load_labeling_borders(file_path)

        if self.is_single(None):
            phase = self.index["phase"][0]
            data = data[data["description"].apply(lambda x: phase in x)]

        return data

    @property
    def reference_heartbeats(self) -> pd.DataFrame:
        """Return the reference heartbeats.

        Returns
        -------
        :class:`~pandas.DataFrame`
            Reference heartbeats as a pandas DataFrame

        """
        return self._load_reference_heartbeats()

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

    def _load_reference_heartbeats(self) -> pd.DataFrame:
        reference_ecg = self.reference_labels_ecg
        reference_heartbeats = reference_ecg.reindex(["heartbeat"], level="channel")
        reference_heartbeats = compute_reference_heartbeats(reference_heartbeats)
        return reference_heartbeats

    def _load_reference_labels(self, channel: str) -> pd.DataFrame:
        participant = self.index["participant"][0]
        phases = self.index["phase"]

        if not (self.is_single(None) or len(phases) == len(self.PHASES)):
            raise ValueError(
                "Reference data can only be accessed for a single participant and ALL phases or "
                "for a single participant and a SINGLE phase."
            )
        reference_data_dict = {}
        rater_type = self.label_type
        if self.label_type == "average":
            # TODO implement
            raise NotImplementedError("Average reference labels are not implemented yet.")
        for phase in phases:
            file_path = self.base_path.joinpath(
                f"data_raw/{participant}/tfm_data/reference_labels/{rater_type}/"
                f"reference_labels_{participant}_{phase}_{channel}.csv"
            )
            reference_data = pd.read_csv(file_path)
            reference_data = reference_data.set_index(["heartbeat_id", "channel", "label"])

            reference_data_dict[phase] = reference_data

        if self.is_single(None):
            return reference_data_dict[phases[0]]
        return pd.concat(reference_data_dict, names=["phase"])

    @property
    def heartbeats(self) -> pd.DataFrame:
        """Segment heartbeats from the ECG data and return the heartbeat borders.

        Returns
        -------
        :class:`~pandas.DataFrame`
            Heartbeats as a pandas DataFrame.

        """
        heartbeat_algo = HeartbeatSegmentationNeurokit()
        heartbeat_algo.extract(ecg=self.ecg, sampling_rate_hz=self.sampling_rate_ecg)
        heartbeats = heartbeat_algo.heartbeat_list_
        return heartbeats

    @staticmethod
    def _cut_to_labeling_borders(data: pd.DataFrame, borders: pd.DataFrame) -> pd.DataFrame:
        start = borders.index[0]
        end = borders.index[-1]
        data = data.loc[start:end]
        return data

    @property
    def metadata(self) -> pd.DataFrame:
        """Return metadata for the selected participants.

        Returns
        -------
        :class:`~pandas.DataFrame`
            Metadata as a pandas DataFrame.

        """
        data = pd.read_csv(self.base_path.joinpath("metadata/demographics.csv"))
        data = data.set_index("participant")

        return data.loc[self.index["participant"].unique()]

    @property
    def age(self) -> pd.DataFrame:
        """Return the age of the selected participants.

        Returns
        -------
        :class:`~pandas.DataFrame`
            Age as a pandas DataFrame.

        """
        return self.metadata[["Age"]]

    @property
    def gender(self) -> pd.DataFrame:
        """Return the gender of the selected participants.

        Returns
        -------
        :class:`~pandas.DataFrame`
            Gender as a pandas DataFrame, recoded as {"M": "Male", "F": "Female"}

        """
        return self.metadata[["Gender"]].replace(self.GENDER_MAPPING)

    @property
    def bmi(self) -> pd.DataFrame:
        """Compute the BMI of the selected participants and return it.

        Returns
        -------
        :class:`~pandas.DataFrame`
            BMI as a pandas DataFrame.

        """
        return bmi(self.metadata[["Weight", "Height"]])
