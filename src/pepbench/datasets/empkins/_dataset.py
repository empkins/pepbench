from collections.abc import Sequence
from functools import cached_property, lru_cache
from itertools import product
from pathlib import Path
from typing import ClassVar

import pandas as pd
from biopsykit.metadata import bmi
from biopsykit.signals.ecg.preprocessing import EcgPreprocessingNeurokit
from biopsykit.signals.ecg.segmentation import HeartbeatSegmentationNeurokit
from biopsykit.signals.icg.preprocessing import IcgPreprocessingBandpass
from biopsykit.utils.dtypes import EcgRawDataFrame, HeartbeatSegmentationDataFrame, IcgRawDataFrame
from biopsykit.utils.file_handling import get_subject_dirs

from pepbench.datasets import BasePepDatasetWithAnnotations
from pepbench.datasets._base_pep_extraction_dataset import MetadataMixin, base_pep_extraction_docfiller
from pepbench.datasets._helper import compute_reference_heartbeats, load_labeling_borders
from pepbench.datasets.empkins._helper import _load_biopac_data, _load_timelog
from pepbench.utils._types import path_t

_cached_get_biopac_data = lru_cache(maxsize=4)(_load_biopac_data)
# cache_dir = "./cachedir"
# memory = Memory(location=cache_dir, verbose=0)
# _cached_get_biopac_data = memory.cache(_load_biopac_data)


@base_pep_extraction_docfiller
class EmpkinsDataset(BasePepDatasetWithAnnotations, MetadataMixin):
    """Dataset class for the EmpkinS Dataset.

    This class is the ``tpcp`` dataset class for the EmpkinS dataset. It provides access to the Biopac data (for ECG
    and ICG), the timelogs for the different experimental phases, the reference annotations for the ECG and ICG
    annotations, as well as metadata like age, gender, and BMI.

    Attributes
    ----------
    %(base_attributes_pep)s
    %(base_attributes_pep_label)s
    %(base_attributes_metadata)s
    timelog : :class:`~pandas.DataFrame`
        Timelog data, indicating the start and end of each experimental phase, as a pandas DataFrame.
    labeling_borders : :class:`~pandas.DataFrame`
        Labeling borders for the selected recording as a pandas DataFrame.

    """

    base_path: Path
    use_cache: bool
    exclude_missing_data: bool

    SAMPLING_RATES: ClassVar[dict[str, int]] = {"ecg": 1000, "icg": 1000}

    PHASES: ClassVar[Sequence[str]] = ["Prep", "Pause_1", "Talk", "Math", "Pause_5"]

    CONDITIONS: ClassVar[Sequence[str]] = ["tsst", "ftsst"]

    GENDER_MAPPING: ClassVar[dict[int, str]] = {1: "Female", 2: "Male"}

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Sequence[str] | None = None,
        subset_index: Sequence[str] | None = None,
        *,
        return_clean: bool = True,
        exclude_missing_data: bool = False,
        use_cache: bool = True,
        only_labeled: bool = False,
        label_type: str = "rater_01",
    ) -> None:
        """Initialize a new ``EmpkinsDataset`` instance.

        Parameters
        ----------
        base_path : :class:`~pathlib.Path` or str
            Path to the root directory of the EmpkinS dataset.
        return_clean : bool
            Whether to return the preprocessed/cleaned ECG and ICG data when accessing the respective properties.
            Default: ``True``.
        exclude_missing_data : bool
            Whether to exclude participants where parts of the data are missing. Default: ``False``.
        use_cache : bool
            Whether to use caching for loading biopac data. Default: ``True``.
        only_labeled : bool
            Whether to only return sections of the biopac data that are labeled (i.e., cut to labeling borders).
            This is necessary when using the dataset for evaluating the performance of PEP extraction algorithms or for
            training ML-based PEP extraction algorithms. Default: ``False``.
        label_type: str, optional
            Which annotations to use. Can be either "rater_01", "rater_02", or "average". Default: "rater_01".

        """
        # ensure pathlib
        self.base_path = base_path
        self.exclude_missing_data = exclude_missing_data
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
        # data is located in a folder named "Data" and data per participant is located in folders named "VP_xx"
        participant_ids = [
            participant_dir.name
            for participant_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_[0-9]{3}")
        ]

        # excludes participants where parts of data are missing
        if self.exclude_missing_data:
            for p_id in self.MISSING_DATA:
                if p_id in participant_ids:
                    participant_ids.remove(p_id)

        index = list(product(participant_ids, self.CONDITIONS, self.PHASES))
        index = pd.DataFrame(index, columns=["participant", "condition", "phase"])
        return index

    @property
    def sampling_rate(self) -> dict[str, float]:
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
            Sampling rate of the ECG data in Hz.

        """
        return self.SAMPLING_RATES["ecg"]

    @property
    def sampling_rate_icg(self) -> int:
        """Return the sampling rate of the ICG signal.

        Returns
        -------
        int
            Sampling rate of the ICG data in Hz.

        """
        return self.SAMPLING_RATES["icg"]

    @cached_property
    def biopac(self) -> pd.DataFrame:
        """Return the biopac data.

        This method returns the biopac data for the selected participant, condition, and phase. If only one participant
        and condition is selected, the entire dataset is returned. If only one participant, condition, and phase is
        selected, only the data for this phase is returned. In all other cases (i.e., data from multiple participants
        or from multiple conditions), a ``ValueError`` is raised.

        Returns
        -------
        :class:`~pandas.DataFrame`
            Biopac data as a pandas DataFrame.

        Raises
        ------
        ValueError
            If the current subset is not a single participant and condition or
            a single participant, condition, and phase.

        """
        participant = self.index["participant"][0]
        condition = self.index["condition"][0]

        if self.is_single(None):
            phase = self.index["phase"][0]
        elif self.is_single(["participant", "condition"]):
            phase = "all"
        else:
            raise ValueError("Biopac data can only be accessed for one single participant and condition at once!")

        data, fs = self._get_biopac_data(participant, condition, phase)

        if self.only_labeled:
            biopac_data_dict = {}
            labeling_borders = self.labeling_borders

            if self.is_single(None):
                biopac_data_dict = self._cut_to_labeling_borders(data, labeling_borders)
            else:
                for phase in self.PHASES:
                    borders = labeling_borders[labeling_borders["description"].apply(lambda x, ph=phase: ph in x)]
                    biopac_data_dict[phase] = self._cut_to_labeling_borders(data, borders)
            return biopac_data_dict

        return data

    @staticmethod
    def _cut_to_labeling_borders(data: pd.DataFrame, labeling_borders: pd.DataFrame) -> pd.DataFrame:
        start_index = labeling_borders["sample_relative"].iloc[0]
        end_index = labeling_borders["sample_relative"].iloc[-1]
        return data.iloc[start_index:end_index]

    @property
    def icg(self) -> IcgRawDataFrame:
        """Return the ICG channel from the biopac data.

        If ``return_clean`` is set to ``True`` in the ``__init__``, the ICG signal is preprocessed and cleaned using
        the :class:`~biopsykit.signals.icg.preprocessing.IcgPreprocessingBandpass` algorithm before returning it.


        Returns
        -------
        :class:`~pandas.DataFrame`
            ICG data as a pandas DataFrame.

        Raises
        ------
        ValueError
            If the current subset is not a single participant, condition, and phase or a single participant and
            condition.

        """
        if not self.is_single(None):
            raise ValueError(
                "ICG data can only be accessed for a single participant, a single condition, and a single phase!"
            )
        icg = self.biopac[["icg_der"]]
        if self.return_clean:
            algo = IcgPreprocessingBandpass()
            algo.clean(icg=icg, sampling_rate_hz=self.sampling_rate_icg)
            return algo.icg_clean_
        return icg

    @property
    def ecg(self) -> EcgRawDataFrame:
        """Return the ECG channel from the biopac data.

        If ``return_clean`` is set to ``True`` in the ``__init__``, the ECG signal is preprocessed and cleaned using the
        :class:`~biopsykit.signals.ecg.preprocessing.EcgPreprocessingNeurokit` algorithm before returning it.

        Returns
        -------
        :class:`~biopsykit.utils.dtypes.EcgRawDataFrame`
            ECG data as a pandas DataFrame.

        """
        if not self.is_single(None):
            raise ValueError(
                "ECG data can only be accessed for a single participant, a single condition, and a single phase!"
            )
        ecg = self.biopac[["ecg"]]
        if self.return_clean:
            algo = EcgPreprocessingNeurokit()
            algo.clean(ecg=ecg, sampling_rate_hz=self.sampling_rate_ecg)
            return algo.ecg_clean_
        return ecg

    @property
    def timelog(self) -> pd.DataFrame:
        """Return the timelog data.

        The timelog data contains information about the start and end of each experimental phase.

        Returns
        -------
        :class:`~pandas.DataFrame`
            Timelog data as a pandas DataFrame.

        """
        if self.is_single(None):
            participant = self.index["participant"][0]
            condition = self.index["condition"][0]
            phase = self.index["phase"][0]
            return self._get_timelog(participant, condition, phase)

        if self.is_single(["participant", "condition"]):
            if not self._all_phases_selected():
                raise ValueError("Timelog can only be accessed for all phases or one specific phase!")

            participant = self.index["participant"][0]
            condition = self.index["condition"][0]
            return self._get_timelog(participant, condition, "all")

        raise ValueError("Timelog can only be accessed for a single participant and a single condition at once!")

    def _get_biopac_data(self, participant_id: str, condition: str, phase: str) -> tuple[pd.DataFrame, int]:
        if self.use_cache:
            data, fs = _cached_get_biopac_data(self.base_path, participant_id, condition)
        else:
            data, fs = _load_biopac_data(self.base_path, participant_id, condition)

        if phase == "all":
            return data, fs
        # cut biopac data to specified phase
        timelog = self.timelog
        phase_start = timelog[phase]["start"].iloc[0]
        phase_end = timelog[phase]["end"].iloc[0]
        data = data.loc[phase_start:phase_end]
        return data, fs

    def _get_timelog(self, participant_id: str, condition: str, phase: str) -> pd.DataFrame:
        return _load_timelog(self.base_path, participant_id, condition, phase)

    def _all_phases_selected(self) -> bool:
        # check if all phases are selected
        return len(self.index["phase"]) == len(self.PHASES)

    @property
    def labeling_borders(self) -> pd.DataFrame:
        """Return the labeling borders for the selected participant, condition, and phase(s).

        Returns
        -------
        :class:`~pandas.DataFrame`
            Labeling borders as a pandas DataFrame.

        """
        participant = self.index["participant"][0]
        condition = self.index["condition"][0]

        if not self.is_single("participant"):
            raise ValueError("Labeling borders can only be accessed for a single participant.")

        file_path = self.base_path.joinpath(
            f"data_per_subject/{participant}/{condition}/biopac/reference_labels/labeling_borders_{participant}_{condition}.csv"
        )
        data = load_labeling_borders(file_path)

        if self.is_single(None):
            phase = self.index["phase"][0]
            data = data[data["description"].apply(lambda x, ph=phase: ph in x)]

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
    def reference_labels_ecg(self) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Return the reference labels for the ECG channel.

        Returns
        -------
        :class:`~pandas.DataFrame`
            Reference labels for the ECG channel as a pandas DataFrame.

        """
        return self._load_reference_labels("ECG")

    @property
    def reference_labels_icg(self) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Return the reference labels for the ICG channel.

        Returns
        -------
        :class:`~pandas.DataFrame`
            Reference labels for the ICG channel as a pandas DataFrame.

        """
        return self._load_reference_labels("ICG")

    def _load_reference_heartbeats(self) -> pd.DataFrame:
        reference_ecg = self.reference_labels_ecg
        reference_heartbeats = reference_ecg.reindex(["heartbeat"], level="channel")
        reference_heartbeats = compute_reference_heartbeats(reference_heartbeats)
        return reference_heartbeats

    def _load_reference_labels(self, channel: str) -> pd.DataFrame:
        participant = self.index["participant"][0]
        condition = self.index["condition"][0]
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
                f"data_per_subject/{participant}/{condition}/biopac/reference_labels/{rater_type}/"
                f"reference_labels_{participant}_{condition}_{phase.lower()}_{channel.lower()}.csv"
            )
            reference_data = pd.read_csv(file_path)
            reference_data = reference_data.set_index(["heartbeat_id", "channel", "label"])

            start_idx = self.get_subset(phase=phase).labeling_borders.iloc[0]
            reference_data = reference_data.assign(
                sample_relative=reference_data["sample_absolute"] - start_idx["sample_absolute"]
            )

            reference_data_dict[phase] = reference_data

        if self.is_single(None):
            return reference_data_dict[phases[0]]
        return pd.concat(reference_data_dict, names=["phase"])

    @property
    def heartbeats(self) -> HeartbeatSegmentationDataFrame:
        """Segment heartbeats from the ECG data and return the heartbeat borders.

        Returns
        -------
        :class:`~biopsykit.utils.dtypes.HeartbeatSegmentationDataFrame`
            Heartbeats as a pandas DataFrame.

        """
        heartbeat_algo = HeartbeatSegmentationNeurokit(variable_length=True)
        heartbeat_algo.extract(ecg=self.ecg, sampling_rate_hz=self.sampling_rate_ecg)
        heartbeats = heartbeat_algo.heartbeat_list_
        return heartbeats

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
            Gender as a pandas DataFrame, recoded as {1: "Female", 2: "Male"}

        """
        return self.metadata[["Gender"]].replace(self.GENDER_MAPPING)

    @property
    def bmi(self) -> pd.DataFrame:
        """Compute the BMI of the selected participants and return it.

        Returns
        -------
        :class:`~pandas.DataFrame`
            BMI as a pandas DataFrame

        """
        return bmi(self.metadata[["Weight", "Height"]])
