# Module for the EmpkinS dataset class.
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
    """Dataset class for the EmpkinS dataset.

    Provides access to Biopac ECG/ICG signals, preprocessed signals, timelogs for
    experimental phases, reference annotations, and participant metadata.

    Parameters
    ----------
    base_path : path-like
        Path to the root directory of the EmpkinS dataset.
    groupby_cols : sequence of str, optional
        Columns to group the dataset index by.
    subset_index : sequence of str, optional
        Subset of the dataset index to operate on.
    return_clean : bool, optional
        If True, return preprocessed/cleaned ECG and ICG signals. Default is True.
    exclude_missing_data : bool, optional
        If True, exclude participants with missing data. Default is False.
    use_cache : bool, optional
        If True, cache loading of Biopac files. Default is True.
    only_labeled : bool, optional
        If True, return only labeled sections (cut to labeling borders). Default is False.
    label_type : {'rater_01', 'rater_02', 'average'}, optional
        Which label set to use for reference annotations. Default is 'rater_01'.

    Attributes
    ----------
    SAMPLING_RATES : dict
        Per-channel sampling rates in Hz.
    PHASES : sequence
        Ordered list of experimental phases.
    CONDITIONS : sequence
        Available experimental conditions.
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
        """Sanitize input parameters."""
        # ensure pathlib
        self.base_path = Path(self.base_path)

    def create_index(self) -> pd.DataFrame:
        """Create the dataset index.

        Returns
        -------
        :class:`~pandas.DataFrame`
            DataFrame containing all combinations of participant IDs, conditions, and phases.
        """
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
        """Return sampling rates of the ECG and ICG signals.

        Returns
        -------
        dict
            Dictionary with the sampling rates of the ECG and ICG signals in Hz.

        """
        return self.SAMPLING_RATES

    @property
    def sampling_rate_ecg(self) -> int:
        """Return sampling rate of the ECG signal.

        Returns
        -------
        int
            Sampling rate of the ECG data in Hz.

        """
        return self.SAMPLING_RATES["ecg"]

    @property
    def sampling_rate_icg(self) -> int:
        """Return sampling rate of the ICG signal.

        Returns
        -------
        int
            Sampling rate of the ICG data in Hz.

        """
        return self.SAMPLING_RATES["icg"]

    @cached_property
    def biopac(self) -> pd.DataFrame:
        """Return biopac data for the current subset.

        Returns
        -------
        :class:`~pandas.DataFrame` or dict
            If a single participant+condition+phase is selected, returns a DataFrame
            containing the Biopac channels. If a single participant+condition but all
            phases are selected and `only_labeled` is True, returns a dict mapping phase
            names to DataFrames. In other multi-subset cases a ValueError is raised.

        Raises
        ------
        ValueError
            If the selection is not a single participant and condition (and optionally
            a single phase).
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
        """Cut biopac data to the labeling borders."""
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
            If not operating on a single participant, condition, and phase/selection
            as required by the API.
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

        Raises
        ------
        ValueError
            If not operating on a single participant, condition, and phase/selection
            as required by the API.
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

        Timelog entries describing experimental phase boundaries.

        Returns
        -------
        :class:`~pandas.DataFrame`
            Timelog rows for the selected participant/condition and (optionally) phase.

        Raises
        ------
        ValueError
            If timelog access is attempted for unsupported selections (e.g., multiple
            participants or conditions).
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
        """Load biopac data for the given participant, condition, and phase.

        Parameters
        ----------
        participant_id : str
            Participant identifier.
        condition : str
            Experimental condition.
        phase : str
            Experimental phase.

        Returns
        -------
        tuple[:class:`~pandas.DataFrame`, int]
            Tuple containing the biopac data DataFrame and the sampling frequency in Hz.
        """
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
        """Load timelog data for the given participant, condition, and phase.

        Parameters
        ----------
        participant_id : str
            Participant identifier.
        condition : str
            Experimental condition.
        phase : str
            Experimental phase.

        Returns
        -------
        :class:`~pandas.DataFrame`
            Timelog data for the specified participant, condition, and phase.
        """
        return _load_timelog(self.base_path, participant_id, condition, phase)

    def _all_phases_selected(self) -> bool:
        """Check if all phases are selected in the current index.

        Returns
        -------
        bool
            True if all phases are selected, False otherwise.
        """
        # check if all phases are selected
        return len(self.index["phase"]) == len(self.PHASES)

    @property
    def labeling_borders(self) -> pd.DataFrame:
        """Labeling borders for the selected participant and condition and phase.

        Returns
        -------
        :class:`~pandas.DataFrame`
            Labeling borders with columns including `sample_absolute` and `description`.

        Raises
        ------
        ValueError
            If not operating on a single participant.
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
        """Return computed reference heartbeat markers derived from ECG reference labels.

        Returns
        -------
        :class:`~pandas.DataFrame`
            Heartbeat segmentation/reference table derived from ECG reference labels.
        """
        return self._load_reference_heartbeats()

    @property
    def reference_labels_ecg(self) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Return reference labels for a given channel and the current selection.

        Returns
        -------
        :class:`~pandas.DataFrame` or dict
            If a single phase is selected, returns a DataFrame of reference labels for
            that phase. If all phases are selected, returns a concatenated DataFrame
            indexed by phase.

        Raises
        ------
        ValueError
            If reference labels are requested for unsupported subset selections.
        """
        return self._load_reference_labels("ECG")

    @property
    def reference_labels_icg(self) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Reference labels for a given channel and the current selection.

        Returns
        -------
        :class:`~pandas.DataFrame` or dict
            If a single phase is selected, returns a DataFrame of reference labels for
            that phase. If all phases are selected, returns a concatenated DataFrame
            indexed by phase.

        Raises
        ------
        ValueError
            If reference labels are requested for unsupported subset selections.
        """
        return self._load_reference_labels("ICG")

    def _load_reference_heartbeats(self) -> pd.DataFrame:
        """Load and compute reference heartbeats from ECG reference labels.

        Returns
        -------
        :class:`~pandas.DataFrame`
            DataFrame containing computed heartbeat segmentation from reference ECG labels.
        """
        reference_ecg = self.reference_labels_ecg
        reference_heartbeats = reference_ecg.reindex(["heartbeat"], level="channel")
        reference_heartbeats = compute_reference_heartbeats(reference_heartbeats)
        return reference_heartbeats

    def _load_reference_labels(self, channel: str) -> pd.DataFrame:
        """Load reference labels for the given channel and current selection.

        Parameters
        ----------
        channel : str
            Channel for which to load reference labels ("ECG" or "ICG").

        Returns
        -------
        :class:`~pandas.DataFrame` or dict
            If a single phase is selected, returns a DataFrame of reference labels for
            that phase. If all phases are selected, returns a concatenated DataFrame
            indexed by phase.
        """
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
        """Heartbeat segmentation computed from the ECG signal.

        Uses HeartbeatSegmentationNeurokit to extract heartbeat borders.

        Returns
        -------
        :class:`~biopsykit.utils.dtypes.HeartbeatSegmentationDataFrame`
            DataFrame describing heartbeat onsets/offsets and related segmentation info.
        """
        heartbeat_algo = HeartbeatSegmentationNeurokit(variable_length=True)
        heartbeat_algo.extract(ecg=self.ecg, sampling_rate_hz=self.sampling_rate_ecg)
        heartbeats = heartbeat_algo.heartbeat_list_
        return heartbeats

    @property
    def metadata(self) -> pd.DataFrame:
        """Return participant metadata.

        Returns
        -------
        :class:`~pandas.DataFrame`
            Participant metadata indexed by participant id. Only rows for the
            currently selected participants are returned.
        """
        data = pd.read_csv(self.base_path.joinpath("metadata/demographics.csv"))
        data = data.set_index("participant")

        return data.loc[self.index["participant"].unique()]

    @property
    def age(self) -> pd.DataFrame:
        """Return age of selected participants.

        Returns
        -------
        :class:`~pandas.DataFrame`
            DataFrame with the `Age` column for the selected participants.
        """
        return self.metadata[["Age"]]

    @property
    def gender(self) -> pd.DataFrame:
        """Return gender of selected participants.

        Returns
        -------
        :class:`~pandas.DataFrame`
            Gender as a pandas DataFrame, recoded as {1: "Female", 2: "Male"}
        """
        return self.metadata[["Gender"]].replace(self.GENDER_MAPPING)

    @property
    def bmi(self) -> pd.DataFrame:
        """Return body-mass index (BMI) for selected participants.

        Returns
        -------
        :class:`~pandas.DataFrame`
            Computed BMI (using demographics `Weight` and `Height`) for the selected participants.
        """
        return bmi(self.metadata[["Weight", "Height"]])
