"""
Base classes for PEP extraction datasets.

Provides base classes and mixins for datasets used in PEP extraction from ICG and ECG data.

.. _custom_dataset_basics:

Classes
-------
:class:`~pepbench.datasets._base_pep_extraction_dataset.BasePepDataset`
    Interface for datasets used for PEP extraction by the :class:`~pepbench.pipelines.PepExtractionPipeline`.
:class:`~pepbench.datasets._base_pep_extraction_dataset.MetadataMixin`
    Mixin for datasets that provide demographic metadata (age, gender, BMI).
:class:`~pepbench.datasets._base_pep_extraction_dataset.PepLabelMixin`
    Mixin for datasets that contain manually labeled PEP data and reference labels.
:class:`~pepbench.datasets._base_pep_extraction_dataset.BasePepDatasetWithAnnotations`
    Combines :class:`BasePepDataset` and :class:`PepLabelMixin` to provide a unified
    interface for evaluation datasets with annotations.

"""
import pandas as pd
from biopsykit.utils.dtypes import EcgRawDataFrame, HeartbeatSegmentationDataFrame, IcgRawDataFrame
from tpcp import Dataset

from pepbench._docutils import make_filldoc

base_pep_extraction_docfiller = make_filldoc(
    {
        "base_attributes_pep": """
        icg : :class:`~biopsykit.utils.dtypes.IcgRawDataFrame`
            The raw ICG data.
        ecg : :class:`~biopsykit.utils.dtypes.EcgRawDataFrame`
            The raw ECG data.
        sampling_rate_ecg : int
            The sampling rate of the ECG data in Hz.
        sampling_rate_icg : int
            The sampling rate of the ICG data in Hz.
        heartbeats : :class:`~biopsykit.utils.dtypes.HeartbeatSegmentationDataFrame`
            The heartbeats extracted from the ECG data.
        """,
        "base_attributes_metadata": """
        age : :class:`~pandas.DataFrame`
            The age of the participants
        gender : :class:`~pandas.DataFrame`
            The gender of the participants
        bmi : :class:`~pandas.DataFrame`
            The BMI of the participants
        metadata : :class:`~pandas.DataFrame`
            The metadata of the participants, consisting of a combination of age, gender, and BMI.
        """,
        "base_attributes_pep_label": """
        reference_pep : :class:`~pandas.DataFrame`
            The reference PEP data.
        reference_heartbeats : :class:`~pandas.DataFrame`
            The reference heartbeats.
        reference_labels_ecg : :class:`~pandas.DataFrame`
            The reference labels for the ECG data.
        reference_labels_icg : :class:`~pandas.DataFrame`
            The reference labels for the ICG data.
        """,
    }
)


@base_pep_extraction_docfiller
class BasePepDataset(Dataset):
    """Interface for datasets for PEP extraction from ICG and ECG data.

    This class defines the interface for datasets that are used for PEP extraction using the
    :class:`~pepbench.pipelines.PepExtractionPipeline`. It provides the necessary properties and methods to access
    the data and metadata required for PEP extraction. It is intended to be subclassed.

    Parameters
    ----------
    groupby_cols : list[str] or str or None, optional
        Columns used to group the dataset, by default None.
    subset_index :  :class:`~pandas.DataFrame` or None, optional
        Subset index for the dataset, by default None.
    return_clean : bool, optional
        Whether to return cleaned data by default, by default True

    Attributes
    ----------
    %(base_attributes_pep)s

    """

    return_clean: bool

    def __init__(
        self,
        groupby_cols: list[str] | str | None = None,
        subset_index: pd.DataFrame | None = None,
        return_clean: bool = True,
    ) -> None:
        """Initialize the :class:`~pepbench.datasets._base_pep_extraction_dataset.BasePepDataset`.

        Parameters
        ----------
        groupby_cols : list[str] or str or None, optional
            Columns used to group the dataset, by default None.
        subset_index :  :class:`~pandas.DataFrame` or None, optional
            Subset index for the dataset, by default None.
        return_clean : bool, optional
            Whether to return cleaned data by default, by default True
        """
        self.return_clean = return_clean
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    @property
    def icg(self) -> IcgRawDataFrame:
        """Return raw ICG data.

        Returns
        -------
        :class:`~biopsykit.utils.dtypes.IcgRawDataFrame`
            The raw ICG data.

        Raises
        ------
        NotImplementedError
            If not implemented in the subclass.
        """
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def ecg(self) -> EcgRawDataFrame:
        """Return raw ECG data.

        Returns
        -------
        :class:`~biopsykit.utils.dtypes.EcgRawDataFrame`
            The raw ECG data.

        Raises
        ------
        NotImplementedError
            If not implemented in the subclass.
        """
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def sampling_rate_ecg(self) -> int:
        """Return sampling rate of the ECG signal in Hz.

        Returns
        -------
        int
            The sampling rate of the ECG data in Hz.

        Raises
        ------
        NotImplementedError
            If not implemented in the subclass.
        """
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def sampling_rate_icg(self) -> int:
        """Return sampling rate of the ICG signal in Hz.

        Returns
        -------
        int
            The sampling rate of the ICG data in Hz.

        Raises
        ------
        NotImplementedError
            If not implemented in the subclass.
        """
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def heartbeats(self) -> HeartbeatSegmentationDataFrame:
        """Return heartbeat segmentation extracted from ECG.

        Returns
        -------
        :class:`~biopsykit.utils.dtypes.HeartbeatSegmentationDataFrame`
            The heartbeats extracted from the ECG data.

        Raises
        ------
        NotImplementedError
            If not implemented in the subclass!
        """
        raise NotImplementedError("This property needs to be implemented in the subclass!")


class MetadataMixin(Dataset):
    """Interface for all datasets that contain certain metadata.

    This interface can be used by datasets that contain metadata like age, gender, and BMI.

    Attributes
    ----------
    base_demographics : :class:`~pandas.DataFrame`
        The base demographics of the participants, including gender, age, and BMI.
    %(base_attributes_metadata)s

    """

    def __init__(self, groupby_cols: list[str] | str | None = None, subset_index: pd.DataFrame | None = None) -> None:
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    @property
    def base_demographics(self) -> pd.DataFrame:
        """Return base demographics of the participants.

        Returns
        -------
        :class:`~pandas.DataFrame`
            The base demographics DataFrame including gender, age, and BMI.
        """
        return pd.concat([self.gender, self.age, self.bmi], axis=1)

    @property
    def age(self) -> pd.DataFrame:
        """Return age of the participants.

        Returns
        -------
        :class:`~pandas.DataFrame`
            The age column(s) indexed by the dataset index.

        Raises
        ------
        NotImplementedError
            If not implemented in the subclass.
        """
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def gender(self) -> pd.DataFrame:
        """Return gender of the participants.

        Returns
        -------
        :class:`~pandas.DataFrame`
            The gender column(s) indexed by the dataset index.

        Raises
        ------
        NotImplementedError
            If not implemented in the subclass.
        """
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def bmi(self) -> pd.DataFrame:
        """Return BMI of the participants.

        Returns
        -------
        :class:`~pandas.DataFrame`
            The BMI column(s) indexed by the dataset index.

        Raises
        ------
        NotImplementedError
            If not implemented in the subclass.
        """
        raise NotImplementedError("This property needs to be implemented in the subclass!")


class PepLabelMixin(Dataset):
    """Interface for all datasets with manually labeled PEP data.

    This interface can be used by datasets that contain manually labeled PEP data. It provides the necessary properties
    to access the reference PEP data and the reference heartbeats.

    Attributes
    ----------
    %(base_attributes_pep_label)s

    """

    def __init__(self, groupby_cols: list[str] | str | None = None, subset_index: pd.DataFrame | None = None) -> None:
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    @property
    def reference_heartbeats(self) -> pd.DataFrame:
        """Return reference heartbeats.

        Returns
        -------
        :class:`~pandas.DataFrame`
            The reference heartbeats.

        Raises
        ------
        NotImplementedError
            If not implemented in the subclass.
        """
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def reference_labels_ecg(self) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Return reference labels for the ECG data.

        Returns
        -------
        :class:`~pandas.DataFrame` or dict
            The reference labels for ECG, possibly split by label types.

        Raises
        ------
        NotImplementedError
            If not implemented in the subclass.
        """
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def reference_labels_icg(self) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Return reference labels for the ICG data.

        Returns
        -------
        :class:`~pandas.DataFrame` or dict
            The reference labels for ICG, possibly split by label types.

        Raises
        ------
        NotImplementedError
            If not implemented in the subclass.
        """
        raise NotImplementedError("This property needs to be implemented in the subclass!")


@base_pep_extraction_docfiller
class BasePepDatasetWithAnnotations(BasePepDataset, PepLabelMixin):
    """Unified interface for datasets used for evaluating PEP extraction algorithms.

    This interface extends the :class:`~pepbench.datasets.BasePepDataset` by adding support for metadata and
    reference PEP data.

    Attributes
    ----------
    %(base_attributes_pep)s
    %(base_attributes_pep_label)s
    %(base_attributes_metadata)s

    """

    only_labeled: bool

    def __init__(
        self,
        groupby_cols: list[str] | str | None = None,
        subset_index: pd.DataFrame | None = None,
        return_clean: bool = True,
        only_labeled: bool = False,
    ) -> None:
        """Initialize the :class:`~pepbench.datasets._base_pep_extraction_dataset.BasePepDatasetWithAnnotations`.

        Parameters
        ----------
        groupby_cols : list[str] or str or None, optional
            Columns used to group the dataset, by default None.
        subset_index : :class:`~pandas.DataFrame` or None, optional
            Subset index for the dataset, by default None.
        return_clean : bool, optional
            Whether to return cleaned data by default, by default True
        only_labeled : bool, optional
            Whether to use only labeled data points, by default False.
        """
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index, return_clean=return_clean)
        self.only_labeled = only_labeled

    @property
    def reference_pep(self) -> pd.DataFrame:
        """Compute the reference PEP values between the reference Q-peak and B-point labels.

        Returns
        -------
        :class:`~pandas.DataFrame`
            DataFrame containing the computed PEP values.
        """
        heartbeats = self.reference_heartbeats
        reference_icg = self.reference_labels_icg
        reference_ecg = self.reference_labels_ecg

        b_points = reference_icg.reindex(["ICG", "Artefact"], level="channel").droplevel("label")
        b_points = self._fill_unlabeled_artefacts(b_points, reference_icg)
        b_point_artefacts = b_points.reindex(["Artefact"], level="channel").droplevel("channel")
        b_points = b_points.reindex(["ICG"], level="channel").droplevel("channel")

        q_peaks = reference_ecg.reindex(["ECG", "Artefact"], level="channel").droplevel("label")
        q_peaks = self._fill_unlabeled_artefacts(q_peaks, reference_ecg)
        q_peak_artefacts = q_peaks.reindex(["Artefact"], level="channel").droplevel("channel")
        q_peaks = q_peaks.reindex(["ECG"], level="channel").droplevel("channel")

        pep_reference = heartbeats.copy()
        pep_reference.columns = [
            f"heartbeat_{col}" if col != "r_peak_sample" else "r_peak_sample" for col in heartbeats.columns
        ]

        pep_reference = pep_reference.assign(
            q_peak_sample=q_peaks["sample_relative"],
            b_point_sample=b_points["sample_relative"],
            nan_reason=pd.NA,
        )
        # fill nan_reason column with artefact information
        pep_reference.loc[b_point_artefacts.index, "nan_reason"] = "icg_artefact"
        pep_reference.loc[q_peak_artefacts.index, "nan_reason"] = "ecg_artefact"

        pep_reference = pep_reference.assign(
            pep_sample=pep_reference["b_point_sample"] - pep_reference["q_peak_sample"]
        )
        pep_reference = pep_reference.assign(pep_ms=pep_reference["pep_sample"] / self.sampling_rate_ecg * 1000)

        # reorder columns
        pep_reference = pep_reference[
            [
                "heartbeat_start_sample",
                "heartbeat_end_sample",
                "q_peak_sample",
                "b_point_sample",
                "pep_sample",
                "pep_ms",
                "nan_reason",
            ]
        ]

        return pep_reference.convert_dtypes(infer_objects=True)

    @staticmethod
    def _fill_unlabeled_artefacts(points: pd.DataFrame, reference_data: pd.DataFrame) -> pd.DataFrame:
        """Fill unlabeled artefacts in the reference labels.

        Parameters
        ----------
        points: :class:`~pandas.DataFrame`
            DataFrame containing the reference labels (either Q-peaks or B-points).
        reference_data : :class:`~pandas.DataFrame`
            DataFrame containing the reference heartbeat segmentation data.

        Returns
        -------
        :class:`~pandas.DataFrame`
            DataFrame with filled unlabeled artefacts.
        """
        # get the indices of reference_icg that are not in b_points.index => they are artefacts but were not labeled
        heartbeat_ids = reference_data.index.get_level_values("heartbeat_id").unique()
        # insert "Artefact" label for artefacts that were not labeled to b_points,
        # set the sample to the middle of the heartbeat
        artefact_ids = list(heartbeat_ids.difference(points.droplevel("channel").index))
        for artefact_id in artefact_ids:
            start_abs, end_abs = reference_data.xs(artefact_id, level="heartbeat_id")["sample_absolute"]
            start_rel, end_rel = reference_data.xs(artefact_id, level="heartbeat_id")["sample_relative"]
            points.loc[(artefact_id, "Artefact"), :] = (int((start_abs + end_abs) / 2), int((start_rel + end_rel) / 2))

        points = points.sort_index()
        return points

