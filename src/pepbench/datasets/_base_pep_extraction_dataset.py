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
    """Interface for all datasets for PEP extraction from ICG and ECG data.

    This class defines the interface for datasets that are used for PEP extraction using the
    :class:`~pepbench.pipelines.PepExtractionPipeline`. It provides the necessary properties and methods to access the
    data and metadata required for PEP extraction. It is not intended to be used directly, but rather as a base class
    for other datasets.

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
        self.return_clean = return_clean
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    @property
    def icg(self) -> IcgRawDataFrame:
        """The raw ICG data.

        Returns
        -------
        :class:`~biopsykit.utils.dtypes.IcgRawDataFrame`
            The raw ICG data.

        """
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def ecg(self) -> EcgRawDataFrame:
        """The raw ECG data.

        Returns
        -------
        :class:`~biopsykit.utils.dtypes.EcgRawDataFrame`
            The raw ECG data.

        """
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def sampling_rate_ecg(self) -> int:
        """The sampling rate of the ECG data in Hz.

        Returns
        -------
        int
            The sampling rate of the ECG data in Hz.

        """
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def sampling_rate_icg(self) -> int:
        """The sampling rate of the ICG data in Hz.

        Returns
        -------
        int
            The sampling rate of the ICG data in Hz.

        """
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def heartbeats(self) -> HeartbeatSegmentationDataFrame:
        """The heartbeats extracted from the ECG data.

        Returns
        -------
        :class:`~biopsykit.utils.dtypes.HeartbeatSegmentationDataFrame`
            The heartbeats extracted from the ECG data.

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
        return pd.concat([self.gender, self.age, self.bmi], axis=1)

    @property
    def age(self) -> pd.DataFrame:
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def gender(self) -> pd.DataFrame:
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def bmi(self) -> pd.DataFrame:
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

    def calculate_pep_manual_labeled(self) -> pd.DataFrame:
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def reference_pep(self) -> pd.DataFrame:
        """The reference PEP data.

        Returns
        -------
        :class:`~pandas.DataFrame`
            The reference PEP data.

        """
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def reference_heartbeats(self) -> pd.DataFrame:
        """The reference heartbeats.

        Returns
        -------
        :class:`~pandas.DataFrame`
            The reference heartbeats.

        """
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def reference_labels_ecg(self) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """The reference labels for the ECG data.

        Returns
        -------
        :class:`~pandas.DataFrame`
            The reference labels for the ECG data.

        """
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def reference_labels_icg(self) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """The reference labels for the ICG data.

        Returns
        -------
        :class:`~pandas.DataFrame`
            The reference labels for the ICG data.

        """
        raise NotImplementedError("This property needs to be implemented in the subclass!")


@base_pep_extraction_docfiller
class BasePepDatasetWithAnnotations(BasePepDataset, PepLabelMixin, MetadataMixin):
    """Unified interface for datasets used for evaluating PEP extraction algorithms.

    This interface extends the :class:`~pepbench.datasets.BasePepDataset` by adding support for metadata and
    reference PEP data.

    Attributes
    ----------
    %(base_attributes_pep)s
    %(base_attributes_pep_label)s
    %(base_attributes_metadata)s

    """

    def __init__(
        self,
        groupby_cols: list[str] | str | None = None,
        subset_index: pd.DataFrame | None = None,
        return_clean: bool = True,
    ) -> None:
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index, return_clean=return_clean)
