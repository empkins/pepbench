from typing import Optional, Union, List

from tpcp import Dataset
import pandas as pd


class BaseUnifiedPepExtractionDataset(Dataset):
    _sampling_rate_ecg: float
    _sampling_rate_icg: float
    _icg: pd.DataFrame
    _ecg: pd.DataFrame
    _clean_icg: pd.DataFrame
    _clean_ecg: pd.DataFrame

    def __init__(
        self, groupby_cols: Optional[Union[List[str], str]] = None, subset_index: Optional[pd.DataFrame] = None
    ):
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    @property
    def icg(self):
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def ecg(self):
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def icg_clean(self):
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def ecg_clean(self):
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def reference_labels_ecg(self):
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def reference_labels_icg(self):
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def sampling_rate_ecg(self):
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def sampling_rate_icg(self):
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    def calculate_pep_manual_labeled(self):
        raise NotImplementedError("This property needs to be implemented in the subclass!")
