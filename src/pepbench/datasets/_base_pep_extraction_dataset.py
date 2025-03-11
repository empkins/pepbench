from typing import Optional, Union

import pandas as pd
from tpcp import Dataset


class BaseUnifiedPepExtractionDataset(Dataset):
    _sampling_rate_ecg: float
    _sampling_rate_icg: float
    _icg: pd.DataFrame
    _ecg: pd.DataFrame
    _clean_icg: pd.DataFrame
    _clean_ecg: pd.DataFrame

    def __init__(
        self, groupby_cols: Optional[Union[list[str], str]] = None, subset_index: Optional[pd.DataFrame] = None
    ) -> None:
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    @property
    def icg(self) -> pd.DataFrame:
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def ecg(self) -> pd.DataFrame:
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def icg_clean(self) -> pd.DataFrame:
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def ecg_clean(self) -> pd.DataFrame:
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def reference_pep(self) -> pd.DataFrame:
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def reference_heartbeats(self) -> pd.DataFrame:
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def reference_labels_ecg(self) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def reference_labels_icg(self) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def sampling_rate_ecg(self) -> int:
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def sampling_rate_icg(self) -> int:
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    @property
    def heartbeats(self) -> pd.DataFrame:
        raise NotImplementedError("This property needs to be implemented in the subclass!")

    def calculate_pep_manual_labeled(self) -> pd.DataFrame:
        raise NotImplementedError("This property needs to be implemented in the subclass!")

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
