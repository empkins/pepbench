"""Helper functions for Guardian datasets.

Utilities for loading and converting Guardian-specific dataset artifacts used by the
Guardian dataset implementation.

Functions
---------
_load_tfm_data
    Load TFM data for a given date and return a dictionary of :class:`~pandas.DataFrame`
    objects indexed by local datetime.

Notes
-----
- The function relies on :class:`~pepbench.datasets.guardian._tfm_loader.TFMLoader`.
- Returned DataFrames use the ``local_datetime`` index.
"""
import pandas as pd

from pepbench.datasets.guardian._tfm_loader import TFMLoader
from pepbench.utils._types import path_t


def _load_tfm_data(base_path: path_t, date: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """
    Load TFM data for a given date.

    Parameters
    ----------
    base_path
    date

    Returns
    -------
    dict[str, pd.DataFrame]
        A dictionary containing TFM data as DataFrames indexed by local datetime.
    """
    tfm_data = TFMLoader.from_mat_file(base_path, date)
    return tfm_data.data_as_dict(index="local_datetime")
