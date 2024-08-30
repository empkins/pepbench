
import pandas as pd

from pepbench._utils._types import path_t
from pepbench.datasets.guardian._tfm_loader import TFMLoader


def _load_tfm_data(base_path: path_t, date: pd.Timestamp) -> dict[str, pd.DataFrame]:
    tfm_data = TFMLoader.from_mat_file(base_path, date)
    return tfm_data.data_as_dict(index="local_datetime")
