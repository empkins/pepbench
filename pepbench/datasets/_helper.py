import ast

import pandas as pd

from pepbench._utils._types import path_t


def load_labeling_borders(file_path: path_t) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    data = data.assign(description=data["description"].apply(lambda s: ast.literal_eval(s)))

    data = data.set_index("timestamp").sort_index()
    return data
