import ast

import pandas as pd

from pepbench.utils._types import path_t

__all__ = ["compute_reference_heartbeats", "load_labeling_borders"]


def load_labeling_borders(file_path: path_t) -> pd.DataFrame:
    """Load the labeling borders from a csv file.

    Parameters
    ----------
    file_path : :class:`pathlib.Path` or str
        The path to the csv file.

    Returns
    -------
    :class:`pandas.DataFrame`
        The labeling borders.

    """
    data = pd.read_csv(file_path)
    data = data.assign(description=data["description"].apply(lambda s: ast.literal_eval(s)))

    data = data.set_index("timestamp").sort_index()
    return data


def compute_reference_heartbeats(heartbeats: pd.DataFrame) -> pd.DataFrame:
    """Reformat the heartbeats DataFrame.

    Parameters
    ----------
    heartbeats : :class:`pandas.DataFrame`
        DataFrame containing the heartbeats.

    Returns
    -------
    :class:`pandas.DataFrame`
        DataFrame containing the reformatted heartbeats.

    """
    heartbeats = heartbeats.droplevel("channel")["sample_relative"].unstack("label")
    heartbeats.columns = [f"{col}_sample" for col in heartbeats.columns]
    return heartbeats
