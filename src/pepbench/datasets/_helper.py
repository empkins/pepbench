"""Helper functions for dataset handling in PepBench.

Provides utility functions used across dataset modules for loading labeling borders
and converting heartbeat segmentation into a reference format.

Functions
---------
:func:`~pepbench.datasets._helper.load_labeling_borders`
    Load labeling borders from a CSV file into a :class:`~pandas.DataFrame`.
:func:`~pepbench.datasets._helper.compute_reference_heartbeats`
    Reformat heartbeat segmentation into per-heartbeat sample columns in a
    :class:`~pandas.DataFrame`.

"""
import ast

import pandas as pd

from pepbench.utils._types import path_t

__all__ = ["compute_reference_heartbeats", "load_labeling_borders"]


def load_labeling_borders(file_path: path_t) -> pd.DataFrame:
    """Load the labeling borders from a CSV file.

    Parameters
    ----------
    file_path : :class:`~pathlib.Path` or str
        Path to the CSV file containing the labeling borders.

    Returns
    -------
    :class:`~pandas.DataFrame`
        The labeling borders as a DataFrame. The function parses the ``description`` column
        using :func:`ast.literal_eval`, sets ``timestamp`` as the index and sorts the index.
    """
    data = pd.read_csv(file_path)
    data = data.assign(description=data["description"].apply(lambda s: ast.literal_eval(s)))

    data = data.set_index("timestamp").sort_index()
    return data


def compute_reference_heartbeats(heartbeats: pd.DataFrame) -> pd.DataFrame:
    """Reformat the heartbeats :class:`~pandas.DataFrame`.

    Parameters
    ----------
    heartbeats : :class:`~pandas.DataFrame`
        DataFrame containing heartbeat segmentation. Expected to have a MultiIndex with a
        ``channel`` level and a ``label`` level, and a ``sample_relative`` column.

    Returns
    -------
    :class:`~pandas.DataFrame`
        DataFrame containing the reformatted heartbeats where ``sample_relative`` values are
        unstacked by ``label`` and column names are suffixed with ``_sample``.
    """
    heartbeats = heartbeats.droplevel("channel")["sample_relative"].unstack("label")
    heartbeats.columns = [f"{col}_sample" for col in heartbeats.columns]
    return heartbeats
