"""Helper utilities for the TimeWindow ICG dataset.

This module provides light utilities used by the TimeWindow ICG dataset loader
for reading plain `*.txt` signal files, matching manual annotation points to
computed heartbeat borders, and generating reference heartbeat border files.

Functions operate on pandas objects and use :class:`biopsykit.signals.ecg.segmentation.HeartbeatSegmentationNeurokit`
for heartbeat extraction when generating reference borders.

Notes
-----
- Signal files are expected to contain three columns in the order:
  ICG, ICG derivative and ECG.
- Time indexing and sampling-rate handling are performed by the caller.
"""
import re

import numpy as np
import pandas as pd
from biopsykit.signals.ecg.segmentation import HeartbeatSegmentationNeurokit

from pepbench.utils._types import path_t


def _load_txt_data(file_path: path_t) -> pd.DataFrame:
    """Load a plain text signal file into a pandas DataFrame.

    The function reads a text file with no header and assigns the column names
    ``['icg', 'icg_der', 'ecg']``. The returned DataFrame contains the raw
    numeric samples in these channels and does not set a time index.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the plain text file to read. Each row is expected to contain
        three numeric samples.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``icg``, ``icg_der`` and ``ecg`` in that order.

    Raises
    ------
    OSError
        If the file cannot be read.
    ValueError
        If the file does not contain exactly three columns.
    """
    data = pd.read_csv(file_path, header=None)
    data.columns = ["icg", "icg_der", "ecg"]
    return data


def _get_match_heartbeat_label_ids(heartbeats: pd.DataFrame, b_points: pd.DataFrame) -> pd.Series:
    """Map manual annotation points (B points) to heartbeat identifiers.

    For each B-point in ``b_points`` the function finds the heartbeat in
    ``heartbeats`` whose ``start_sample`` and ``end_sample`` interval contains
    the B-point's ``sample_relative`` value. The output is a :class:`pandas.Series`
    that maps each B-point index (index of ``b_points``) to the integer
    heartbeat index (row index from ``heartbeats``).

    Parameters
    ----------
    heartbeats : pandas.DataFrame
        DataFrame of heartbeat borders. Must contain integer or numeric columns
        ``start_sample`` and ``end_sample`` representing inclusive sample indices
        for each heartbeat.
    b_points : pandas.DataFrame
        DataFrame of annotation points. Must contain a column ``sample_relative``
        with the sample index (relative to the phase start) for each B-point.
        The index of this DataFrame is used as the key in the returned Series.

    Returns
    -------
    pandas.Series
        Series indexed by the B-point indices (the index of ``b_points``) and
        containing the matched heartbeat index (int). B-points without a match
        are omitted from the returned Series.

    Notes
    -----
    - If a B-point lies outside all heartbeat intervals it will be dropped from
      the returned mapping.
    """
    heartbeat_ids = pd.Series(index=heartbeats.index, name="heartbeat_id")
    heartbeat_ids.index.name = "heartbeat_id_b_point"
    for i, b_point in b_points.iterrows():
        idx = np.where(
            (heartbeats["start_sample"] <= b_point["sample_relative"])
            & (heartbeats["end_sample"] >= b_point["sample_relative"])
        )[0]
        if len(idx) == 0:
            # If no match is found, add NaN
            heartbeat_ids[i] = np.nan
        else:
            # If a match is found, add the index of the heartbeat
            heartbeat_ids[i] = idx[0]

    heartbeat_ids = heartbeat_ids.dropna().astype(int)
    return heartbeat_ids


def generate_heartbeat_borders(base_path: path_t) -> None:
    """Generate and save heartbeat border files for all recordings.

    The function scans the expected ``signals`` and ``annotations`` subfolders
    under ``base_path`` for available recordings, computes heartbeat borders
    using :class:`biopsykit.signals.ecg.segmentation.HeartbeatSegmentationNeurokit`
    (sampling rate fixed at 2000 Hz), rounds results to two decimal places and
    writes CSV files into a ``reference_heartbeats`` subfolder.

    Parameters
    ----------
    base_path : str or pathlib.Path
        Root folder of the dataset. The function expects the following
        subfolders to exist:
        - ``signals`` : contains ``IDN<id>.txt`` files with raw signals.
        - ``annotations`` : contains annotation CSV files (used to determine
          which recordings to process).
        - ``reference_heartbeats`` : will be created if missing and receives
          the generated CSV files.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If required input folders (``signals`` or ``annotations``) are missing
        or cannot be read.

    Notes
    -----
    - The internal sampling frequency used for heartbeat extraction is 2000 Hz.
    - Output CSV files are named ``IDN<id>.csv`` and contain the heartbeat
      list as produced by the segmentation algorithm.
    """
    data_folder = base_path.joinpath("signals")
    annotation_folder = base_path.joinpath("annotations")
    heartbeat_folder = base_path.joinpath("reference_heartbeats")
    heartbeat_folder.mkdir(parents=True, exist_ok=True)
    for annotation_path in sorted(annotation_folder.glob("*.csv")):
        matches = re.findall(r"IDN(\d+).csv", str(annotation_path.name))
        p_id = matches[0]
        data_path = data_folder.joinpath(f"IDN{p_id}.txt")
        data = _load_txt_data(data_path)
        fs = 2000

        data.index /= fs
        data.index.name = "t"

        heartbeat_algo = HeartbeatSegmentationNeurokit()
        heartbeat_algo.extract(ecg=data[["ecg"]], sampling_rate_hz=fs)
        heartbeats = heartbeat_algo.heartbeat_list_

        heartbeats = heartbeats.round(2)

        heartbeat_path = heartbeat_folder.joinpath(f"IDN{p_id}.csv")
        heartbeats.to_csv(heartbeat_path)
