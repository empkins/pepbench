"""Helper utilities for loading EmpkinS dataset files.

This module provides small helpers used by the Empkins dataset implementation:
building data paths, loading Biopac recordings, and loading processed timelogs.
Functions assume the dataset layout used by the EmpkinS data export and rely on
`biopsykit` IO utilities.

Notes
-----
Expected layout under the dataset root:
- `data_per_subject/{participant_id}/{condition}/biopac/raw/biopac_data_{participant_id}_{condition}.acq`
- `data_per_subject/{participant_id}/{condition}/timelog/processed/{participant_id}_{condition}_processed_timelog.csv`
"""
from pathlib import Path

import pandas as pd
from biopsykit.io import load_atimelogger_file
from biopsykit.io.biopac import BiopacDataset

from pepbench.utils._types import path_t


def _build_data_path(base_path: path_t, participant_id: str, condition: str) -> Path:
    """Construct path to a participant/condition data directory.

    Parameters
    ----------
    base_path : path-like
        Root path of the EmpkinS dataset.
    participant_id : str
        Participant identifier used in the dataset folder names.
    condition : str
        Experimental condition name.

    Returns
    -------
    :class:`~pathlib.Path`
        Path to `data_per_subject/{participant_id}/{condition}`.

    Raises
    ------
    AssertionError
        If the constructed path does not exist.
    """
    data_path = base_path.joinpath(f"data_per_subject/{participant_id}/{condition}")
    assert data_path.exists()
    return data_path


def _load_biopac_data(base_path: path_t, participant_id: str, condition: str) -> tuple[pd.DataFrame, int]:
    """Load Biopac data for a given participant and condition.

    The function expects an ACQ file at
    `data_per_subject/{participant_id}/{condition}/biopac/raw/biopac_data_{participant_id}_{condition}.acq`.
    It uses `biopsykit.io.biopac.BiopacDataset` to read the file and converts the
    signal to a pandas DataFrame indexed by `local_datetime`.

    Parameters
    ----------
    base_path : :class:`~pathlib.Path`
        Root path of the EmpkinS dataset.
    participant_id : str
        Participant identifier.
    condition : str
        Experimental condition name.

    Returns
    -------
    tuple[:class:`~pandas.DataFrame`, int]
        Tuple of `(biopac_df, sampling_rate)` where `biopac_df` is the Biopac
        channels as a DataFrame indexed by `local_datetime` and `sampling_rate` is
        the per-channel sampling rate (Hz) (one representative value is returned).

    Raises
    ------
    AssertionError
        If the participant/condition directory does not exist.
    FileNotFoundError
        If the expected ACQ file is missing.
    RuntimeError
        If the Biopac file cannot be parsed by `BiopacDataset`.
    """
    biopac_dir_path = _build_data_path(base_path, participant_id=participant_id, condition=condition).joinpath(
        "biopac/raw"
    )

    biopac_file_path = biopac_dir_path.joinpath(f"biopac_data_{participant_id}_{condition}.acq")

    biopac_data = BiopacDataset.from_acq_file(biopac_file_path)
    biopac_df = biopac_data.data_as_df(index="local_datetime")
    fs = next(iter(biopac_data._sampling_rate.values()))
    return biopac_df, fs


def _load_timelog(base_path: path_t, participant_id: str, condition: str, phase: str) -> pd.DataFrame:
    """Load processed timelog entries for a participant/condition and (optional) phase.

    The function reads the CSV produced by the ATIMelogger processing step and
    returns either the timelog columns for a single `phase` or a coarse timelog
    containing the main experimental phases if `phase == "all"`. The CSV is parsed
    with timezone `"Europe/Berlin"`.

    Parameters
    ----------
    base_path : :class:`~pathlib.Path`
        Root path of the EmpkinS dataset.
    participant_id : str
        Participant identifier.
    condition : str
        Experimental condition name.
    phase : str
        Phase name to select (e.g., `'baseline'`, `'task'`) or `'all'` to return a
        coarse timelog.

    Returns
    -------
    :class:`~pandas.DataFrame`
        Timelog rows/columns for the requested phase or a coarse timelog when
        `phase == "all"`.

    Raises
    ------
    AssertionError
        If the participant/condition directory does not exist.
    FileNotFoundError
        If the expected processed timelog CSV is missing.
    """
    timelog_dir_path = _build_data_path(base_path, participant_id=participant_id, condition=condition).joinpath(
        "timelog/processed"
    )
    timelog_file_path = timelog_dir_path.joinpath(f"{participant_id}_{condition}_processed_timelog.csv")
    timelog = load_atimelogger_file(timelog_file_path, timezone="Europe/Berlin")

    if phase == "all":
        timelog_coarse = timelog.drop("Talk_1", axis=1, level=0)
        timelog_coarse = timelog_coarse.drop("Talk_2", axis=1, level=0)
        timelog_coarse = timelog_coarse.drop("Math_1", axis=1, level=0)
        timelog_coarse = timelog_coarse.drop("Math_2", axis=1, level=0)
        return timelog_coarse
    timelog = timelog.iloc[:, timelog.columns.get_level_values(0) == phase]
    return timelog
