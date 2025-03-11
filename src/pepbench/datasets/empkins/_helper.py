from pathlib import Path

import pandas as pd
from biopsykit.io import load_atimelogger_file
from biopsykit.io.biopac import BiopacDataset

from pepbench.utils._types import path_t


def _build_data_path(base_path: path_t, participant_id: str, condition: str) -> Path:
    data_path = base_path.joinpath(f"data_per_subject/{participant_id}/{condition}")
    assert data_path.exists()
    return data_path


def _load_biopac_data(base_path: path_t, participant_id: str, condition: str) -> tuple[pd.DataFrame, int]:
    biopac_dir_path = _build_data_path(base_path, participant_id=participant_id, condition=condition).joinpath(
        "biopac/raw"
    )

    biopac_file_path = biopac_dir_path.joinpath(f"biopac_data_{participant_id}_{condition}.acq")

    biopac_data = BiopacDataset.from_acq_file(biopac_file_path)
    biopac_df = biopac_data.data_as_df(index="local_datetime")
    fs = next(iter(biopac_data._sampling_rate.values()))
    return biopac_df, fs


def _load_timelog(base_path: path_t, participant_id: str, condition: str, phase: str) -> pd.DataFrame:
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
