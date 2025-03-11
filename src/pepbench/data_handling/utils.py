from typing import Optional

import pandas as pd

__all__ = ["reindex_empkins", "rename_empkins", "reindex_guardian", "rename_guardian"]

condition_mapping_empkins = {"tsst": "TSST", "ftsst": "f-TSST"}
phase_mapping_empkins = {
    "Prep": "Preparation",
    "Pause_1": "Pause 1",
    "Talk": "Talk",
    "Math": "Math",
    "Pause_5": "Pause 5",
}
phase_mapping_guardian = {
    "Pause": "Pause",
    "Valsalva": "Valsalva",
    "HoldingBreath": "Apnea",
    "TiltUp": "Tilt-Up",
    "TiltDown": "Tilt-Down",
}


def reindex_empkins(data: pd.DataFrame, after_rename: Optional[bool] = False) -> pd.DataFrame:
    if after_rename:
        return data.reindex(condition_mapping_empkins.values(), level="condition").reindex(
            phase_mapping_empkins.values(), level="phase"
        )
    return data.reindex(condition_mapping_empkins.keys(), level="condition").reindex(
        phase_mapping_empkins.keys(), level="phase"
    )


def rename_empkins(data: pd.DataFrame) -> pd.DataFrame:
    return data.rename(condition_mapping_empkins, level="condition").rename(phase_mapping_empkins, level="phase")


def reindex_guardian(data: pd.DataFrame, after_rename: Optional[bool] = False) -> pd.DataFrame:
    if after_rename:
        return data.reindex(phase_mapping_guardian.values(), level="phase")
    return data.reindex(phase_mapping_guardian.keys(), level="phase")


def rename_guardian(data: pd.DataFrame) -> pd.DataFrame:
    return data.rename(phase_mapping_guardian, level="phase")
