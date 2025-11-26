"""Utility functions for data handling."""

import pandas as pd

__all__ = ["reindex_empkins", "reindex_guardian", "rename_empkins", "rename_guardian"]

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


def reindex_empkins(data: pd.DataFrame, after_rename: bool = False) -> pd.DataFrame:
    """Reindex data from the :class:`~pepbench.datasets.guardian._dataset.EmpkinsDataset.

    The reindexing is performed according to the condition (tsst, ftsst) and phase (Prep, Pause_1, Talk, Math, Pause_5)
    mappings. The reindexing can be performed *before* or *after* the data has been renamed
    (using :func:`rename_empkins`).

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        Data from the :class:`~pepbench.datasets.guardian._dataset.EmpkinsDataset.
    after_rename : bool, optional
        ``True`` if the data has already been renamed using :func:`rename_empkins`, ``False`` otherwise.
        Default: ``False``

    Returns
    -------
    :class:`pandas.DataFrame`
        Reindexed data from the :class:`~pepbench.datasets.guardian._dataset.EmpkinsDataset.

    """
    if after_rename:
        return data.reindex(condition_mapping_empkins.values(), level="condition").reindex(
            phase_mapping_empkins.values(), level="phase"
        )
    return data.reindex(condition_mapping_empkins.keys(), level="condition").reindex(
        phase_mapping_empkins.keys(), level="phase"
    )


def rename_empkins(data: pd.DataFrame) -> pd.DataFrame:
    """Rename the data from the :class:`~pepbench.datasets.guardian._dataset.EmpkinsDataset.

    The renaming is performed according to the condition (tsst -> TSST, ftsst -> f-TSST) and phase
    (Prep -> Preparation, Pause_1 -> Pause 1, Talk -> Talk, Math -> Math, Pause_5 -> Pause 5) mappings.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        Data from the :class:`~pepbench.datasets.guardian._dataset.EmpkinsDataset.

    Returns
    -------
    :class:`pandas.DataFrame`
        Renamed data from the :class:`~pepbench.datasets.guardian._dataset.EmpkinsDataset.

    """
    return data.rename(condition_mapping_empkins, level="condition").rename(phase_mapping_empkins, level="phase")


def reindex_guardian(data: pd.DataFrame, after_rename: bool | None = False) -> pd.DataFrame:
    """Reindex data from the :class:`~pepbench.datasets.guardian._dataset.GuardianDataset`.

    The reindexing is performed according to the phase (Pause, Valsalva, HoldingBreath, TiltUp, TiltDown) mappings.
    The reindexing can be performed *before* or *after* the data has been renamed (using :func:`rename_guardian`).

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        Data from the :class:`~pepbench.datasets.guardian._dataset.GuardianDataset.
    after_rename : bool, optional
        ``True`` if the data has already been renamed using :func:`rename_guardian`, ``False`` otherwise.
        Default: ``False``

    Returns
    -------
    :class:`pandas.DataFrame`
        Reindexed data from the :class:`~pepbench.datasets.guardian._dataset.GuardianDataset.

    """
    if after_rename:
        return data.reindex(phase_mapping_guardian.values(), level="phase")
    return data.reindex(phase_mapping_guardian.keys(), level="phase")


def rename_guardian(data: pd.DataFrame) -> pd.DataFrame:
    """Rename the data from the :class:`~pepbench.datasets.guardian._dataset.GuardianDataset.

    The renaming is performed according to the phase (Pause -> Pause, Valsalva -> Valsalva, HoldingBreath -> Apnea,
    TiltUp -> Tilt-Up, TiltDown -> Tilt-Down) mappings.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        Data from the :class:`~pepbench.datasets.guardian._dataset.GuardianDataset.

    Returns
    -------
    :class:`pandas.DataFrame`
        Renamed data from the :class:`~pepbench.datasets.guardian._dataset.GuardianDataset.

    """
    return data.rename(phase_mapping_guardian, level="phase")
