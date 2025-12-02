import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from pepbench.datasets.empkins._dataset import EmpkinsDataset


def _make_minimal_structure(base_path: Path, participants=("VP_001",)):
    """
    Create minimal folder layout expected by EmpkinsDataset.create_index:
      - data_per_subject/{participant}/{condition}
    and a metadata/demographics.csv file.
    """
    data_per_subject = base_path.joinpath("data_per_subject")
    data_per_subject.mkdir(parents=True, exist_ok=True)

    for p in participants:
        for cond in ("tsst", "ftsst"):
            (data_per_subject / p / cond).mkdir(parents=True, exist_ok=True)

    # create a simple demographics.csv required by metadata properties
    meta_dir = base_path.joinpath("metadata")
    meta_dir.mkdir(parents=True, exist_ok=True)
    demographics = pd.DataFrame(
        {
            "participant": list(participants),
            "Age": [30 for _ in participants],
            "Gender": [1 for _ in participants],  # maps to "Female" by GENDER_MAPPING
            "Weight": [70.0 for _ in participants],
            # biopsykit expects Height in cm (range ~50-250) — use 175 cm instead of 1.75 m
            "Height": [175.0 for _ in participants],
        }
    )
    demographics.to_csv(meta_dir.joinpath("demographics.csv"), index=False)


def test_create_index_and_constants(tmp_path):
    base = tmp_path / "empkins"
    _make_minimal_structure(base, participants=("VP_001", "VP_002"))

    ds = EmpkinsDataset(base_path=base)

    # create_index should find both participants and expand by CONDITIONS and PHASES
    idx = ds.create_index()
    expected_rows = 2 * len(ds.CONDITIONS) * len(ds.PHASES)
    assert idx.shape[0] == expected_rows

    # sampling rates and constants
    assert ds.sampling_rate_ecg == ds.SAMPLING_RATES["ecg"] == 1000
    assert ds.sampling_rate_icg == ds.SAMPLING_RATES["icg"] == 1000
    assert "Prep" in ds.PHASES
    assert set(ds.CONDITIONS) == {"tsst", "ftsst"}


def test_metadata_age_gender_bmi(tmp_path):
    base = tmp_path / "empkins"
    _make_minimal_structure(base, participants=("VP_001",))

    ds = EmpkinsDataset(base_path=base)

    # metadata should load and index by participant
    meta = ds.metadata
    assert "VP_001" in meta.index
    assert meta.loc["VP_001", "Age"] == 30

    # age and gender properties
    age_df = ds.age
    assert age_df.loc["VP_001", "Age"] == 30

    gender_df = ds.gender
    # mapping 1 -> "Female"
    assert gender_df.loc["VP_001", "Gender"] == "Female"

    # bmi should compute from Weight and Height (Height now in cm)
    bmi_df = ds.bmi
    # BMI = 70 / ((175/100)^2)
    assert pytest.approx(bmi_df.loc["VP_001", "BMI"], rel=1e-3) == 70.0 / ((175.0 / 100) ** 2)


def test__get_biopac_data_uses_monkeypatched_loader(tmp_path, monkeypatch):
    base = tmp_path / "empkins"
    _make_minimal_structure(base, participants=("VP_001",))

    ds = EmpkinsDataset(base_path=base)

    # prepare a small dummy biopac dataframe
    dummy_df = pd.DataFrame(
        {
            "ecg": np.arange(10),
            "icg_der": np.zeros(10),
        },
        index=pd.RangeIndex(0, 10),
    )

    # monkeypatch the cached loader in the dataset module to avoid reading .acq files
    monkeypatch.setattr(
        "pepbench.datasets.empkins._dataset._cached_get_biopac_data",
        lambda base_path, participant_id, condition: (dummy_df, 1000),
    )

    data, fs = ds._get_biopac_data("VP_001", "tsst", "all")
    assert fs == 1000
    assert "ecg" in data.columns
    assert list(data["ecg"]) == list(range(10))


def test__get_timelog_monkeypatched(tmp_path, monkeypatch):
    base = tmp_path / "empkins"
    _make_minimal_structure(base, participants=("VP_001",))

    ds = EmpkinsDataset(base_path=base)

    # create a dummy timelog DataFrame that the loader should return
    dummy_timelog = pd.DataFrame({"start": [0], "end": [9]}, index=pd.Index(["Prep"], name="phase"))

    # patch the internal _load_timelog used by the dataset module
    monkeypatch.setattr(
        "pepbench.datasets.empkins._dataset._load_timelog",
        lambda base_path, participant_id, condition, phase: dummy_timelog if phase == "Prep" else dummy_timelog,
    )

    result = ds._get_timelog("VP_001", "tsst", "Prep")
    assert isinstance(result, pd.DataFrame)
    assert "start" in result.columns
    assert result["start"].iloc[0] == 0


if __name__ == "__main__":
    pytest.main([__file__])
