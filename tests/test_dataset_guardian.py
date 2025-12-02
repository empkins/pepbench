import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from pepbench.datasets.guardian import GuardianDataset


def _make_minimal_structure(base_path: Path, participants=("GDN0001",)):
    """
    Create minimal folder layout expected by GuardianDataset.create_index:
      - metadata/dataset_overview.csv  (semicolon-separated)
      - metadata/demographics.csv
      - metadata/recording_timestamps.xlsx
    """
    meta_dir = base_path.joinpath("metadata")
    meta_dir.mkdir(parents=True, exist_ok=True)

    overview = pd.DataFrame({"participant": list(participants)})
    overview.to_csv(meta_dir.joinpath("dataset_overview.csv"), index=False, sep=";")

    # demographics with Height in cm (biopsykit expects ~50-250)
    demographics = pd.DataFrame(
        {
            "participant": list(participants),
            "Age": [30 for _ in participants],
            "Gender": ["F" for _ in participants],
            "Weight": [70.0 for _ in participants],
            "Height": [175.0 for _ in participants],
        }
    )
    demographics.to_csv(meta_dir.joinpath("demographics.csv"), index=False)

    # recording timestamps required by GuardianDataset.date (pandas Timestamp; tz_localize happens inside code)
    recording_dates = pd.DataFrame(
        {
            "participant": list(participants),
            "date": [pd.Timestamp("2020-01-01") for _ in participants],
        }
    )
    recording_dates.to_excel(meta_dir.joinpath("recording_timestamps.xlsx"), index=False)


def test_create_index_and_constants(tmp_path):
    base = tmp_path / "guardian"
    _make_minimal_structure(base, participants=("GDN0001", "GDN0002"))

    ds = GuardianDataset(base_path=base)

    idx = ds.create_index()
    expected_rows = 2 * len(ds.PHASES)
    assert idx.shape[0] == expected_rows

    assert isinstance(ds.sampling_rates, dict)
    assert ds.sampling_rate_ecg == ds.SAMPLING_RATES["ecg_2"] == 500
    assert ds.sampling_rate_icg == ds.SAMPLING_RATES["icg_der"] == 500
    assert "Pause" in ds.PHASES


def test_metadata_age_gender_bmi(tmp_path):
    base = tmp_path / "guardian"
    _make_minimal_structure(base, participants=("GDN0001",))

    ds = GuardianDataset(base_path=base)

    meta = ds.metadata
    assert "GDN0001" in meta.index
    assert meta.loc["GDN0001", "Age"] == 30

    age_df = ds.age
    assert age_df.loc["GDN0001", "Age"] == 30

    gender_df = ds.gender
    assert gender_df.loc["GDN0001", "Gender"] == "Female"

    bmi_df = ds.bmi
    assert pytest.approx(bmi_df.loc["GDN0001", "BMI"], rel=1e-3) == 70.0 / ((175.0 / 100) ** 2)


def test_tfm_ecg_icg_with_monkeypatched_loader(tmp_path, monkeypatch):
    base = tmp_path / "guardian"
    _make_minimal_structure(base, participants=("GDN0001",))

    # construct dataset restricted to a single participant+phase via a DataFrame
    subset = pd.DataFrame([{"participant": "GDN0001", "phase": "Pause"}])
    ds = GuardianDataset(base_path=base, return_clean=False, subset_index=subset)

    dummy_df = pd.DataFrame(
        {
            "ecg_1": np.arange(10),
            "ecg_2": np.arange(10) + 100,
            "icg_der": np.zeros(10),
        },
        index=pd.RangeIndex(0, 10),
    )

    monkeypatch.setattr(
        "pepbench.datasets.guardian._dataset._cached_get_tfm_data",
        lambda path, date: {phase: dummy_df for phase in ds.PHASES},
    )

    ecg = ds.ecg  # should select ecg_2 and rename to 'ecg'
    assert "ecg" in ecg.columns
    assert list(ecg["ecg"]) == list(dummy_df["ecg_2"])

    icg = ds.icg  # should return icg_der column
    assert "icg_der" in icg.columns
    assert list(icg["icg_der"]) == list(dummy_df["icg_der"])


def test_labeling_borders_monkeypatched(tmp_path, monkeypatch):
    base = tmp_path / "guardian"
    _make_minimal_structure(base, participants=("GDN0001",))

    subset = pd.DataFrame([{"participant": "GDN0001", "phase": "Pause"}])
    ds = GuardianDataset(base_path=base, subset_index=subset)

    dummy_borders = pd.DataFrame(
        {
            "sample_absolute": [0, 9],
            "description": ["Pause - segment", "Pause - end"],
        },
        index=[0, 9],
    )

    monkeypatch.setattr(
        "pepbench.datasets.guardian._dataset.load_labeling_borders",
        lambda file_path: dummy_borders,
    )

    borders = ds.labeling_borders
    assert isinstance(borders, pd.DataFrame)
    assert "description" in borders.columns
    assert any(borders["description"].str.contains("Pause"))




if __name__ == "__main__":
    pytest.main([__file__])
