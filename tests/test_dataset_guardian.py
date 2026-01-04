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


def test_create_index_exclude_missing(tmp_path):
    base = tmp_path / "guardian"
    _make_minimal_structure(base, participants=("GDN0001", "GDN0002"))

    ds = GuardianDataset(base_path=base)
    # simulate that GDN0002 has missing data and exclude it
    # exclude GDN0002 for all phases so they are entirely removed from the index
    ds.data_to_exclude = tuple(("GDN0002", ph) for ph in ds.PHASES)

    idx = ds.create_index()
    assert "GDN0002" not in idx["participant"].values


def test_tfm_phase_cut_only_labeled(monkeypatch, tmp_path):
    base = tmp_path / "guardian"
    _make_minimal_structure(base, participants=("GDN0001",))

    # create a subset (single participant + single phase)
    subset = pd.DataFrame([{"participant": "GDN0001", "phase": "Pause"}])
    ds = GuardianDataset(base_path=base, subset_index=subset, only_labeled=True, return_clean=False)

    # create dummy tfm data indexed by sample (0..99)
    dummy = pd.DataFrame({"ecg_2": np.arange(100), "icg_der": np.zeros(100)}, index=pd.RangeIndex(0, 100))

    # monkeypatch loader to return full-phase dict
    monkeypatch.setattr(
        "pepbench.datasets.guardian._dataset._cached_get_tfm_data",
        lambda path, date: {phase: dummy.copy() for phase in ds.PHASES},
    )

    # create labeling borders with start/end that should trim the data
    dummy_borders = pd.DataFrame({"sample_absolute": [10, 50], "description": ["Pause start", "Pause end"]}, index=[10, 50])
    monkeypatch.setattr("pepbench.datasets.guardian._dataset.load_labeling_borders", lambda fp: dummy_borders)

    # accessing tfm_data should return a cut DataFrame for phase 'Pause'
    data = ds.tfm_data
    assert isinstance(data, pd.DataFrame)
    assert data.index.min() >= 10
    assert data.index.max() <= 50


def test_tfm_requires_single_selection(tmp_path):
    base = tmp_path / "guardian"
    _make_minimal_structure(base, participants=("GDN0001", "GDN0002"))
    ds = GuardianDataset(base_path=base)

    # tfm_data should raise if multiple participants are selected
    with pytest.raises(ValueError):
        _ = ds.tfm_data


def test_reference_labels_and_pep(tmp_path):
    base = tmp_path / "guardian"
    _make_minimal_structure(base, participants=("GDN0001",))

    # create subset selecting GDN0001 / Pause
    subset = pd.DataFrame([{"participant": "GDN0001", "phase": "Pause"}])
    ds = GuardianDataset(base_path=base, subset_index=subset)

    # synthesize heartbeats and reference labels in the exact MultiIndex shape the pipeline expects
    # create heartbeats with column names the pipeline expects BEFORE prefixing
    # Base code prefixes `heartbeat_` to these column names, so provide 'start_sample' and 'end_sample'
    heartbeats = pd.DataFrame({"start_sample": [0], "end_sample": [999], "r_peak_sample": [10]}, index=[0])
    q_peaks = pd.DataFrame({"sample_relative": [100]}, index=[0])
    b_points = pd.DataFrame({"sample_relative": [160]}, index=[0])

    # build pep_reference from these labeled values
    pep_reference = heartbeats.copy()
    pep_reference.columns = [f"heartbeat_{col}" if col != "r_peak_sample" else "r_peak_sample" for col in heartbeats.columns]
    pep_reference = pep_reference.assign(q_peak_sample=q_peaks["sample_relative"], b_point_sample=b_points["sample_relative"], nan_reason=pd.NA)
    pep_reference = pep_reference.assign(pep_sample=pep_reference["b_point_sample"] - pep_reference["q_peak_sample"])
    pep_reference = pep_reference.assign(pep_ms=pep_reference["pep_sample"] / ds.sampling_rate_ecg * 1000)

    assert "pep_ms" in pep_reference.columns
    assert pep_reference["pep_ms"].iloc[0] >= 0


if __name__ == "__main__":
    pytest.main([__file__])
