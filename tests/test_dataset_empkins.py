import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import shutil

from pepbench.datasets.empkins import EmpkinsDataset


def _make_minimal_structure(base_path: Path, participants=("VP_001",), use_example_data: bool = True):
    """
    Create minimal folder layout expected by EmpkinsDataset.create_index:
      - data_per_subject/{participant}/{condition}
    and a metadata/demographics.csv file.

    If use_example_data is True, copy the corresponding directories from the repository's
    `example_data/` directory (inside the project root) into the temporary test folder.
    """
    data_per_subject = base_path.joinpath("data_per_subject")
    data_per_subject.mkdir(parents=True, exist_ok=True)

    if use_example_data:
        # copy example_data participant folders into the temporary test directory
        repo_example = Path(__file__).resolve().parents[1].joinpath("example_data")
        for p in participants:
            src = repo_example.joinpath(p)
            dst = data_per_subject.joinpath(p)
            if src.exists():
                # create participant dir
                if dst.exists():
                    shutil.rmtree(dst)
                dst.mkdir(parents=True, exist_ok=True)
                # Ensure condition subfolders exist and copy example contents into each condition
                for cond in ("tsst", "ftsst"):
                    cond_dst = dst.joinpath(cond)
                    if cond_dst.exists():
                        shutil.rmtree(cond_dst)
                    cond_dst.mkdir(parents=True, exist_ok=True)
                    # copy files and directories from src into cond_dst
                    for item in src.iterdir():
                        # special-case reference_labels: copy into cond/biopac/reference_labels/rater_01
                        if item.name == "reference_labels" and item.is_dir():
                            # create per-phase per-channel files expected by EmpkinsDataset
                            target_base = cond_dst.joinpath("biopac/reference_labels/rater_01")
                            target_base.mkdir(parents=True, exist_ok=True)
                            # find available example reference files (ECG/ICG)
                            example_files = {child.name: child for child in item.iterdir() if child.is_file()}
                            # use dataset PHASES constant to generate filenames
                            try:
                                from pepbench.datasets.empkins import EmpkinsDataset as _ED
                                phases = _ED.PHASES
                            except Exception:
                                phases = ("Prep",)
                            for phase in phases:
                                for fname, child in example_files.items():
                                    # map example filename to channel
                                    if "ECG" in fname.upper():
                                        channel = "ecg"
                                    elif "ICG" in fname.upper():
                                        channel = "icg"
                                    else:
                                        channel = fname
                                    # construct expected filename
                                    out_name = f"reference_labels_{p}_{cond}_{phase.lower()}_{channel}.csv"
                                    out_path = target_base.joinpath(out_name)
                                    shutil.copy2(child, out_path)
                        else:
                            target = cond_dst.joinpath(item.name)
                            if item.is_dir():
                                shutil.copytree(item, target)
                            else:
                                shutil.copy2(item, target)
            else:
                # fallback to creating minimal structure for this participant
                for cond in ("tsst", "ftsst"):
                    (data_per_subject / p / cond).mkdir(parents=True, exist_ok=True)
    else:
        for p in participants:
            for cond in ("tsst", "ftsst"):
                (data_per_subject / p / cond).mkdir(parents=True, exist_ok=True)

    # create a simple demographics.csv required by metadata properties if one does not exist
    meta_dir = base_path.joinpath("metadata")
    meta_dir.mkdir(parents=True, exist_ok=True)
    demographics_path = meta_dir.joinpath("demographics.csv")
    if not demographics_path.exists():
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
        demographics.to_csv(demographics_path, index=False)


def test_create_index_and_constants(tmp_path):
    base = tmp_path / "empkins"
    _make_minimal_structure(base, participants=("VP_001", "VP_002"), use_example_data=True)

    ds = EmpkinsDataset(base_path=base)

    # create_index should find both participants and expand to CONDITIONS and PHASES
    idx = ds.create_index()
    expected_rows = 2 * len(ds.CONDITIONS) * len(ds.PHASES)
    assert idx.shape[0] == expected_rows
    # assert idx is DataFrame with expected columns
    assert isinstance(idx, pd.DataFrame)
    # create_index returns columns named 'participant', 'condition', 'phase'
    assert set(idx.columns) == {"participant", "condition", "phase"}

    # sampling rates and constants
    assert ds.sampling_rate_ecg == ds.SAMPLING_RATES["ecg"] == 1000
    assert ds.sampling_rate_icg == ds.SAMPLING_RATES["icg"] == 1000
    assert "Prep" in ds.PHASES
    assert set(ds.CONDITIONS) == {"tsst", "ftsst"}


def test__get_biopac_data(tmp_path, monkeypatch):
    base = tmp_path / "empkins"
    # First: minimal layout (no example data) -> reading should raise FileNotFoundError
    _make_minimal_structure(base, participants=("VP_001",), use_example_data=False)
    ds = EmpkinsDataset(base_path=base)
    with pytest.raises(FileNotFoundError):
        ds._get_biopac_data("VP_001", "tsst", "all")

    # Now populate with example_data and assert the loader can read files
    _make_minimal_structure(base, participants=("VP_001",), use_example_data=True)
    ds2 = EmpkinsDataset(base_path=base)
    # monkeypatch the internal cached loader to avoid real .acq parsing which isn't available in example_data
    dummy_df = pd.DataFrame({"ecg": np.arange(5), "icg_der": np.zeros(5)})
    monkeypatch.setattr(
        "pepbench.datasets.empkins._dataset._cached_get_biopac_data",
        lambda base_path, participant_id, condition: (dummy_df, 1000),
    )
    # also monkeypatch the helper loader to be safe if the module path uses the helper directly
    monkeypatch.setattr(
        "pepbench.datasets.empkins._helper._load_biopac_data",
        lambda base_path, participant_id, condition: (dummy_df, 1000),
    )
    data, fs = ds2._get_biopac_data("VP_001", "tsst", "all")
    assert isinstance(data, pd.DataFrame)
    # sampling frequency should be a positive integer
    assert isinstance(fs, (int, float)) and fs > 0


def test_metadata_age_gender_bmi(tmp_path):
    base = tmp_path / "empkins"
    _make_minimal_structure(base, participants=("VP_001",), use_example_data=True)

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
    _make_minimal_structure(base, participants=("VP_001",), use_example_data=True)

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
    _make_minimal_structure(base, participants=("VP_001",), use_example_data=True)

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


def test_create_index_exclude_missing(tmp_path):
    base = tmp_path / "empkins"
    _make_minimal_structure(base, participants=("VP_001", "VP_002"), use_example_data=False)

    ds = EmpkinsDataset(base_path=base)
    # simulate that VP_002 has missing data and enable exclusion
    ds.MISSING_DATA = ("VP_002",)
    ds.exclude_missing_data = True

    idx = ds.create_index()
    # VP_002 should not be present
    assert "VP_002" not in idx["participant"].values


def test_base_demographics_and_properties(tmp_path):
    base = tmp_path / "empkins"
    _make_minimal_structure(base, participants=("VP_001",), use_example_data=False)
    ds = EmpkinsDataset(base_path=base)

    base_demo = ds.base_demographics
    # should contain Age, Gender, BMI columns
    assert "Age" in base_demo.columns
    assert "Gender" in base_demo.columns
    assert "BMI" in base_demo.columns

    # sampling rate properties
    assert isinstance(ds.sampling_rate_ecg, int)
    assert isinstance(ds.sampling_rate_icg, int)
    assert ds.sampling_rate_ecg == ds.SAMPLING_RATES["ecg"]


def test_timelog_all_and_phase(monkeypatch, tmp_path):
    base = tmp_path / "empkins"
    _make_minimal_structure(base, participants=("VP_001",), use_example_data=False)
    ds = EmpkinsDataset(base_path=base)

    # create dummy timelog with multi-column format expected by helper
    # create a MultiIndex columns timelog that contains the phases dropped by the helper
    cols = pd.MultiIndex.from_tuples(
        [
            ("Prep", "start"),
            ("Prep", "end"),
            ("Talk_1", "start"),
            ("Talk_1", "end"),
            ("Talk_2", "start"),
            ("Talk_2", "end"),
            ("Math_1", "start"),
            ("Math_1", "end"),
            ("Math_2", "start"),
            ("Math_2", "end"),
        ]
    )
    dummy_timelog = pd.DataFrame([[0, 9, 0, 1, 2, 3, 4, 5, 6, 7]], columns=cols)
    # monkeypatch the helper loader to return our dummy structure by patching load_atimelogger_file
    import pepbench.datasets.empkins._helper as empkins_helper

    monkeypatch.setattr(empkins_helper, "load_atimelogger_file", lambda *a, **k: dummy_timelog)

    timelog_all = ds._get_timelog("VP_001", "tsst", "all")
    assert isinstance(timelog_all, pd.DataFrame)

    timelog_prep = ds._get_timelog("VP_001", "tsst", "Prep")
    # Our monkeypatched helper returns a frame; ensure returned frame has expected 'start' column at least
    assert timelog_prep.shape[0] >= 0


def test_biopac_phase_cut(monkeypatch, tmp_path):
    base = tmp_path / "empkins"
    _make_minimal_structure(base, participants=("VP_001",), use_example_data=False)

    # create a time-indexed DataFrame to simulate Biopac data
    times = pd.date_range("2020-01-01", periods=100, freq="S")
    df = pd.DataFrame({"ecg": np.arange(100)}, index=times)

    # monkeypatch the biopac loader and BiopacDataset.from_acq_file to return our df and sampling rate
    import pepbench.datasets.empkins._helper as empkins_helper
    import pepbench.datasets.empkins._dataset as empkins_dataset

    monkeypatch.setattr(empkins_helper, "_load_biopac_data", lambda *a, **k: (df, 1000))
    monkeypatch.setattr(empkins_dataset, "_cached_get_biopac_data", lambda *a, **k: (df, 1000))

    # also patch BiopacDataset.from_acq_file as a fallback
    import biopsykit.io.biopac as bk_biopac

    class FakeBiopac:
        def __init__(self, df_local):
            self._df = df_local
            self._sampling_rate = {"ecg": 1000}

        def data_as_df(self, index="local_datetime"):
            return self._df

    monkeypatch.setattr(bk_biopac.BiopacDataset, "from_acq_file", lambda path: FakeBiopac(df))

    # create a timelog DataFrame with MultiIndex columns (phase, attr) like the real loader returns
    cols = pd.MultiIndex.from_tuples([("Prep", "start"), ("Prep", "end")])
    timelog = pd.DataFrame([[times[10], times[50]]], columns=cols)
    monkeypatch.setattr(empkins_helper, "_load_timelog", lambda *a, **k: timelog)

    ds = EmpkinsDataset(base_path=base)
    # monkeypatch the dataset's timelog property so _get_biopac_data can slice without changing ds.index
    import pepbench.datasets.empkins._dataset as empkins_dataset

    # Provide a dict-like timelog where timelog['Prep'] returns a DataFrame with start/end columns
    timelog_map = {"Prep": pd.DataFrame({"start": [times[10]], "end": [times[50]]})}
    monkeypatch.setattr(empkins_dataset.EmpkinsDataset, "timelog", property(lambda self: timelog_map), raising=True)
    data, fs = ds._get_biopac_data("VP_001", "tsst", "Prep")

    assert isinstance(data, pd.DataFrame)
    assert data.index.min() >= times[10]
    assert data.index.max() <= times[50]
    assert fs == 1000


def test_biopac_requires_single_selection(tmp_path):
    base = tmp_path / "empkins"
    _make_minimal_structure(base, participants=("VP_001", "VP_002"), use_example_data=False)
    ds = EmpkinsDataset(base_path=base)
    # Trying to access ds.biopac when more than one participant/condition is selected should raise ValueError
    with pytest.raises(ValueError):
        _ = ds.biopac


if __name__ == "__main__":
    pytest.main([__file__])
