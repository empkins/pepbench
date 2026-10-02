"""Tests for the :mod:`pepbench.example_data` module."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from tpcp import Dataset

from pepbench.example_data import get_example_dataset


@contextmanager
def does_not_raise() -> Iterator[None]:
    """Context manager that does nothing, used to mark code that is expected not to raise."""
    yield


class TestExampleData:
    """Tests for loading the example dataset."""

    def test_get_example_data(self) -> None:
        """Test that the example dataset is a tpcp Dataset."""
        dataset = get_example_dataset()
        assert isinstance(dataset, Dataset)

    def test_vp001_ecg_and_reference_heartbeats(self) -> None:
        """Test that VP_001 exposes ECG data and reference heartbeats."""
        dataset = get_example_dataset()
        # ensure get_subset for a known participant does not raise and returns a Dataset
        with does_not_raise():
            subset = dataset.get_subset(participant="VP_001")
        assert isinstance(subset, Dataset)

        # subset should expose an 'ecg' attribute and that object should have a plot method
        assert hasattr(subset, "ecg")
        ecg_obj = subset.ecg
        assert callable(getattr(ecg_obj, "plot", None))

        # subset should expose 'reference_heartbeats' and it should be non-empty
        assert hasattr(subset, "reference_heartbeats")
        rh = subset.reference_heartbeats
        # support any iterable/sequence type: try len(), fallback to iterating once
        try:
            assert len(rh) > 0
        except TypeError:
            # if rh is an iterator, convert to list
            rh_list = list(rh)
            assert len(rh_list) > 0

    def test_invalid_participant_raises(self) -> None:
        """Test that selecting an unknown participant raises a KeyError."""
        dataset = get_example_dataset()
        # tpcp raises KeyError when a filter value is not present in the index
        with pytest.raises(KeyError):
            dataset.get_subset(participant="VP_invalid")

    def test_data_files_exist(self) -> None:
        """Test that the expected example data files exist."""
        # make sure the example data is available before checking the files
        get_example_dataset()
        # derive example_data directory relative to repository root (cwd)
        data_dir = Path.cwd() / "example_data"
        assert data_dir.exists()
        assert data_dir.is_dir()

        # Check for expected files in the data directory
        expected_files = [
            "VP_001/vp_001_ecg_data.csv",
            "VP_001/vp_001_icg_data.csv",
            "VP_001/reference_labels/VP_001_labeling_borders.csv",
            "VP_001/reference_labels/VP_001_reference_labels_ECG.csv",
            "VP_001/reference_labels/VP_001_reference_labels_ICG.csv",
            "VP_002/vp_002_ecg_data.csv.gz",
            "VP_002/vp_002_icg_data.csv.gz",
            "VP_002/reference_labels/VP_002_labeling_borders.csv",
            "VP_002/reference_labels/VP_002_reference_labels_ECG.csv",
            "VP_002/reference_labels/VP_002_reference_labels_ICG.csv",
        ]

        for file_name in expected_files:
            file_path = data_dir / file_name
            assert file_path.exists(), f"Expected file {file_name} does not exist in {data_dir}"


if __name__ == "__main__":
    pytest.main([__file__])
