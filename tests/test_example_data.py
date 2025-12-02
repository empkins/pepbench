from contextlib import contextmanager
import pytest
from pathlib import Path

from tpcp import Dataset

from pepbench.example_data import get_example_dataset


@contextmanager
def does_not_raise():
    yield


class TestExampleData:
    def test_get_example_data(self) -> None:
        dataset = get_example_dataset()
        assert isinstance(dataset, Dataset)

    def test_vp001_ecg_and_reference_heartbeats(self) -> None:
        dataset = get_example_dataset()
        # ensure get_subset for a known participant does not raise and returns a Dataset
        with does_not_raise():
            subset = dataset.get_subset(participant="VP_001")
        assert isinstance(subset, Dataset)

        # subset should expose an 'ecg' attribute and that object should have a plot method
        assert hasattr(subset, "ecg")
        ecg_obj = getattr(subset, "ecg")
        assert callable(getattr(ecg_obj, "plot", None))

        # subset should expose 'reference_heartbeats' and it should be non-empty
        assert hasattr(subset, "reference_heartbeats")
        rh = getattr(subset, "reference_heartbeats")
        # support any iterable/sequence type: try len(), fallback to iterating once
        try:
            assert len(rh) > 0
        except TypeError:
            # if rh is an iterator, convert to list
            rh_list = list(rh)
            assert len(rh_list) > 0

    def test_invalid_participant_raises(self) -> None:
        dataset = get_example_dataset()
        # tpcp raises KeyError when a filter value is not present in the index
        with pytest.raises(KeyError):
            dataset.get_subset(participant="VP_invalid")

    def test_data_files_exist(self) -> None:
        dataset = get_example_dataset()
        # derive example_data directory relative to repository root (cwd)
        data_dir = Path.cwd() / "example_data"
        assert data_dir.exists() and data_dir.is_dir()

        # Check for expected files in the data directory
        expected_files = [
            "VP_001/vp_001_ecg_data.csv",
            "VP_001/vp_001_icg_data.csv",
            "VP_001/reference_labels/VP_001_labeling_borders.csv",
            "VP_001/reference_labels/VP_001_reference_labels_ECG.csv",
            "VP_001/reference_labels/VP_001_reference_labels_ICG.csv",
            # TODO: add more expected files as needed
        ]

        for file_name in expected_files:
            file_path = data_dir / file_name
            assert file_path.exists(), f"Expected file {file_name} does not exist in {data_dir}"


if __name__ == "__main__":
    pytest.main([__file__])
