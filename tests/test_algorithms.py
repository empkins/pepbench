from contextlib import contextmanager
import importlib
import pytest

from pepbench import algorithms as alg_pkg
from pepbench.example_data import get_example_dataset


@contextmanager
def does_not_raise():
    yield

class TestAlgorithms:

    def test_algorithms_package_exports(self):
        expected_submodules = [
            "ecg",
            "icg",
            "heartbeat_segmentation",
            "preprocessing",
            "outlier_correction",
        ]
        for name in expected_submodules:
            assert hasattr(alg_pkg, name), f"pepbench.algorithms is missing submodule {name}"


    def test_submodule_all_members_callable(self):
        submodules = [
            "ecg",
            "icg",
            "heartbeat_segmentation",
            "preprocessing",
            "outlier_correction",
        ]
        for sub in submodules:
            module = importlib.import_module(f"pepbench.algorithms.{sub}")
            names = getattr(module, "__all__", None)
            assert names is not None and len(names) > 0, f"{sub}.__all__ should be defined and non-empty"
            for item in names:
                attr = getattr(module, item, None)
                assert attr is not None, f"{item} listed in {sub}.__all__ but not importable"
                assert callable(attr), f"{item} in {sub} is not callable"


    def test_preprocessing_classes_instantiable(self):
        mod = importlib.import_module("pepbench.algorithms.preprocessing")
        names = getattr(mod, "__all__", [])
        # try to instantiate each exported preprocessing class; skip on TypeError (requires args)
        for cls_name in names:
            cls = getattr(mod, cls_name, None)
            assert cls is not None, f"{cls_name} not found in preprocessing module"
            try:
                with does_not_raise():
                    instance = cls()
                    assert instance is not None
            except TypeError:
                pytest.skip(f"{cls_name} requires constructor arguments; skipping instantiation check")


    def test_example_data_and_ecg_algorithm_instantiation(self):
        dataset = get_example_dataset()
        subset = dataset.get_subset(participant="VP_001")
        assert hasattr(subset, "ecg")
        # ensure at least one ECG algorithm class can be instantiated
        ecg_mod = importlib.import_module("pepbench.algorithms.ecg")
        ecg_names = getattr(ecg_mod, "__all__", [])
        assert len(ecg_names) > 0, "No ECG algorithms exported"
        for name in ecg_names:
            cls = getattr(ecg_mod, name, None)
            assert cls is not None
            # try to instantiate; skip if constructor requires args
            try:
                with does_not_raise():
                    inst = cls()
                    assert inst is not None
                # only need to check the first instantiable one
                break
            except TypeError:
                # try next algorithm
                continue
        else:
            pytest.skip("No ECG algorithm could be instantiated without arguments; skipping runtime instantiation checks")

if __name__ == "__main__":
    pytest.main([__file__])