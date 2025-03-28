from contextlib import contextmanager

from tpcp import Dataset

from pepbench.example_data import get_example_dataset


@contextmanager
def does_not_raise():
    yield


class TestExampleData:
    def test_get_example_data(self):
        dataset = get_example_dataset()
        assert isinstance(dataset, Dataset)
