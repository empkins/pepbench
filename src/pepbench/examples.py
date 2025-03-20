__all__ = ["get_example_dataset"]

from pepbench.datasets._example_dataset import ExampleDataset


def get_example_dataset(return_clean: bool = True):
    return ExampleDataset(return_clean=return_clean)
