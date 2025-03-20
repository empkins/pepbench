"""Example dataset for testing and demonstration purposes."""

__all__ = ["get_example_dataset"]

from pepbench.datasets._example_dataset import ExampleDataset


def get_example_dataset(return_clean: bool = True) -> ExampleDataset:
    """Get an example dataset.

    Parameters
    ----------
    return_clean : bool, optional
        Whether to return cleaned/preprocessed signals when accessing the dataset or not. Default: True
        See the documentation of :class:`~pepbench.datasets.ExampleDataset` for more information.

    """
    return ExampleDataset(return_clean=return_clean)
