"""Example dataset for testing and demonstration purposes."""

__all__ = ["get_example_dataset"]

from pepbench import __version__
from pathlib import Path

from pepbench.datasets._example_dataset import ExampleDataset

LOCAL_EXAMPLE_PATH = Path(__file__).parent.parent.parent.joinpath("example_data")

PEPPI = None

if not (LOCAL_EXAMPLE_PATH / "README.md").is_file():
    import pooch

    GITHUB_FOLDER_PATH = "https://raw.githubusercontent.com/empkins/pepbench/{version}/example_data/"

    PEPPI = pooch.create(
        # Use the default cache folder for the operating system
        path=pooch.os_cache("pepbench"),
        # The remote data is on GitHub
        base_url=GITHUB_FOLDER_PATH,
        version=f"v{__version__}",
        version_dev="main",
        registry=None,
        # The name of an environment variable that *can* overwrite the path
        env="PEPBENCH_DATA_DIR",
    )

    # Get registry file from package_data
    # The registry file can be recreated by running the task `poe update_example_data`
    registry_file = LOCAL_EXAMPLE_PATH.joinpath("_example_data_registry.txt")
    # Load this registry file
    PEPPI.load_registry(registry_file)


def get_example_dataset(return_clean: bool = True) -> ExampleDataset:
    """Get an example dataset.

    Parameters
    ----------
    return_clean : bool, optional
        Whether to return cleaned/preprocessed signals when accessing the dataset or not. Default: True
        See the documentation of :class:`~pepbench.datasets.ExampleDataset` for more information.

    Returns
    -------
    :class:`~pepbench.datasets.ExampleDataset`
        An example dataset for testing and demonstration purposes.

    """
    fname = PEPPI.fetch("example_dataset.zip")
    return ExampleDataset(return_clean=return_clean)
