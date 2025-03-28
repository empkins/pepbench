"""Example dataset for testing and demonstration purposes."""

__all__ = ["get_example_dataset"]

from pathlib import Path

from pepbench import __version__
from pepbench.datasets._example_dataset import ExampleDataset

LOCAL_EXAMPLE_PATH = Path(__file__).parent.parent.parent.joinpath("example_data")

PEPPI = None

if not LOCAL_EXAMPLE_PATH.exists():
    # if not (LOCAL_EXAMPLE_PATH / "README.md").is_file():
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


def _pooch_get_folder(folder_path: Path) -> Path:
    """Get the path to the example data folder.

    If the data is not available locally, it will be downloaded from the remote repository.
    For this we use pooch to download all files that start with the folder name.
    """
    if PEPPI is None:
        return folder_path

    rel_folder_path = folder_path.relative_to(LOCAL_EXAMPLE_PATH)

    matching_files = []
    for f in PEPPI.registry:
        try:
            Path(f).relative_to(rel_folder_path)
        except ValueError:
            continue
        matching_files.append(Path(PEPPI.fetch(f, progressbar=True)))

    return PEPPI.abspath / rel_folder_path


def _pooch_get_file(file_path: Path) -> Path:
    """Get the path to the example data file.

    If the data is not available locally, it will be downloaded from the remote repository.
    For this we use pooch to download all files that start with the folder name.

    """
    if PEPPI is None:
        return file_path

    rel_folder_path = file_path.relative_to(LOCAL_EXAMPLE_PATH)

    return Path(PEPPI.fetch(str(rel_folder_path), progressbar=True))


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
    fname = _pooch_get_file(LOCAL_EXAMPLE_PATH.joinpath("example_dataset.zip"))
    return ExampleDataset(example_file_path=fname, return_clean=return_clean)
