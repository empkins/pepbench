"""Project-specific typing helpers and small utilities for type-checked code.

This module collects a few convenient type aliases and a small runtime helper
used across the package to simplify type hints and common checks.

Types
-----
path_t
    Type alias for path-like inputs accepted by functions (``str`` or
    :class:`pathlib.Path`).
str_t
    Type alias for string-or-sequence-of-strings inputs.
arr_t
    Type alias for array-like objects used in the project (:class:`pandas.DataFrame`,
    :class:`pandas.Series` or :class:`numpy.ndarray`).

Functions
---------
check_file_exists
    Ensure a file path exists, raising :class:`FileNotFoundError` otherwise.

Notes
-----
- Type aliases are implemented using :class:`typing.TypeVar` to keep compatibility
  with type checkers while remaining concise in annotations.
- The helper ``check_file_exists`` normalizes the input to :class:`pathlib.Path`
  and raises a clear error message containing the absolute path.

Examples
--------
>>> from pepbench.utils._types import check_file_exists
>>> check_file_exists("data/example.csv")  # raises FileNotFoundError if missing

See Also
--------
:mod:`pepbench.utils._timing`
    Small timing utilities used in diagnostics.
"""

from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd

__all__ = ["arr_t", "check_file_exists", "path_t", "str_t"]

# Helper alias allowing Hashable or raw str (useful for keys that may be strings)
_Hashable = Hashable | str

# Public type aliases for concise annotations across the codebase
path_t = TypeVar("path_t", str, Path)  # pylint:disable=invalid-name
str_t = TypeVar("str_t", str, Sequence[str])  # pylint:disable=invalid-name
arr_t = TypeVar("arr_t", pd.DataFrame, pd.Series, np.ndarray)  # pylint:disable=invalid-name
T = TypeVar("T")


def check_file_exists(file_path: path_t) -> None:
    """Raise :class:`FileNotFoundError` if ``file_path`` does not exist.

    The function accepts either a :class:`pathlib.Path` or a string-like path.
    It converts the input to :class:`pathlib.Path` and raises a
    :class:`FileNotFoundError` with the absolute path for easier debugging.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the file to check.

    Raises
    ------
    FileNotFoundError
        If the resolved absolute path does not point to an existing file.

    Examples
    --------
    >>> check_file_exists("nonexistent.csv")
    Traceback (most recent call last):
        ...
    FileNotFoundError: No file /full/path/nonexistent.csv exists!
    """
    # ensure pathlib
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"No file {file_path.absolute()} exists!")
