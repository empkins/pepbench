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

check_data_is_df
    Validate that an object is a :class:`pandas.DataFrame` and raise
    :class:`pepbench.utils.exceptions.ValidationError` if not. Useful to perform
    precondition checks in library functions that expect a DataFrame input.

check_data_is_series
    Validate that an object is a :class:`pandas.Series` and raise
    :class:`pepbench.utils.exceptions.ValidationError` if not.

check_data_is_BasePepDatasetWithAnnotations
    Runtime validator that imports and checks whether an object is an
    instance of :class:`pepbench.datasets._base_pep_extraction_dataset.BasePepDatasetWithAnnotations`.
    The import is delayed inside the function to avoid circular imports at module
    import time.

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
from pepbench.utils.exceptions import ValidationError

import numpy as np
import pandas as pd

__all__ = ["arr_t",
           "check_file_exists",
           "path_t",
           "str_t",
           "is_str_t",
           "check_data_is_str_t",
           "check_data_is_df",
           "check_data_is_series",
           "check_data_is_BasePepDatasetWithAnnotations"]

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

def check_data_is_df(data: object) -> None:
    """
    Raise ValidationError if data is not a pandas DataFrame.

    Parameters
    ----------
    data : object
        The data to check.

    Raises
    ------
    ValidationError
        If data is not a pandas DataFrame.

    """
    if not isinstance(data, pd.DataFrame):
        raise ValidationError(f"Expected data to be a pandas DataFrame, got {type(data)} instead.")

def check_data_is_series(data: object) -> None:
    """
    Raise TypeError if data is not a pandas Series.

    Parameters
    ----------
    data : object
        The data to check.

    Raises
    ------
    ValidationError
        If data is not a pandas Series.

    """
    if not isinstance(data, pd.Series):
        raise ValidationError(f"Expected data to be a pandas Series, got {type(data)} instead.")

def check_data_is_BasePepDatasetWithAnnotations(data: object) -> None:
    """
    Raise ValidationError if data is not an instance of BasePepDatasetWithAnnotations.

    The import is performed inside the function to avoid circular imports at module import time.

    Parameters
    ----------
    data : object
        The object to validate.

    Raises
    ------
    ValidationError
        If data is not an instance of BasePepDatasetWithAnnotations.
    """
    from pepbench.datasets._base_pep_extraction_dataset import BasePepDatasetWithAnnotations

    if not isinstance(data, BasePepDatasetWithAnnotations):
        raise ValidationError(
            f"Expected data to be a BasePepDatasetWithAnnotations, got {type(data)} instead."
        )

def is_str_t(value: object) -> bool:
    """Return True when ``value`` conforms to the ``str_t`` type alias.

    ``str_t`` is defined as either a :class:`str` or a sequence of strings.

    Rules implemented:
    - A bare ``str`` returns True.
    - ``bytes``/``bytearray`` are considered non-text sequences and return False.
    - Any other ``collections.abc.Sequence`` is accepted only if all
      its elements are :class:`str`.

    The function is defensive and returns False for non-sequences and
    for sequences containing non-string elements.
    """
    # accept direct string
    if isinstance(value, str):
        return True
    # explicitly reject bytes-like
    if isinstance(value, (bytes, bytearray)):
        return False

    # pandas: accept Index and Series if their non-null elements are strings
    if isinstance(value, (pd.Series, pd.Index)):
        # take numpy array of values
        arr = value.to_numpy()
        # empty -> accept
        if arr.size == 0:
            return True
        # if numpy has string dtype
        if np.issubdtype(arr.dtype, np.str_):
            return True
        # otherwise check non-null elements for Python str
        for item in arr:
            if pd.isna(item):
                continue
            if not isinstance(item, str):
                return False
        return True

    # numpy arrays: accept if string dtype or all non-null items are str
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return True
        if np.issubdtype(value.dtype, np.str_):
            return True
        if value.dtype == object:
            for item in value:
                if pd.isna(item):
                    continue
                if not isinstance(item, str):
                    return False
            return True
        return False

    # for other sequences, ensure every element is str
    if isinstance(value, Sequence):
        try:
            return all(isinstance(item, str) for item in value)
        except TypeError:
            # not iterable in the expected way
            return False
    return False


def check_data_is_str_t(value: object) -> None:
    """Raise :class:`ValidationError` if ``value`` is not a valid ``str_t``.

    This mirrors the other ``check_...`` helpers in this module and is useful
    for validating user inputs or function parameters that accept either a
    single string or a list/tuple of strings.
    """
    if not is_str_t(value):
        raise ValidationError(f"Expected a str or sequence of str, got {type(value)} instead.")

def check_data_is_patht(value: object) -> None:
    """Raise :class:`ValidationError` if ``value`` is not a valid ``path_t``.

    This mirrors the other ``check_...`` helpers in this module and is useful
    for validating user inputs or function parameters that accept either a
    string or a :class:`pathlib.Path`.
    """
    if not isinstance(value, (str, Path)):
        raise ValidationError(f"Expected a str or pathlib.Path, got {type(value)} instead.")
