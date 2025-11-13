"""Timing utilities for measuring execution duration.

Utilities for simple wall-clock timing using a context manager. The helpers
in this module provide a convenient lightweight way to capture start and end
timestamps together with a measured runtime in seconds. This is intended for
coarse logging and diagnostics, not high-precision benchmarking.

Classes
-------
MeasureTimeResults
    Typed dictionary describing the structure returned by :func:`measure_time`.

Functions
---------
measure_time
    Context manager that yields a dictionary with start/end timestamps and runtime.

Notes
-----
- Timestamps include both a UTC POSIX timestamp and an ISO-8601 local datetime
  string to ease downstream display and logging.
- The measured ``runtime_s`` uses :func:`time.perf_counter` and is suitable for
  elapsed-time measurement but not for sub-microsecond precision benchmarking.

Examples
--------
>>> from pepbench.utils._timing import measure_time
>>> with measure_time() as t:
...     do_work()
>>> print(t["runtime_s"])
0.1234
"""
import contextlib
import time
from collections.abc import Generator
from datetime import datetime
from typing import TypedDict


class MeasureTimeResults(TypedDict):
    """Results returned by :func:`measure_time`.

    Keys
    ----
    start_datetime_utc_timestamp : float
        POSIX UTC timestamp (seconds since epoch) at context entry.
    start_datetime : str
        ISO-8601 localized datetime string at context entry.
    end_datetime_utc_timestamp : float
        POSIX UTC timestamp at context exit.
    end_datetime : str
        ISO-8601 localized datetime string at context exit.
    runtime_s : float
        Elapsed wall-clock time in seconds measured with :func:`time.perf_counter`.
    """

    start_datetime_utc_timestamp: float
    start_datetime: str
    end_start_datetime_utc_timestamp: float
    end_start_datetime: str
    runtime: float


@contextlib.contextmanager
def measure_time() -> Generator[MeasureTimeResults, None, None]:
    """Context manager to measure execution time.

    Yields
    ------
    MeasureTimeResults
        Mutable mapping populated with entry/exit timestamps and measured runtime.

    Notes
    -----
    - Use this for coarse-grained timing and logging. This is not meant for high precision timing.
    - The context manager populates the dictionary before yielding so callers can
      attach additional metadata while the timed block executes.
    """
    results = {
        "start_datetime_utc_timestamp": datetime.utcnow().timestamp(),
        "start_datetime": datetime.now().astimezone().isoformat(),
    }
    start_time = time.perf_counter()
    yield results
    end_time = time.perf_counter()
    results["end_datetime_utc_timestamp"] = datetime.utcnow().timestamp()
    results["end_datetime"] = datetime.now().astimezone().isoformat()
    results["runtime_s"] = end_time - start_time
