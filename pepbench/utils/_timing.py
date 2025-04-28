import contextlib
import time
from collections.abc import Generator
from datetime import datetime
from typing import TypedDict


class MeasureTimeResults(TypedDict):
    """Results of the measure_time context manager."""

    start_datetime_utc_timestamp: float
    start_datetime: str
    end_start_datetime_utc_timestamp: float
    end_start_datetime: str
    runtime: float


@contextlib.contextmanager
def measure_time() -> Generator[MeasureTimeResults, None, None]:
    """Context manager to measure the execution time.

    Note: This is not meant for high precision timing.
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
