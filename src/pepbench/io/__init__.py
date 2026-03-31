"""Input/output helpers for challenge results and unit conversions.

This module provides convenience functions for loading and saving evaluation challenge results and for
simple unit conversions related to sampling rates. The helper functions are thin wrappers around the
implementations in :mod:`pepbench.io._io` and are intended to provide a stable public API for users of
the package.

The primary public functions are:

- :func:`pepbench.io.convert_hz_to_ms` — convert a frequency in Hertz to a period in milliseconds.
- :func:`pepbench.io.load_challenge_results_from_folder` — load evaluation results stored in a folder
  and return them in the package's canonical format.

See :mod:`pepbench.io._io` for implementation details.

"""
from pepbench.io._io import convert_hz_to_ms, load_challenge_results_from_folder

__all__ = ["convert_hz_to_ms", "load_challenge_results_from_folder"]
