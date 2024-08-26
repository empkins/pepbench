"""A set of custom exceptions."""

__all__ = ["SamplingRateMismatchException"]


class SamplingRateMismatchException(Exception):
    """An error indicating a mismatch in sampling rates."""
