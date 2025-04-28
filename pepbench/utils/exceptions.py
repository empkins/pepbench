"""A set of custom exceptions."""

__all__ = ["SamplingRateMismatchError"]


class SamplingRateMismatchError(Exception):
    """An error indicating a mismatch in sampling rates."""
