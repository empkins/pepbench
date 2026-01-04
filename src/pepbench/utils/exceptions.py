"""A set of custom exceptions."""

__all__ = ["SamplingRateMismatchError", "ValidationError"]


class SamplingRateMismatchError(Exception):
    """An error indicating a mismatch in sampling rates."""

class ValidationError(Exception):
    """An error indicating a validation failure."""