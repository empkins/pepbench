r"""Per-sample aggregator utilities for PEP evaluation.

Provides a small adapter around :class:`tpcp.validate.Aggregator` that applies a
user-provided function to horizontally stacked per-sample arrays. This is useful
for scoring functions that produce per-sample results (arrays) and where an
aggregator should operate on the concatenation of those arrays across datapoints.

Classes
-------
PerSampleAggregator
    Aggregator that applies a callable to the horizontally stacked array of per-sample values.

Notes
-----
- The adapter expects the supplied function to accept a single sequence/array
  (typically a 1D or 2D NumPy array) and return either a scalar or a dictionary
  of scalar metrics (e.g. ``{"mean": 0.0, "std": 1.0}``).
- The class preserves the ``return_raw_scores`` behaviour from
  :class:`tpcp.validate.Aggregator`.
"""

from collections.abc import Callable, Sequence

import numpy as np
from tpcp.validate import Aggregator


class PerSampleAggregator(Aggregator[np.ndarray]):
    """Aggregator that applies a function to the concatenated per-sample arrays.

    This aggregator is intended for cases where the scoring function returns
    per-sample values as NumPy arrays for each datapoint. The aggregator stacks
    the provided arrays horizontally and forwards the combined array to the
    user-supplied callable.

    Parameters
    ----------
    func : Callable[[Sequence[np.ndarray]], float | dict[str, float]]
        Callable that accepts a sequence or array of values (typically the result
        of :func:`numpy.hstack`) and returns either a scalar or a mapping of
        scalar metrics.
    return_raw_scores : bool, optional
        Passed to the base :class:`tpcp.validate.Aggregator` initializer and
        controls whether raw (unaggregated) scores are kept. Default is ``True``.

    Examples
    --------
    >>> from pepbench.evaluation._scoring import mean_and_std
    >>> agg = PerSampleAggregator(mean_and_std)
    >>> agg.aggregate([np.array([1.0, 2.0]), np.array([3.0])])
    {'mean': 2.0, 'std': 0.816496580927726}

    """
    def __init__(
        self,
        func: Callable[[Sequence[np.ndarray]], float | dict[str, float]],
        *,
        return_raw_scores: bool = True,
    ) -> None:
        self.func = func
        super().__init__(return_raw_scores=return_raw_scores)

    def aggregate(self, /, values: Sequence[np.ndarray], **_) -> dict[str, float]:  # noqa: ANN003
        """Aggregate a sequence of per-sample arrays.

        The method horizontally stacks the input arrays and calls the user-provided
        function with the resulting array. The callable is expected to return a
        scalar or a dictionary of scalars; this method returns that dictionary.

        Parameters
        ----------
        values : Sequence[np.ndarray]
            Sequence of NumPy arrays produced by the scoring function for each datapoint.

        Returns
        -------
        dict[str, float]
            Aggregated metrics returned by ``func`` applied to the stacked arrays.

        """
        return self.func(np.hstack(values))
