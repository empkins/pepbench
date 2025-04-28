from collections.abc import Sequence
from typing import Callable, Union

import numpy as np
from tpcp.validate import Aggregator


class PerSampleAggregator(Aggregator[np.ndarray]):
    def __init__(
        self,
        func: Callable[[Sequence[np.ndarray]], Union[float, dict[str, float]]],
        *,
        return_raw_scores: bool = True,
    ) -> None:
        self.func = func
        super().__init__(return_raw_scores=return_raw_scores)

    def aggregate(self, /, values: Sequence[np.ndarray], **_) -> dict[str, float]:  # noqa: ANN003
        return self.func(np.hstack(values))
