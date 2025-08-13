"""Evaluation challenges for systematic evaluation of PEP extraction pipelines.

The systematic evaluation of different PEP extraction pipelines requires a standardized evaluation procedure.
This can be achieved by defining a challenge that evaluates a pipeline on a given dataset and returns the results.
For that, `tpcp` provides the necessary functionality to define such challenges and compare the results of different
pipelines, implemented in :class:`pepbench.evaluation.PepEvaluationChallenge`.

The evaluation challenge takes a dataset
and a scoring function and evaluates the performance of a PEP extraction pipeline on the given dataset. While the
scoring function can be customized, the default scoring function is provided in
:func:`pepbench.evaluation.score_pep_evaluation`.

"""

from pepbench.evaluation._evaluation import ChallengeResults, PepEvaluationChallenge
from pepbench.evaluation._scoring import score_pep_evaluation

__all__ = ["ChallengeResults", "PepEvaluationChallenge", "score_pep_evaluation"]
