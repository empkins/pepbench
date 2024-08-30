"""Module for the evaluation of PEP extraction pipelines."""

from pepbench.evaluation._evaluation import PepEvaluationChallenge
from pepbench.evaluation._scoring import score_pep_evaluation

__all__ = ["PepEvaluationChallenge", "score_pep_evaluation"]
