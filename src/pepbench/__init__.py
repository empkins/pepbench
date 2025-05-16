"""Top-level module for the PEPBench package."""

from importlib import metadata

from pepbench import algorithms, datasets, evaluation, pipelines

__version__ = metadata.version(__name__)

__all__ = ["algorithms", "datasets", "evaluation", "pipelines"]
