"""Heartbeat matching utilities for evaluating heartbeat detection.

The comparison of detected heartbeat lists against reference annotations is a common step when
evaluating heartbeat detection algorithms. This module exposes the matching functionality used
to align reference and detected heartbeats and to derive matching-based metrics.

The primary public function is :func:`pepbench.heartbeat_matching.match_heartbeat_lists`, which
accepts reference and detected heartbeat sequences (lists or arrays) and returns matchings and
summary information according to a configurable tolerance. See
:mod:`pepbench.heartbeat_matching._heartbeat_matching` for implementation details and parameter
documentation.
"""

from pepbench.heartbeat_matching._heartbeat_matching import match_heartbeat_lists

__all__ = ["match_heartbeat_lists"]
