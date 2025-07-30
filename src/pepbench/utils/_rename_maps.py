import numpy as np

from pepbench.utils._types import str_t

__all__ = [
    "get_nan_reason_mapping",
    "rename_algorithms",
    "rename_metrics",
]

_ylabel_mapping = {
    "pep_ms": "PEP [ms]",
    "rr_interval_ms": "RR-Interval [ms]",
    "error_per_sample_ms": "Error [ms]",
    "absolute_error_per_sample_ms": "Absolute Error [ms]",
    "absolute_relative_error_per_sample_percent": "Absolute Relative Error [%]",
}

_xlabel_mapping = {
    "phase": "Phase",
    "participant": "Participant",
    "condition": "Condition",
}

_algo_level_mapping = {
    "q_peak_algorithm": "Q-Peak Detection",
    "b_point_algorithm": "B-Point Detection",
    "outlier_correction_algorithm": "Outlier Correction",
}

_algorithm_mapping = {
    "b-point-reference": "Ref. B-Point",
    "q-peak-reference": "Ref. Q-Peak",
    "none": "None",
    "stern1985": "Ste85",
    "sherwood1990": "She90",
    "debski1993-second-derivative": "Deb93SD",
    "martinez2004": "Mar04",
    "lozano2007-linear-regression": "Loz07LR",
    "lozano2007-quadratic-regression": "Loz07QR",
    "arbol2017-isoelectric-crossings": "Arb17IC",
    "arbol2017-second-derivative": "Arb17SD",
    "arbol2017-third-derivative": "Arb17TD",
    "forouzanfar2018": "For18",
    "miljkovic2022": "Mil22",
    "pale2021": "Pal21",
    "drost2022": "Dro22",
    "linear-interpolation": "LinInt",
}
_algorithm_mapping.update(**{f"vanlien2013-{i}-ms": f"Van13 ({i} ms)" for i in np.arange(32, 44, 2)})
_algorithm_mapping.update()

_metric_mapping = {
    "mean": "Mean",
    "std": "SD",
    "total": "Total",
    "absolute_error_per_sample_ms": "Absolute Error [ms]",
    "error_per_sample_ms": "Error [ms]",
    "relative_error_per_sample_percent": "Relative Error [%]",
}

_nan_reason_mapping = {
    "c_point_nan": "No C-point detected",
    "r_peak_nan": "No R-peak detected",
    "negative_pep": "Negative PEP",
    "no_iso_crossing_before_c_point": "No Isoelectric Crossing before C-point detected",
    "no_local_minimum": "No Local Minimum detected",
    "no_monotonic_segment": "No Monotonically Increasing Segment detected",
    "no_zero_crossing": "No Zero Crossings detected",
    "invalid_b_point_search_window": "Invalid B-point Search Window",
}

_nan_reason_mapping_short = {
    "c_point_nan": "No C",
    "r_peak_nan": "No R",
    "negative_pep": "Neg. PEP",
    "no_iso_crossing_before_c_point": "No IC",
    "no_local_minimum": "No Loc. Min.",
    "no_monotonic_segment": "No Mon. Incr.",
    "no_zero_crossing": "No ZCs",
    "invalid_b_point_search_window": "Inv. B Window",
}


def rename_metrics(metrics: str_t) -> str_t:
    if isinstance(metrics, str):
        return _metric_mapping.get(metrics, metrics)
    return [_metric_mapping.get(metric, metric) for metric in metrics]


def rename_algorithms(algorithms: str_t) -> str_t:
    if isinstance(algorithms, str):
        return _algorithm_mapping.get(algorithms, algorithms)
    return [_algorithm_mapping.get(algo, algo) for algo in algorithms]


def get_nan_reason_mapping() -> dict:
    return {_nan_reason_mapping_short[k]: _nan_reason_mapping[k] for k in _nan_reason_mapping}
