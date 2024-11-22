import numpy as np

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
    "drost2022": "Dro22",
    "linear-interpolation": "LinInt",
}
_algorithm_mapping.update(**{f"vanlien2013-{i}-ms": f"Van13 ({i} ms)" for i in np.arange(32, 44, 2)})
_algorithm_mapping.update()

_metric_mapping = {
    "absolute_error_per_sample_ms": "Absolute Error [ms]",
    "error_per_sample_ms": "Error [ms]",
    "relative_error_per_sample_percent": "Relative Error [%]",
}
