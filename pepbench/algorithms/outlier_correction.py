"""Module for ICG outlier correction algorithms."""

from biopsykit.signals.icg.outlier_correction import (
    OutlierCorrectionDummy,
    OutlierCorrectionForouzanfar2018,
    OutlierCorrectionInterpolation,
)

__all__ = ["OutlierCorrectionDummy", "OutlierCorrectionInterpolation", "OutlierCorrectionForouzanfar2018"]
