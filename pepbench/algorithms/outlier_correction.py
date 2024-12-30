"""Module for ICG outlier correction algorithms."""

from biopsykit.signals.icg.outlier_correction import (
    OutlierCorrectionDummy,
    OutlierCorrectionForouzanfar2018,
    OutlierCorrectionLinearInterpolation,
)

__all__ = ["OutlierCorrectionDummy", "OutlierCorrectionLinearInterpolation", "OutlierCorrectionForouzanfar2018"]
