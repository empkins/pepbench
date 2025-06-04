"""Module for plotting PEP ML-Estimator results"""

import inspect
from collections.abc import Callable, Sequence
from typing import Any

import biopsykit as bp
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from fau_colors import cmaps, colors_all
from matplotlib import pyplot as plt

from pepbench.data_handling import get_data_for_algo
from pepbench.data_handling._data_handling import get_performance_metric, get_reference_data
from pepbench.plotting._base_plotting import _plot_blandaltman, _plot_paired
from pepbench.plotting._utils import _get_fig_ax, _get_fig_axs, _remove_duplicate_legend_entries
from pepbench.utils._rename_maps import (
    _algo_level_mapping,
    _algorithm_mapping,
    _metric_mapping,
    _xlabel_mapping,
    _ylabel_mapping,
)

__all__ = [
    "boxplot_algorithm_performance",
]

best_ml_estimators = {
    "SScaler-SFM-SVR-RR": ('StandardScaler', 'SelectFromModel', 'SVR'),
    "SScaler-SKB-RFR": ('StandardScaler', 'SelectKBest', 'RandomForestRegressor'),
    "SScaler-SFM-RFR": ('StandardScaler', 'SelectFromModel', 'RandomForestRegressor'),
    "SScaler-SFM-SVR": ('StandardScaler', 'SelectFromModel', 'SVR'),
}

def boxplot_algorithm_performance(

) -> tuple[plt.Figure, plt.Axes]: