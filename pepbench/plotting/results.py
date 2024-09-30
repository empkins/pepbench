from typing import Optional

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from pepbench.plotting._utils import _get_fig_ax

_ylabel_mapping = {
    "pep_ms": "PEP [ms]",
    "rr_interval_ms": "RR-Interval [ms]",
    "error_per_sample_ms": "Error [ms]",
    "absolute_error_per_sample_ms": "Absolute Error [ms]",
    "absolute_relative_error_per_sample_percent": "Absolute Relative Error [%]",
}

_xlabel_mapping = {"phase": "Phase", "participant": "Participant", "condition": "Condition"}


def plot_reference_pep(
    data: pd.DataFrame, x: str, y: Optional[str] = "pep_ms", hue: Optional[str] = None, **kwargs: dict
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = _get_fig_ax(**kwargs)

    if hue is None:
        sns.boxplot(data=data.reset_index(), x=x, y=y, ax=ax)
    else:
        sns.boxplot(data=data.reset_index(), x=x, y=y, hue=hue, ax=ax)

    ax.set_ylabel(_ylabel_mapping[y])
    ax.set_xlabel(_xlabel_mapping[x])

    fig.tight_layout()
    return fig, ax
