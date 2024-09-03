from typing import Union, Sequence

from matplotlib import pyplot as plt


def _get_fig_ax(**kwargs) -> tuple[plt.Figure, plt.Axes]:
    ax: Union[plt.Axes, None] = kwargs.pop("ax", None)

    if ax is not None:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(**kwargs)
    return fig, ax


def _get_fig_axs(**kwargs) -> tuple[plt.Figure, Sequence[plt.Axes]]:
    axs: Union[plt.Axes, None] = kwargs.pop("axs", None)

    if axs is not None:
        fig = axs[0].get_figure()
    else:
        fig, axs = plt.subplots(**kwargs)
    return fig, axs
