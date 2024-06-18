from itertools import zip_longest

import matplotlib as mpl
import matplotlib.pyplot as plt
import corner
import numpy as np
import pandas as pd


def plot_corner_multi(
    samples,
    weights=None,
    labels=None,
    filename: str = "corner.pdf",
    **kwargs,
):
    """
    Generate a corner plot for multiple posteriors.

    Parameters
    ----------
    samples : list[pd.DataFrame]
        List of sample sets. The DataFrame column names are used as parameter labels.
    weights : list[np.ndarray or None] or None
        List of weights sets. The length of each array should be the same as the length of
        the corresponding samples.
    labels : list[str or None] or None
        Labels for the posteriors.
    filename : str
        Where to save samples.

    Other Parameters
    ----------------
    legend_font_size: int
        Font size used in legend. Defaults to 50.
    Also contains additional parameters forwarded to corner.corner.
    """
    # Define plot properties
    cmap = "Dark2"
    corner_params = {
        "smooth": 1.0,
        "smooth1d": 1.0,
        "plot_datapoints": False,
        "plot_density": False,
        "plot_contours": True,
        "levels": [0.5, 0.9],
        "bins": 30,
    }
    corner_params.update(kwargs)

    serif_old = mpl.rcParams["font.family"]
    mpl.rcParams["font.family"] = "serif"
    linewidth_old = mpl.rcParams["lines.linewidth"]
    mpl.rcParams["lines.linewidth"] = 2.5

    # In case a single corner plot is desired, convert to lists to iterate.
    if not isinstance(samples, list):
        samples = [samples]
    if not isinstance(weights, list):
        weights = [weights]
    if not isinstance(labels, list):
        labels = [labels]

    # Only plot common parameters for all sample sets, keeping the same order as the
    # first sample set.
    common_parameters = [
        p
        for p in samples[0].columns
        if p in set.intersection(*(set(s.columns) for s in samples))
    ]

    fig = None
    handles = []
    for i, (s, w, l) in enumerate(zip_longest(samples, weights, labels)):
        color = mpl.colors.rgb2hex(plt.get_cmap(cmap)(i))
        fig = corner.corner(
            s[common_parameters].to_numpy(),
            labels=common_parameters,
            weights=w,
            color=color,
            no_fill_contours=True,
            fig=fig,
            **corner_params,
        )
        handles.append(
            plt.Line2D([], [], color=color, label=l, linewidth=5, markersize=20)
        )

    # Eliminate spacing between the 2D plots
    if len(common_parameters) > 8:
        fig.subplots_adjust(wspace=0, hspace=0)
    else:
        space = 1 / (4 * len(common_parameters))
        fig.subplots_adjust(wspace=space, hspace=space)

    fig.legend(
        handles=handles,
        loc="upper right",
        fontsize=kwargs.get("legend_font_size", 50),
        labelcolor="linecolor",
    )

    # Customize tick and label properties for each axis
    for i, ax in enumerate(fig.get_axes()):
        if ax.get_xlabel():
            ax.tick_params(
                axis="x", labelsize=14, length=6, width=1.5
            )  # Adjust labelsize, length, and width
            ax.xaxis.label.set_size(16)  # Adjust x-axis label font size
        else:
            ax.tick_params(axis="x", which="both", bottom=False)
        if ax.get_ylabel():
            ax.tick_params(
                axis="y", labelsize=14, length=6, width=1.5
            )  # Adjust labelsize, length, and width
            ax.yaxis.label.set_size(16)  # Adjust x-axis label font size
        else:
            ax.tick_params(axis="y", which="both", left=False)

        # Turn off grid
        ax.grid()

    # Save the figure
    plt.savefig(filename)

    # Reset rcParams to original values
    mpl.rcParams["font.family"] = serif_old
    mpl.rcParams["lines.linewidth"] = linewidth_old

    return fig
