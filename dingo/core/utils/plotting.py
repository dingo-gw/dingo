import re
from itertools import zip_longest

import matplotlib as mpl
import matplotlib.pyplot as plt
import corner
import numpy as np
import pandas as pd
from bilby.core.prior import PriorDict, Prior


def _sanitize_latex_label(label: str) -> str:
    """Fix common LaTeX label issues for matplotlib mathtext rendering.

    When calibration priors are serialized via repr() and reconstructed via
    PriorDict, backslashes in latex_label strings can get double-escaped
    (e.g., ``$\\phi^H1_0$`` becomes ``$\\\\phi^H1_0$``). Matplotlib's mathtext
    parser cannot parse the double backslash as a LaTeX command.

    Additionally, bilby's CalibrationPriorDict generates superscripts without
    braces for multi-character detector names (``^H1`` instead of ``^{H1}``).

    Parameters
    ----------
    label : str
        A LaTeX label string, typically in math mode.

    Returns
    -------
    str
        The sanitized label.
    """
    if not isinstance(label, str):
        return label
    if not (label.startswith("$") and label.endswith("$")):
        return label

    inner = label[1:-1]

    # Fix double backslashes before LaTeX command names
    inner = re.sub(r"\\\\([a-zA-Z])", r"\\\1", inner)

    # Fix missing braces around multi-character superscripts: ^H1 -> ^{H1}
    inner = re.sub(r"\^([A-Za-z0-9]{2,})(?![}])", r"^{\1}", inner)

    # Fix missing braces around multi-character subscripts: _10 -> _{10}
    inner = re.sub(r"_([A-Za-z0-9]{2,})(?![}])", r"_{\1}", inner)

    return "$" + inner + "$"


def get_latex_labels(prior: PriorDict) -> dict:
    """
    Get the latex labels for prior parameters. If no latex label exists within the
    prior object, try to choose based on parameter key. Finally, return the parameter key.

    Labels are sanitized to fix double-backslash escaping and missing braces that
    can occur with calibration parameters after prior serialization roundtrips.

    Parameters
    ----------
    prior : PriorDict

    Returns
    -------
    dict of latex labels
    """
    labels = {}
    for k, v in prior.items():
        l = v.latex_label
        if l is None:
            l = Prior._default_latex_labels.get(k, k)
        labels[k] = _sanitize_latex_label(l)
    return labels


def plot_corner_multi(
    samples,
    weights=None,
    labels=None,
    filename: str = "corner.pdf",
    latex_labels_dict: dict = None,
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
    latex_labels_dict : dict
        Dictionary of latex labels.

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
    if latex_labels_dict:
        parameter_labels = [latex_labels_dict.get(p, p) for p in common_parameters]
    else:
        parameter_labels = common_parameters

    # Compute a common parameter range across all sample sets. This ensures
    # every corner call uses identical bins, so the 1D marginal densities are
    # on a consistent scale regardless of sample size or weight distribution.
    all_data = pd.concat([s[common_parameters] for s in samples], ignore_index=True)
    common_range = [
        (all_data[p].min(), all_data[p].max()) for p in common_parameters
    ]

    fig = None
    handles = []
    for i, (s, w, l) in enumerate(zip_longest(samples, weights, labels)):
        color = mpl.colors.rgb2hex(plt.get_cmap(cmap)(i))
        # Normalize weights to sum to 1 so that smoothed 1D marginals are
        # comparable across sample sets with different sizes or total weights.
        n = len(s)
        w_normalized = np.ones(n) / n if w is None else np.asarray(w) / np.sum(w)
        fig = corner.corner(
            s[common_parameters].to_numpy(),
            labels=parameter_labels,
            weights=w_normalized,
            color=color,
            no_fill_contours=True,
            range=common_range,
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
