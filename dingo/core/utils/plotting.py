import re
from itertools import zip_longest

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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


# Target widths in inches for PRL figures.
_TARGET_WIDTHS = {
    "column": 3.375,
    "full": 7.0,
}

# Reference apparent sizes (points) that look good at the final rendered width.
# These are multiplied by (fig_width / target_width) to get actual sizes in the
# figure, so that after the PDF is scaled to target_width they appear at these
# reference values.
_PUB_REF = {
    "label_size": 9,
    "tick_label_size": 8,
    "linewidth": 0.8,
    "contour_linewidth": 1.0,
    "tick_length": 3.5,
    "tick_width": 0.6,
    "legend_size": 10,
    "legend_linewidth": 3,
    "legend_markersize": 8,
    "truth_linewidth": 0.6,
}


def plot_corner_multi(
    samples,
    weights=None,
    labels=None,
    filename: str = "corner.pdf",
    latex_labels_dict: dict = None,
    target_width=None,
    truths=None,
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
    target_width : str or float or None
        If set, scale fonts, linewidths, and tick sizes so the figure looks good
        when rendered at this width. Accepts "column" (~3.4 in), "full" (~7 in),
        or a width in inches. If None, use legacy (unscaled) sizes.
    truths : dict or None
        Dictionary mapping parameter names to true values. If provided, truth
        lines are drawn on the corner plot.

    Other Parameters
    ----------------
    legend_font_size: int
        Font size used in legend (only used when target_width is None).
    Also contains additional parameters forwarded to corner.corner.
    """
    # Define plot properties
    cmap = "tab10"
    corner_params = {
        "smooth": 1.0,
        "smooth1d": 1.0,
        "plot_datapoints": False,
        "plot_density": False,
        "plot_contours": True,
        "levels": [0.5, 0.9],
        "bins": 30,
        "max_n_ticks": 4,
        "labelpad": 0.07 if target_width is not None else 0.0,
    }
    corner_params.update(kwargs)

    serif_old = mpl.rcParams["font.family"]
    mpl.rcParams["font.family"] = "serif"
    linewidth_old = mpl.rcParams["lines.linewidth"]

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
    ndim = len(common_parameters)

    if latex_labels_dict:
        parameter_labels = [latex_labels_dict.get(p, p) for p in common_parameters]
    else:
        parameter_labels = common_parameters

    # Resolve target width and compute scale factor.
    if target_width is not None:
        if isinstance(target_width, str):
            target_width_in = _TARGET_WIDTHS[target_width]
        else:
            target_width_in = float(target_width)
        # corner uses ~2 inches per panel by default
        fig_width = ndim * 2.0
        scale = fig_width / target_width_in
    else:
        scale = None

    if scale is not None:
        mpl.rcParams["lines.linewidth"] = _PUB_REF["contour_linewidth"] * scale
    else:
        mpl.rcParams["lines.linewidth"] = 2.5

    # Compute a common parameter range across all sample sets. This ensures
    # every corner call uses identical bins, so the 1D marginal densities are
    # on a consistent scale regardless of sample size or weight distribution.
    all_data = pd.concat([s[common_parameters] for s in samples], ignore_index=True)
    common_range = [
        (all_data[p].min(), all_data[p].max()) for p in common_parameters
    ]

    # Resolve truth values: accept dict (explicit param) or list (via kwargs).
    # Pop from corner_params so truths are only drawn once (on the first call).
    kwargs_truths = corner_params.pop("truths", None)
    if truths is not None:
        truth_values = [truths.get(p) for p in common_parameters]
    elif kwargs_truths is not None:
        truth_values = kwargs_truths
    else:
        truth_values = None

    fig = None
    handles = []
    legend_lw = _PUB_REF["legend_linewidth"] * scale if scale else 5
    legend_ms = _PUB_REF["legend_markersize"] * scale if scale else 20
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
            truths=truth_values if i == 0 else None,
            truth_color="0.3",
            fig=fig,
            **corner_params,
        )
        handles.append(
            plt.Line2D(
                [], [], color=color, label=l, linewidth=legend_lw, markersize=legend_ms
            )
        )

    # Eliminate spacing between the 2D plots
    if target_width is not None or ndim > 8:
        fig.subplots_adjust(wspace=0, hspace=0)
    else:
        space = 1 / (4 * ndim)
        fig.subplots_adjust(wspace=space, hspace=space)

    if scale is not None:
        legend_fs = _PUB_REF["legend_size"] * scale
    else:
        legend_fs = kwargs.get("legend_font_size", ndim * 5)

    fig.legend(
        handles=handles,
        loc="upper right",
        fontsize=legend_fs,
        labelcolor="linecolor",
    )

    # Customize tick and label properties for each axis
    if scale is not None:
        tick_ls = _PUB_REF["tick_label_size"] * scale
        tick_len = _PUB_REF["tick_length"] * scale
        tick_w = _PUB_REF["tick_width"] * scale
        label_fs = _PUB_REF["label_size"] * scale
        truth_lw = _PUB_REF["truth_linewidth"] * scale
    else:
        tick_ls, tick_len, tick_w = 14, 6, 1.5
        label_fs = 16
        truth_lw = None

    tick_pad = 4 * scale if scale else 2
    
    for ax in fig.get_axes():
        n_ticks = corner_params.get("max_n_ticks", 4)
        tick_margin = corner_params.get("tick_margin_frac", 0.05)

        x_locator = MaxNLocator(nbins=n_ticks, min_n_ticks=2, prune="lower")
        y_locator = MaxNLocator(nbins=n_ticks, min_n_ticks=2, prune="lower")

        if ax.get_xlabel():
            ax.xaxis.set_major_locator(x_locator)
            ax.tick_params(
                axis="x", labelsize=tick_ls, length=tick_len, width=tick_w,
                pad=tick_pad,
            )
            ax.xaxis.label.set_size(label_fs)
            ax.xaxis.labelpad = 10

            # Remove ticks too close to the axis boundaries
            xlim = ax.get_xlim()
            xrange = xlim[1] - xlim[0]
            margin = xrange * tick_margin
            ax.set_xticks([
                t for t in ax.get_xticks()
                if (t - xlim[0]) > margin and (xlim[1] - t) > margin
            ])
        else:
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        if ax.get_ylabel():
            ax.yaxis.set_major_locator(y_locator)
            ax.tick_params(
                axis="y", labelsize=tick_ls, length=tick_len, width=tick_w,
                pad=tick_pad,
            )
            ax.yaxis.label.set_size(label_fs)
            ax.yaxis.labelpad = 10

            # Remove ticks too close to the axis boundaries
            ylim = ax.get_ylim()
            yrange = ylim[1] - ylim[0]
            margin = yrange * tick_margin
            ax.set_yticks([
                t for t in ax.get_yticks()
                if (t - ylim[0]) > margin and (ylim[1] - t) > margin
            ])
        else:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)

        # Style truth lines
        if truth_lw is not None:
            for line in ax.get_lines():
                if line.get_linestyle() in ["--", "dashed"]:
                    line.set_linewidth(truth_lw)

        ax.grid(False)

    # Save the figure
    plt.savefig(filename, bbox_inches="tight")

    # Reset rcParams to original values
    mpl.rcParams["font.family"] = serif_old
    mpl.rcParams["lines.linewidth"] = linewidth_old

    return fig
