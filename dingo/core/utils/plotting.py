from itertools import zip_longest
from typing import Optional
import matplotlib as mpl
import matplotlib.pyplot as plt
import corner
import numpy as np
import pandas as pd

LATEX_PARAM_DICT = {
    "mass1": r"$m_1$",
    "mass2": r"$m_2$",
    "chirp_mass": r"$\mathcal{M}_c~[\text{M}_{\odot}]$",
    "mass_ratio": r"$q$",
    "a_1": r"$a_1$",
    "a_2": r"$a_2$",
    "tilt_1": r"$\theta_1$",
    "tilt_2": r"$\theta_2$",
    "phi_12": r"$\phi_{12}$",
    "phi_jl": r"$\phi_{jl}$",
    "theta_jn": r"$\theta_{jn}$",
    "luminosity_distance": r"$d_L$",
    "geocent_time": r"$t_c$",
    "ra": r"$\alpha$",
    "dec": r"$\delta$",
    "psi": r"$\psi$",
    "phase": r"$\phi$",
    "chi_eff": r"$\chi_{\mathrm{eff}}$",
    "chi_p": r"$\chi_p$",
    "log_prob": r"$\log p$",
    "log_prior": r"$\log p(\theta)$",
    "log_likelihood": r"$\log p(x|\theta)$",
    "weights": r"$w$",
    "H1_time_proxy": r"$\hat{t}(\mathrm{H})$",
    "L1_time_proxy": r"$\hat{t}(\mathrm{L})$",
    "V1_time_proxy": r"$\hat{t}(\mathrm{V})$",
}


def plot_corner_multi(
    samples,
    weights=None,
    labels: Optional[list[str]] = None,
    colors: Optional[list[str]] = None,
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
    colors: list[str or None] or None
        Colors for the posteriors.
    filename : str
        Where to save samples.
    latex_labels_dict : dict
        Dictionary of latex labels.

    Other Parameters
    ----------------
    legend_kwargs: dict
        Parameters passed to `fig.legend()`.
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
    if "truths" in corner_params and "truth_color" not in corner_params:
        corner_params["truth_color"] = "black"

    serif_old = mpl.rcParams["font.family"]
    mpl.rcParams["font.family"] = "serif"
    linewidth_old = mpl.rcParams["lines.linewidth"]
    mpl.rcParams["lines.linewidth"] = 2.5

    # Set default fontsize for x- and y-labels
    if "label_kwargs" not in kwargs:
        kwargs["label_kwargs"] = {"fontsize": 16}
    elif "fontsize" not in kwargs["label_kwargs"]:
        kwargs["label_kwargs"]["fontsize"] = 16

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
        parameter_labels = [
            LATEX_PARAM_DICT[p] if p in LATEX_PARAM_DICT else p
            for p in common_parameters
        ]

    fig = None
    handles = []
    for i, (s, w, l) in enumerate(zip_longest(samples, weights, labels)):
        if colors is not None:
            color = colors[i]
        else:
            color = mpl.colors.rgb2hex(plt.get_cmap(cmap)(i))
        fig = corner.corner(
            s[common_parameters].to_numpy(),
            labels=parameter_labels,
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
        labelcolor="linecolor",
        **kwargs["legend_kwargs"],
    )

    # Customize tick properties for each axis (corner doesn't allow this)
    for i, ax in enumerate(fig.get_axes()):
        if "label_kwargs" in kwargs and "fontsize" in kwargs["label_kwargs"]:
            fontsize = kwargs["label_kwargs"]["fontsize"]
            # Scale factors (empirically chosen)
            label_size = fontsize * 0.875
            tick_length = fontsize * 0.375
            tick_width = fontsize * 0.01
        else:
            label_size = 14
            tick_length = 6
            tick_width = 1.5
        if ax.get_xlabel():
            ax.tick_params(
                axis="x", labelsize=label_size, length=tick_length, width=tick_width
            )  # Adjust labelsize, length, and width
        else:
            ax.tick_params(axis="x", which="both", bottom=False)
        if ax.get_ylabel():
            ax.tick_params(
                axis="y", labelsize=label_size, length=tick_length, width=tick_width
            )  # Adjust labelsize, length, and width
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
