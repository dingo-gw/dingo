import numpy as np
import scipy
import pandas as pd


def generate_cornerplot(*sample_sets, filename=None):
    try:
        from chainconsumer import ChainConsumer
    except ImportError:
        return

    parameters = [
        p
        for p in sample_sets[0]["samples"].keys()
        if p in set.intersection(*tuple(set(s["samples"].keys()) for s in sample_sets))
    ]
    N = len(sample_sets)

    c = ChainConsumer()
    for s in sample_sets:
        c.add_chain(s["samples"][parameters], color=s["color"], name=s["name"])
    c.configure(
        linestyles=["-"] * N,
        linewidths=[1.5] * N,
        sigmas=[np.sqrt(2) * scipy.special.erfinv(x) for x in [0.5, 0.9]],
        shade=[False] + [True] * (N - 1),
        shade_alpha=0.3,
        bar_shade=False,
        label_font_size=10,
        tick_font_size=10,
        usetex=False,
        legend_kwargs={"fontsize": 30},
        kde=0.7,
    )
    c.plotter.plot(filename=filename)


def load_ref_samples(ref_samples_file, drop_geocent_time=True):
    # Todo: this function should be made more flexible for other formats and parameters
    from bilby.gw.conversion import component_masses_to_chirp_mass

    columns = [
        "mass_1",
        "mass_2",
        "phase",
        "geocent_time",
        "luminosity_distance",
        "a_1",
        "a_2",
        "tilt_1",
        "tilt_2",
        "phi_12",
        "phi_jl",
        "theta_jn",
        "psi",
        "ra",
        "dec",
    ]
    samples = np.load(ref_samples_file, allow_pickle=True)["samples"][:,:15]
    samples = pd.DataFrame(data=samples, columns=columns)
    # add chirp mass and mass ratio.
    Mc = component_masses_to_chirp_mass(samples["mass_1"], samples["mass_2"])
    q = samples["mass_2"] / samples["mass_1"]
    samples.insert(loc=0, column="chirp_mass", value=Mc)
    samples.insert(loc=0, column="mass_ratio", value=q)
    if drop_geocent_time:
        samples.drop(columns="geocent_time", inplace=True)
    return samples
