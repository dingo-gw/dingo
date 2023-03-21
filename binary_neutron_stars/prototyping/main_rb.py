import numpy as np

np.random.seed(1)
import yaml
import matplotlib.pyplot as plt
from dingo.gw.domains import build_domain, FrequencyDomain
from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.waveform_generator import WaveformGenerator
from dingo.gw.dataset.generate_dataset import generate_parameters_and_polarizations

from multibanded_frequency_domain import MultibandedFrequencyDomain
from multibanding_utils import (
    get_periods,
    number_of_zero_crossings,
    get_decimation_bands_adaptive,
    get_decimation_bands_from_chirp_mass,
    duration_LO,
)
from heterodyning import factor_fiducial_waveform, change_heterodyning

if __name__ == "__main__":
    num_processes = 9

    with open("waveform_dataset_settings.yaml", "r") as f:
        settings = yaml.safe_load(f)
    ufd = build_domain(settings["domain"])
    prior = build_prior_with_defaults(settings["intrinsic_prior"])
    waveform_generator = WaveformGenerator(domain=ufd, **settings["waveform_generator"])

    # generate polarizations
    parameters, polarizations = generate_parameters_and_polarizations(
        waveform_generator, prior, settings["num_samples"], num_processes
    )
    # heterodyne with LO chirp phase
    sf = np.max(np.abs(polarizations["h_plus"]))
    polarizations_het = {
        k: factor_fiducial_waveform(v, ufd, np.array(parameters["chirp_mass"])) / sf
        for k, v in polarizations.items()
    }

    bands = get_decimation_bands_adaptive(
        ufd,
        np.concatenate(list(polarizations_het.values())),
        min_num_bins_per_period=8,
        delta_f_max=3.0,
    )
    x = ufd()
    mfd = MultibandedFrequencyDomain(bands, ufd)
    print(len(mfd))
    print(bands)
    hp_het = polarizations_het["h_plus"]
    fig = plt.figure()
    fig.set_size_inches((8, 8))
    plt.plot(
        x, np.min(get_periods(hp_het.real, upper_bound_for_monotonicity=False), axis=0)
    )
    plt.plot(
        x, np.min(get_periods(hp_het.real, upper_bound_for_monotonicity=True), axis=0)
    )
    plt.yscale("log")
    plt.ylabel("f in Hz")
    plt.xlim(ufd.f_min, ufd.f_max * 1.1)
    plt.xscale("log")
    plt.ylabel("Period [bins]")
    plt.show()
