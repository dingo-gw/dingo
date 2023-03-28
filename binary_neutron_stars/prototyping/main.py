import numpy as np

np.random.seed(1)
import matplotlib.pyplot as plt
from dingo.gw.domains import build_domain
from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.waveform_generator import WaveformGenerator
from dingo.gw.dataset.generate_dataset import generate_parameters_and_polarizations

from dingo.gw.domains.multibanded_frequency_domain import MultibandedFrequencyDomain
from heterodyning import (
    heterodyne_LO,
    factor_fiducial_waveform,
    change_heterodyning,
)

if __name__ == "__main__":
    # build domain
    uniform_frequency_domain_settings = {
        "type": "FrequencyDomain",
        "f_min": 20,
        "f_max": 1024,
        "delta_f": 1 / 256,
        "window_factor": 1.0,
    }
    original_domain = build_domain(uniform_frequency_domain_settings)

    # build prior
    intrinsic_prior_settings = {
        "mass_1": "bilby.core.prior.Constraint(minimum=1.0, maximum=2.5)",
        "mass_2": "bilby.core.prior.Constraint(minimum=1.0, maximum=2.5)",
        "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass(minimum=1.0, maximum=2.0)",
        "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.125, maximum=1.0)",
        "phase": "default",
        "a_1": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99)",
        "a_2": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99)",
        "tilt_1": "default",
        "tilt_2": "default",
        "phi_12": "default",
        "phi_jl": "default",
        "theta_jn": "default",
        "luminosity_distance": "100.0",
        "geocent_time": "0.0",
    }
    prior = build_prior_with_defaults(intrinsic_prior_settings)

    # build waveform generator
    wfg_settings = {"approximant": "IMRPhenomPv2_NRTidal", "f_ref": 10}
    num_samples = 10
    num_processes = 0

    waveform_generator = WaveformGenerator(domain=original_domain, **wfg_settings)

    parameters, polarizations = generate_parameters_and_polarizations(
        waveform_generator, prior, num_samples, num_processes
    )
    chirp_mass = np.array(parameters["chirp_mass"])
    mass_ratio = np.array(parameters["mass_ratio"])
    chirp_mass_pert = chirp_mass + np.random.normal(0, 0e-3, size=chirp_mass.shape)

    mfd = MultibandedFrequencyDomain.init_for_decimation(
        original_domain, chirp_mass_min=1.0, alpha_bands=4, delta_f_max=8.0
    )
    mfd.initialize_decimation()
    hp = polarizations["h_plus"]
    hp_dec = mfd.decimate(hp)
    hp_dec_het = factor_fiducial_waveform(hp_dec, mfd, chirp_mass)

    hp_het = factor_fiducial_waveform(hp, original_domain, chirp_mass)

    hp_hetp = factor_fiducial_waveform(hp, original_domain, chirp_mass_pert)
    hp_hetp_dec = mfd.decimate(hp_hetp)

    hp_het_dec_hetp = change_heterodyning(
        hp_het_dec,
        mfd,
        new_kwargs={"chirp_mass": chirp_mass_pert},
        old_kwargs={"chirp_mass": chirp_mass},
    )

    hp_het_dec_heti = factor_fiducial_waveform(
        hp_het_dec, mfd, chirp_mass, inverse=True
    )
    hp_het_dec_heti_hetp = factor_fiducial_waveform(
        hp_het_dec_heti, mfd, chirp_mass_pert
    )

    for idx in (0, 1, 2):
        plt.title(f"{idx}, {chirp_mass[idx]}")
        plt.plot(mfd(), hp_hetp_dec[idx])
        # plt.plot(mfd(), hp_het_dec_heti_hetp[idx])
        # plt.plot(mfd(), hp_het_dec[idx])
        plt.plot(mfd(), hp_het_dec_hetp[idx])
        plt.plot(
            mfd(),
            (hp_hetp_dec[idx] - hp_het_dec_hetp[idx]) * 10,
            label="deviation, magnified with 10x",
        )
        plt.xlim((20, 30))
        plt.show()

    hp_het_0 = heterodyne_LO(hp[0], original_domain, parameters["chirp_mass"][0])
    # hp_het_mb = mfd.decimate_to_domain(hp_het)
    #
    # hp_mb = mfd.decimate_to_domain()

    # for idx in (0,1,2):
    #     plt.title(f"{idx}, {chirp_mass[idx]}")
    #     plt.plot(original_domain(), hp[idx])
    #     plt.plot(original_domain(), hp_het[idx])
    #     plt.xlim((20, 30))
    #     plt.show()

    for idx in (0, 1, 2):
        plt.title(f"{idx}, {chirp_mass[idx]}")
        plt.plot(mfd(), hp_dec_het[idx])
        plt.plot(mfd(), hp_het_dec[idx])
        plt.xlim((20, 30))
        plt.show()

    mfd.decimate(hp)

    assert np.all(mfd.decimate(original_domain()) == mfd.sample_frequencies)

    print(mfd._bands)
