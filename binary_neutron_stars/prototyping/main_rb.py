import numpy as np

np.random.seed(1)
import yaml
import matplotlib.pyplot as plt
from torchvision.transforms import Compose

from dingo.gw.domains import build_domain
from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.waveform_generator import WaveformGenerator
from dingo.gw.dataset.generate_dataset import (
    WaveformDataset,
    generate_parameters_and_polarizations,
    train_svd_basis,
)
from dingo.gw.SVD import ApplySVD

from dingo.gw.domains.multibanded_frequency_domain import (
    MultibandedFrequencyDomain,
    get_periods,
    get_decimation_bands_adaptive,
)
from bns_transforms import ApplyHeterodyning, ApplyDecimation

if __name__ == "__main__":
    num_processes = 10

    with open("waveform_dataset_settings.yaml", "r") as f:
        settings = yaml.safe_load(f)
    ufd = build_domain(settings["domain"])
    prior = build_prior_with_defaults(settings["intrinsic_prior"])
    waveform_generator = WaveformGenerator(domain=ufd, **settings["waveform_generator"])
    waveform_generator_het = WaveformGenerator(
        domain=ufd, transform=ApplyHeterodyning(ufd), **settings["waveform_generator"]
    )

    # generate polarizations
    parameters_het, polarizations_het = generate_parameters_and_polarizations(
        waveform_generator_het, prior, 100, num_processes
    )

    bands = get_decimation_bands_adaptive(
        ufd,
        np.concatenate(list(polarizations_het.values())),
        min_num_bins_per_period=16,
        delta_f_max=3.0,
    )
    mfd = MultibandedFrequencyDomain(bands, ufd.domain_dict)
    print(len(mfd))
    print(bands)

    # Compute mismatches: multiband->heterodyne vs. heterodyne->decimate

    hp_het = polarizations_het["h_plus"]
    fig = plt.figure()
    fig.set_size_inches((8, 8))
    x = ufd()
    # plt.plot(
    #     x, np.min(get_periods(hp_het.real, upper_bound_for_monotonicity=False), axis=0)
    # )
    # plt.plot(
    #     x, np.min(get_periods(hp_het.real, upper_bound_for_monotonicity=True), axis=0)
    # )
    periods = get_periods(hp_het.real, upper_bound_for_monotonicity=False)
    for perc in [1, 3, 5, 10]:
        plt.plot(np.percentile(periods, perc, axis=0))
    plt.yscale("log")
    plt.ylabel("f in Hz")
    plt.xlim(ufd.f_min, ufd.f_max * 1.1)
    plt.xscale("log")
    plt.ylabel("Period [bins]")
    plt.show()

    transforms = Compose([ApplyHeterodyning(ufd), ApplyDecimation(mfd)])
    waveform_generator_het_dec = WaveformGenerator(
        domain=ufd, transform=transforms, **settings["waveform_generator"]
    )

    num_samples = (
        settings["compression"]["svd"]["num_training_samples"]
        + settings["compression"]["svd"]["num_validation_samples"]
    )
    parameters_het_dec, polarizations_het_dec = generate_parameters_and_polarizations(
        waveform_generator_het_dec, prior, num_samples, num_processes
    )

    svd_dataset = WaveformDataset(
        dictionary={
            "parameters": parameters_het_dec,
            "polarizations": polarizations_het_dec,
            "settings": None,
        }
    )
    svd_dataset.domain = mfd
    basis, n_train, n_test = train_svd_basis(
        svd_dataset,
        settings["compression"]["svd"]["size"],
        settings["compression"]["svd"]["num_training_samples"],
    )

    transforms = Compose(
        [ApplyHeterodyning(ufd), ApplyDecimation(mfd), ApplySVD(basis)]
    )
    waveform_generator_het_dec_svd = WaveformGenerator(
        domain=ufd, transform=transforms, **settings["waveform_generator"]
    )
    params_het_dec_svd, pols_het_dec_svd = generate_parameters_and_polarizations(
        waveform_generator_het_dec_svd, prior, 100, num_processes
    )
    print(polarizations_het["h_plus"].shape)
    print(polarizations_het_dec["h_plus"].shape)
    print(pols_het_dec_svd["h_plus"].shape)

    # test tidal deformability
    parameters = prior.sample()
    pols_0 = waveform_generator_het_dec.generate_hplus_hcross(
        {k: v for k, v in parameters.items() if k not in ["lambda_1", "lambda_2"]}
    )
    pols_1 = waveform_generator_het_dec.generate_hplus_hcross(parameters)
    for pol_name in pols_0.keys():
        plt.title(pol_name)
        plt.plot(pols_0[pol_name], label="lambda=0")
        plt.plot(
            pols_1[pol_name],
            label=f"lambda_1 = {parameters['lambda_1']:.1f}, lambda_2 ="
            f" {parameters['lambda_2']:.1f}",
        )
        plt.legend()
        plt.show()


