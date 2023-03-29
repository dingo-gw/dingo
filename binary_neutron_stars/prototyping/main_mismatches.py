import numpy as np

np.random.seed(1)
import yaml
import matplotlib.pyplot as plt
from torchvision.transforms import Compose
import pandas as pd
from multiprocessing import Pool
from threadpoolctl import threadpool_limits

from dingo.gw.domains import build_domain
from dingo.gw.gwutils import get_mismatch
from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.waveform_generator import WaveformGenerator, generate_waveforms_parallel
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
from heterodyning import change_heterodyning


def print_mismatches(mismatches):
    print(f"n = {len(mismatches)}")
    print(f"  Mean mismatch = {np.mean(mismatches)}")
    print(f"  Standard deviation = {np.std(mismatches)}")
    print(f"  Max mismatch = {np.max(mismatches)}")
    print(f"  Median mismatch = {np.median(mismatches)}")
    print(f"  Percentiles:")
    print(f"    99    -> {np.percentile(mismatches, 99)}")
    print(f"    99.9  -> {np.percentile(mismatches, 99.9)}")
    print(f"    99.99 -> {np.percentile(mismatches, 99.99)}")


if __name__ == "__main__":
    num_processes = 10
    with open("waveform_dataset_settings.yaml", "r") as f:
        settings = yaml.safe_load(f)
    ufd = build_domain(settings["domain"])
    prior = build_prior_with_defaults(settings["intrinsic_prior"])

    # Build WaveformGenerators:
    #   wfg_het:        in uniform frequency domain
    #   wfg_het_mfd:    in multibanded frequency domain
    wfg_het = WaveformGenerator(
        domain=ufd, transform=ApplyHeterodyning(ufd), **settings["waveform_generator"]
    )
    _, pols_het = generate_parameters_and_polarizations(
        wfg_het, prior, 100, num_processes
    )
    bands = get_decimation_bands_adaptive(
        ufd,
        np.concatenate(list(pols_het.values())),
        min_num_bins_per_period=16,
        delta_f_max=2.0,
    )
    mfd = MultibandedFrequencyDomain(bands, ufd.domain_dict)
    print(len(mfd), bands)
    wfg_het_mfd = WaveformGenerator(
        domain=mfd, transform=ApplyHeterodyning(mfd), **settings["waveform_generator"]
    )

    # generate polarizations
    parameters = pd.DataFrame(prior.sample(1000))
    with threadpool_limits(limits=1, user_api="blas"):
        with Pool(processes=num_processes) as pool:
            pols_het = generate_waveforms_parallel(wfg_het, parameters, pool)
            pols_het_mfd = generate_waveforms_parallel(wfg_het_mfd, parameters, pool)
    pols_het_dec = {pol_name: mfd.decimate(pol) for pol_name, pol in pols_het.items()}

    ##############
    ### Checks ###
    ##############

    # 1) Check mismatch between waveforms generated at mfd sample frequencies and
    # waveforms generated at full ufd resolution, which are then decimated.
    mismatches = [
        get_mismatch(
            pols_het_mfd[pol_name],
            pols_het_dec[pol_name],
            mfd,
            "aLIGO_ZERO_DET_high_P_asd.txt",
        )
        for pol_name in pols_het_mfd.keys()
    ]
    print_mismatches(np.array(mismatches).flatten())

    # 2) Check mismatches of decompressed training waveforms. We have multiple
    # approximations here, due to decimation.
    #
    # Desired order of transforms:
    #   - wf in ufd
    #   - heterodyne with chirp_mass_proxy
    #   - apply time shift
    #   - decimate to mfd
    #
    # Order of transforms in practice:
    #   - wf in mfd, heterodyned with chirp_mass
    #   - change heterodyning from chirp_mass to chirp_mass_proxy
    #   - apply time shift
    #
    # We have to apply these transforms in this order (starting with mfd waveforms
    # already, as opposed to applying transforms and decimating to mfd only in the end)
    # as we can't store full ufd waveforms due to space and compute constraints.
    # However, changing the order will result in (hopefully small) mismatches,
    # as not all ufd bins that are averaged to a single mfd bin will be transformed in
    # the same way; instead, the applied phases will be slightly different.

    delta_chirp_mass_max = 0.002
    delta_time_max = 0.003

    # change heterodyning chirp mass
    kwargs_het_old = {"chirp_mass": np.array(parameters["chirp_mass"])}
    kwargs_het_new = {
        "chirp_mass": np.array(parameters["chirp_mass"])
        + np.random.choice([-1, 1], size=len(parameters)) * delta_chirp_mass_max
    }
    # reference
    print("a")
    pols_hetp = {
        pol_name: change_heterodyning(pol, ufd, kwargs_het_old, kwargs_het_new)
        for pol_name, pol in pols_het.items()
    }
    print("b")
    pols_hetp_dec = {pol_name: mfd.decimate(pol) for pol_name, pol in pols_hetp.items()}
    # ours
    print("c")
    pols_hetp_mfd = {
        pol_name: change_heterodyning(pol, mfd, kwargs_het_old, kwargs_het_new)
        for pol_name, pol in pols_het_mfd.items()
    }
    # mismatches
    mismatches = [
        get_mismatch(
            pols_hetp_mfd[pol_name],
            pols_hetp_dec[pol_name],
            mfd,
            "aLIGO_ZERO_DET_high_P_asd.txt",
        )
        for pol_name in pols_het_mfd.keys()
    ]
    print_mismatches(np.array(mismatches).flatten())


    # For time shifts, we can easily check manually. The phase differences within a
    # decimation segment (i.e., ufd bins that are averaged to a single mfd bin) is
    # upper-bounded by
    #               phase_max = 2 * pi * delta_t_max * delta_f_max.
    # We want phase_max << 1. For delta_t_max * delta_f_max = 0.001 s * 2.0 1/s = 0.002,
    # deviations due to time shifts should be very small.

    print("cp")
