import argparse

import numpy as np
import yaml
from scipy.interpolate import interp1d

from dingo.gw.dataset import generate_parameters_and_polarizations
from dingo.gw.dataset._multibanded_domain_utils import (build_extreme_prior,
                                                        print_mismatch_stats)
from dingo.gw.domains import MultibandedFrequencyDomain, build_domain
from dingo.gw.gwutils import get_mismatch
from dingo.gw.waveform_generator import (NewInterfaceWaveformGenerator,
                                         WaveformGenerator,
                                         generate_waveforms_parallel)


def _evaluate_multibanding_main(
    settings_file: str,
    num_samples: int,
):
    with open(settings_file, "r") as f:
        settings = yaml.safe_load(f)

    # Ignore any compression settings
    if "compression" in settings:
        del settings["compression"]

    prior = build_extreme_prior(settings)
    print("Prior")
    for k, v in prior.items():
        print(f"{k}: {v}")

    domain = build_domain(settings["domain"])
    print("\nDomain")
    print(domain.domain_dict)

    if not isinstance(domain, MultibandedFrequencyDomain):
        raise ValueError("Waveform dataset domain not a MultibandedFrequencyDomain.")

    if settings["waveform_generator"].get("new_interface", False):
        waveform_generator_mfd = NewInterfaceWaveformGenerator(
            domain=domain,
            **settings["waveform_generator"],
        )
        waveform_generator_ufd = NewInterfaceWaveformGenerator(
            domain=domain.base_domain,
            **settings["waveform_generator"],
        )
    else:
        waveform_generator_mfd = WaveformGenerator(
            domain=domain,
            **settings["waveform_generator"],
        )
        waveform_generator_ufd = WaveformGenerator(
            domain=domain.base_domain,
            **settings["waveform_generator"],
        )

    # Generate MFD waveforms.
    parameters, polarizations_mfd = generate_parameters_and_polarizations(
        waveform_generator_mfd, prior, num_samples, 1
    )

    # Generate UFD waveforms, re-using the parameter choices from before.
    polarizations_ufd = generate_waveforms_parallel(waveform_generator_ufd, parameters)

    # Compare UFD waveforms against MFD waveforms interpolated to MFD.
    mismatches = {}
    ufd = domain.base_domain
    mfd = domain
    for pol, d in polarizations_mfd.items():
        mismatches[pol] = np.empty(len(d))
        for i in range(len(d)):
            mfd_interpolated = interp1d(mfd(), d[i], fill_value="extrapolate")(ufd())
            mismatches[pol][i] = get_mismatch(
                polarizations_ufd[pol][i],
                mfd_interpolated,
                ufd,
                asd_file="aLIGO_ZERO_DET_high_P_asd.txt",
            )

    mismatches = np.concatenate(list(mismatches.values()))
    print_mismatch_stats(mismatches, num_samples)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Evaluate performance of multibanding on waveform dataset.",
    )
    parser.add_argument(
        "--settings-file",
        type=str,
        required=True,
        help="YAML file containing database settings",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5000,
        help="Number of waveform evaluations for comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _evaluate_multibanding_main(args.settings_file, args.num_samples)
