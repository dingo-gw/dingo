import argparse

import numpy as np
import yaml
from scipy.interpolate import interp1d

from dingo.gw.domains import build_domain, MultibandedFrequencyDomain
from dingo.gw.gwutils import get_mismatch
from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.dataset.new_generate import (
    generate_waveforms_parallel,
    new_generate_parameters_and_polarizations,
)
from dingo.gw.waveform_generator.new_api import build_waveform_generator


def _evaluate_multibanding_main(
    settings_file: str,
    num_samples: int,
):
    with open(settings_file, "r") as f:
        settings = yaml.safe_load(f)

    # Ignore any compression settings
    if "compression" in settings:
        del settings["compression"]

    # Update prior to challenge the multi-banding:
    #
    # (a) Set geocent_time = 0.12 s (boundary of usual prior + Earth-radius crossing time)
    # (b) Set chirp mass to bottom end of prior.
    prior = build_prior_with_defaults(settings["intrinsic_prior"])
    settings["intrinsic_prior"]["geocent_time"] = 0.12
    settings["intrinsic_prior"]["chirp_mass"] = prior["chirp_mass"].minimum
    # Rebuild prior with updated settings.
    prior = build_prior_with_defaults(settings["intrinsic_prior"])
    print("Prior")
    for k, v in prior.items():
        print(f"{k}: {v}")

    domain = build_domain(settings["domain"])
    print("\nDomain")
    print(domain.domain_dict)

    if not isinstance(domain, MultibandedFrequencyDomain):
        raise ValueError("Waveform dataset domain not a MultibandedFrequencyDomain.")

    # Strip legacy new_interface flag; new API dispatches by approximant.
    wfg_kwargs = {
        k: v for k, v in settings["waveform_generator"].items() if k != "new_interface"
    }

    waveform_generator_mfd = build_waveform_generator(wfg_kwargs, domain=domain)
    waveform_generator_ufd = build_waveform_generator(
        wfg_kwargs, domain=domain.base_domain
    )

    # Generate MFD waveforms.
    parameters, polarizations_mfd = new_generate_parameters_and_polarizations(
        waveform_generator_mfd, prior, num_samples, 1
    )

    # Generate UFD waveforms, re-using the parameter choices from before.
    polarizations_ufd = generate_waveforms_parallel(
        waveform_generator_ufd, parameters
    )

    # Compare UFD waveforms against MFD waveforms interpolated to MFD.
    mismatches = {}
    ufd = domain.base_domain
    mfd = domain
    for pol_name in ("h_plus", "h_cross"):
        mfd_data = getattr(polarizations_mfd, pol_name)
        ufd_data = getattr(polarizations_ufd, pol_name)
        mismatches[pol_name] = np.empty(len(mfd_data))
        for i in range(len(mfd_data)):
            mfd_interpolated = interp1d(
                mfd(), mfd_data[i], fill_value="extrapolate"
            )(ufd())
            mismatches[pol_name][i] = get_mismatch(
                ufd_data[i],
                mfd_interpolated,
                ufd,
                asd_file="aLIGO_ZERO_DET_high_P_asd.txt",
            )

    print("\nMismatches between UFD waveforms and MFD waveforms interpolated to MFD.")
    print(
        "This is a conservative estimate of the MFD performance when training "
        "networks."
    )
    mismatches = np.concatenate([v for v in mismatches.values()])
    print(f"num_samples = {num_samples}")
    print("  Mean mismatch = {}".format(np.mean(mismatches)))
    print("  Standard deviation = {}".format(np.std(mismatches)))
    print("  Max mismatch = {}".format(np.max(mismatches)))
    print("  Median mismatch = {}".format(np.median(mismatches)))
    print("  Percentiles:")
    print("    99    -> {}".format(np.percentile(mismatches, 99)))
    print("    99.9  -> {}".format(np.percentile(mismatches, 99.9)))
    print("    99.99 -> {}".format(np.percentile(mismatches, 99.99)))


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
