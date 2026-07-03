import logging
from typing import Dict

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.interpolate import interp1d

from dingo.gw.dataset import generate_parameters_and_polarizations
from dingo.gw.domains import build_domain, MultibandedFrequencyDomain
from dingo.gw.gwutils import get_mismatch
from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.waveform_generator import (
    NewInterfaceWaveformGenerator,
    WaveformGenerator,
    generate_waveforms_parallel,
)

log = logging.getLogger(__name__)
logging.captureWarnings(True)


def _settings_from_config(cfg: DictConfig) -> Dict:
    settings = OmegaConf.to_container(cfg, resolve=True)
    settings.pop("num_samples", None)
    return settings


def evaluate_multibanding(
    settings: Dict,
    num_samples: int,
):
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
    log.info("Prior")
    for k, v in prior.items():
        log.info(f"{k}: {v}")

    domain = build_domain(settings["domain"])
    log.info("\nDomain")
    log.info(domain.domain_dict)

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

    log.info(
        "\nMismatches between UFD waveforms and MFD waveforms interpolated to MFD."
    )
    log.info(
        "This is a conservative estimate of the MFD performance when training "
        "networks."
    )
    mismatches = np.concatenate([v for v in mismatches.values()])
    log.info(f"num_samples = {num_samples}")
    log.info("  Mean mismatch = {}".format(np.mean(mismatches)))
    log.info("  Standard deviation = {}".format(np.std(mismatches)))
    log.info("  Max mismatch = {}".format(np.max(mismatches)))
    log.info("  Median mismatch = {}".format(np.median(mismatches)))
    log.info("  Percentiles:")
    log.info("    99    -> {}".format(np.percentile(mismatches, 99)))
    log.info("    99.9  -> {}".format(np.percentile(mismatches, 99.9)))
    log.info("    99.99 -> {}".format(np.percentile(mismatches, 99.99)))


@hydra.main(
    version_base="1.3",
    config_path="../../../configs",
    config_name="evaluate_multibanded_domain",
)
def main(cfg: DictConfig) -> None:
    settings = _settings_from_config(cfg)
    evaluate_multibanding(settings, cfg.num_samples)


if __name__ == "__main__":
    main()
