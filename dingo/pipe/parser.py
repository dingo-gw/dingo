"""Modified version of bilby_pipe parser."""

import configargparse

from bilby_pipe.bilbyargparser import BilbyArgParser
from bilby_pipe.parser import StoreBoolean
from bilby_pipe.parser import create_parser as create_bilby_pipe_parser
from bilby_pipe.utils import (
    ENVIRONMENT_DEFAULTS,
    logger,
    nonefloat,
    noneint,
    nonestr,
)

logger.name = "dingo_pipe"


def create_parser(top_level=True, usage=None):
    """Creates the BilbyArgParser for dingo_pipe

    Parameters
    ----------
    top_level:
        If true, parser is to be used at the top-level with requirement
        checking etc., else it is an internal call and will be ignored.

    Returns
    -------
    parser: BilbyArgParser instance
        Argument parser

    """
    bilby_pipe_parser = create_bilby_pipe_parser(top_level=top_level, usage=usage)

    bilby_pipe_parser.exclusive_keys.update(
        {
            "Detector arguments": ["--psd-dict", "--asd-dataset"],
            "Injection arguments": ["--injection-file", "--injection-dict"],
            "Prior arguments": ["--prior-file", "--prior-dict"],
        }
    )

    dingo_parser = bilby_pipe_parser.add_argument_group(
        title="DINGO-specific arguments",
    )

    if top_level is False:
        dingo_parser.add(
            "--event-data-file",
            type=nonestr,
            default=None,
            help="Filename for the event: only used internally by sampling and "
            "importance_sampling",
        )
        dingo_parser.add(
            "--proposal-samples-file",
            type=nonestr,
            default=None,
            help="Filename for the proposal samples: only used internally by "
            "importance_sampling",
        )
        dingo_parser.add(
            "--importance-sampling-generation",
            action="store_true",
            help="Whether to prepare data based on the updated importance sampling "
            "settings rather than network settings. This is used internally for "
            "data generation, when preparing different data for the importance "
            "sampling stage.",
        )

    dingo_parser.add(
        "--asd-dataset",
        type=nonestr,
        default=None,
        help="DINGO ASDDataset file to be used for injections. If specified, dingo_pipe "
        "will generate PSD files based on random ASDs in the dataset.",
    )

    dingo_parser.add(
        "--dingo-injection",
        action="store_true",
        help=(
            "If true, use the DINGO Injection class to generate the injection signal. "
            "Otherwise, use bilby_pipe. When using DINGO for injections, the noise is "
            "still generated using bilby_pipe. Defaults to false."
        ),
    )
    dingo_parser.add(
        "--save-bilby-data-dump",
        action=StoreBoolean,
        default=False,
        help=(
            "If given, will also save a data dump consistent with the DINGO injection."
            "This is useful when comparing with Bilby"
        ),
    )

    # Added for Dingo.
    dingo_parser.add(
        "--model",
        type=str,
        required=True,
        help="Neural network model for generating posterior samples.",
    )
    dingo_parser.add(
        "--model-init",
        type=nonestr,
        default=None,
        help="Neural network model for generating samples to initialize Gibbs sampling."
        "Must be provided if the main model is a GNPE model. ",
    )
    dingo_parser.add(
        "--model-reference-time",
        type=nonefloat,
        default=None,
        help="Reference time for neural network. Do not add this manually as it is "
        "specified by the network.",
    )
    dingo_parser.add(
        "--recover-log-prob",
        action=StoreBoolean,
        default=True,
        help="For GNPE models, whether to recover the log probability by training an "
        "unconditional flow for the GNPE proxies.",
    )
    dingo_parser.add(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for sampling. Choices are 'cpu' and 'cuda'. Default 'cuda'.",
    )
    dingo_parser.add(
        "--num-gnpe-iterations",
        type=int,
        default=30,
        help="Number of GNPE iterations to perform when using a GNPE model. Default 30.",
    )
    dingo_parser.add(
        "--num-samples",
        type=int,
        default=50000,
        help="Number of posterior samples desired. Default is 50000",
    )
    dingo_parser.add(
        "--batch-size",
        type=int,
        default=50000,
        help="Number of samples per batch. This is limited by the amount of GPU RAM. "
        "Default is 50000",
    )
    dingo_parser.add(
        "--density-recovery-settings",
        type=str,
        default="ProxyRecoveryDefault",
        help=(
            "Dictionary of density-recovery-settings to pass in, e.g., {num_samples: "
            "400_000, nde_settings: (...)} OR pass pre-defined set of "
            "density-recovery-settings {ProxyRecoveryDefault}"
        ),
    )
    dingo_parser.add(
        "--importance-sample",
        action=StoreBoolean,
        default=True,
        help=("Whether to perform importance sampling on result. (Default: True)"),
    )
    dingo_parser.add(
        "--importance-sampling-settings",
        type=str,
        default="Default",
        help=(
            "Dictionary of importance-sampling-settings to pass in, e.g., "
            "{synthetic_phase: {approximation_22_mode: False, (...)}} OR pass"
            "pre-defined set of density-recovery-settings {PhaseRecoveryDefault}"
        ),
    )
    dingo_parser.add(
        "--importance-sampling-updates",
        type=str,
        default="{}",
        help=(
            "Dictionary of updated settings to be used for importance sampling, "
            "including new prior, data conditioning, and waveform approximant. This "
            "will get populated with any settings provided elsewhere that would "
            "otherwise override model settings. This is useful for tweaking settings "
            "that networks were trained with."
        ),
    )

    return bilby_pipe_parser
