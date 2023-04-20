from pathlib import Path

from bilby_pipe.bilbyargparser import BilbyArgParser
from bilby_pipe.utils import get_command_line_arguments, logger, parse_args

from dingo.gw.result import Result

logger.name = "dingo_pipe"


def create_parser():
    """Generate a parser for the plot script

    Returns
    -------
    parser: BilbyArgParser
        A parser with all the default options already added

    """
    parser = BilbyArgParser(ignore_unknown_config_file_keys=True)
    parser.add("--result", type=str, required=True, help="The result file")
    parser.add("--label", type=str, default="")
    # parser.add("--calibration", action="store_true", help="Generate calibration plot")
    parser.add("--corner", action="store_true", help="Generate corner plot")
    parser.add("--weights", action="store_true", help="Generate plot of importance "
                                                      "weights")
    parser.add("--log_probs", action="store_true", help="Generate plot of target"
                                                        "versus proposal log probability")
    # parser.add("--marginal", action="store_true", help="Generate marginal plots")
    # parser.add("--skymap", action="store_true", help="Generate skymap")
    # parser.add("--waveform", action="store_true", help="Generate waveform")
    parser.add(
        "--outdir", type=str, required=False, help="The directory to save the plots in"
    )
    parser.add(
        "--format",
        type=str,
        default="png",
        help="Format for making bilby_pipe plots, can be [png, pdf, html]. "
        "If specified format is not supported, will default to png.",
    )
    return parser


def main():
    args, _ = parse_args(get_command_line_arguments(), create_parser())

    logger.info(f"Generating plots for results file {args.result}")

    result = Result(file_name=args.result)

    if args.corner:
        logger.info("Generating corner plot.")
        result.plot_corner(filename=f"{args.outdir}/{args.label}_corner.pdf")

    if args.weights:
        logger.info("Generating weights plot.")
        result.plot_weights(filename=f"{args.outdir}/{args.label}_weights.png")

    if args.log_probs:
        logger.info("Generating log probability plot.")
        result.plot_log_probs(filename=f"{args.outdir}/{args.label}_log_probs.png")
