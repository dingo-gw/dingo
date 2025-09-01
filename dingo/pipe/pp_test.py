import argparse
from pathlib import Path

from bilby_pipe.utils import logger

from dingo.core.result import make_pp_plot
from dingo.gw.result import Result

logger.name = "dingo_pipe"


def create_parser():
    parser = argparse.ArgumentParser(
        prog="dingo_pipe PP test",
        usage="Generates a pp plot from a directory containing a set of results",
    )
    parser.add_argument("directory", help="Path to the result files")
    parser.add_argument(
        "--outdir", help="Path to output directory, defaults to input directory "
    )
    parser.add_argument("--label", help="Additional label to use for output")
    parser.add_argument(
        "--print", action="store_true", help="Print the list of filenames used"
    )
    parser.add_argument(
        "-n", type=int, help="Number of samples to truncate to", default=None
    )
    parser.add_argument(
        "--filter",
        type=str,
        help="A string to match and filtering results. If not specified, will look "
        "first for '*importance_sampling.hdf5', then '*sampling.hdf5'.",
        default=None,
    )
    return parser


def get_results_filenames(args):
    p = Path(args.directory)
    if args.filter:
        results_files = list(p.glob(f"*{args.filter}*.hdf5"))
    else:
        results_files = list(p.glob("*importance_sampling.hdf5"))
        if len(results_files) == 0:
            results_files = list(p.glob("*sampling.hdf5"))
    if len(results_files) == 0:
        raise FileNotFoundError(f"No results found in path {args.directory}")

    if args.n is not None:
        logger.info(f"Truncating to first {args.n} results")
        results_files = results_files[: args.n]
    return results_files


def get_basename(args):
    if args.outdir is None:
        args.outdir = args.directory
    basename = f"{args.outdir}/"
    if args.label is not None:
        basename += f"{args.label}_"
    return basename


def main(args=None):
    if args is None:
        args = create_parser().parse_args()
    results_filenames = get_results_filenames(args)
    results = [Result(file_name=f) for f in results_filenames]
    basename = get_basename(args)

    logger.info("Generating PP plot.")
    # keys = [
    #     name
    #     for name, p in results[0].prior.items()
    #     if isinstance(p, str) or p.is_fixed is False
    # ]
    logger.info(f"Parameters = {results[0].search_parameter_keys}")
    make_pp_plot(results, filename=f"{basename}pp.pdf")
    if "weights" in results[0].samples:
        logger.info("Generating IS PP plot.")
        make_pp_plot(results, filename=f"{basename}pp_IS.pdf", weighted=True)
