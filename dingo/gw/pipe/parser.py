#
#  Modified version of bilby_pipe parser.
#

import argparse

import configargparse
from bilby_pipe.bilbyargparser import BilbyArgParser
from bilby_pipe.utils import get_version_information, logger, nonefloat, noneint, nonestr


class StoreBoolean(argparse.Action):
    """argparse class for robust handling of booleans with configargparse

    When using configargparse, if the argument is setup with
    action="store_true", but the default is set to True, then there is no way,
    in the config file to switch the parameter off. To resolve this, this class
    handles the boolean properly.

    """

    def __call__(self, parser, namespace, value, option_string=None):
        value = str(value).lower()
        if value in ["true"]:
            setattr(namespace, self.dest, True)
        else:
            setattr(namespace, self.dest, False)


def create_parser(top_level=True):
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
    parser = BilbyArgParser(
        usage="Perform inference with dingo based on a .ini file.",
        ignore_unknown_config_file_keys=False,
        allow_abbrev=False,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add("ini", type=str, is_config_file=True, help="Configuration ini file")
    parser.add("-v", "--verbose", action="store_true", help="Verbose output")
    # parser.add(
    #     "--version",
    #     action="version",
    #     version=f"%(prog)s={get_version_information()}\nbilby={bilby.__version__}",
    # )

    calibration_parser = parser.add_argument_group(
        "Calibration arguments",
        description="Which calibration model and settings to use.",
    )

    data_gen_pars = parser.add_argument_group(
        "Data generation arguments",
        description="How to generate the data, e.g., from a list of gps times or "
                    "simulated Gaussian noise.",
    )
    data_gen_pars.add(
        "--trigger-time",
        default=None,
        type=nonestr,
        help=(
            "Either a GPS trigger time, or the event name (e.g. GW150914). "
            "For event names, the gwosc package is used to identify the "
            "trigger time"
        ),
    )
    data_gen_pars.add(
        "--channel-dict",
        type=nonestr,
        default=None,
        help=(
            "Channel dictionary: keys relate to the detector with values "
            "the channel name, e.g. 'GDS-CALIB_STRAIN'. For GWOSC open data, "
            "set the channel-dict keys to 'GWOSC'. Note, the "
            "dictionary should follow basic python dict syntax."
        ),
    )

    submission_parser = parser.add_argument_group(
        title="Job submission arguments",
        description="How the jobs should be formatted, e.g., which job scheduler to use.",
    )
    submission_parser.add("--label", type=str, default="label", help="Output label")
    submission_parser.add(
        "--local",
        action="store_true",
        help="Run the job locally, i.e., not through a batch submission",
    )
    submission_parser.add(
        "--local-generation",
        action="store_true",
        help=(
            "Run the data generation job locally. This may be useful for "
            "running on a cluster where the compute nodes do not have "
            "internet access. For HTCondor, this is done using the local "
            "universe, for slurm, the jobs will be run at run-time"
        ),
    )
    submission_parser.add(
        "--local-plot", action="store_true", help="Run the plot job locally"
    )
    submission_parser.add(
        "--accounting",
        type=nonestr,
        help="Accounting group to use (see, https://accounting.ligo.org/user)",
    )
    submission_parser.add(
        "--accounting-user",
        type=nonestr,
        help="Accounting group user to use (see, https://accounting.ligo.org/user)",
    )
    submission_parser.add(
        "--outdir",
        type=str,
        default="outdir",
        help="The output directory. If outdir already exists, an auto-incrementing "
             "naming scheme is used",
    )
    submission_parser.add(
        "--request-disk",
        type=float,
        default=5,
        help="Disk allocation request in GB. Default is 5GB.",
    )
    submission_parser.add(
        "--request-memory-generation",
        type=nonefloat,
        default=None,
        help="Memory allocation request (GB) for data generation step",
    )
    submission_parser.add(
        "--scheduler",
        type=str,
        default="condor",
        help="Format submission script for specified scheduler.",
    )
    submission_parser.add(
        "--scheduler-args",
        type=nonestr,
        default=None,
        help=(
            "Space-separated #SBATCH command line args to pass to slurm. "
            "The args needed will depend on the setup of your slurm scheduler."
            "Please consult documentation for your local cluster (slurm only)."
        ),
    )
    submission_parser.add(
        "--scheduler-module",
        type=nonestr,
        action="append",
        default=None,
        help="Space-separated list of modules to load at runtime (slurm only)",
    )
    submission_parser.add(
        "--scheduler-env",
        type=nonestr,
        default=None,
        help="Python environment to activate (slurm only)",
    )
    submission_parser.add(
        "--scheduler-analysis-time", type=nonestr, default="7-00:00:00", help=""
    )
    submission_parser.add(
        "--condor-job-priority",
        type=int,
        default=0,
        help=(
            "Job priorities allow a user to sort their HTCondor jobs to determine "
            "which are tried to be run first. "
            "A job priority can be any integer: larger values denote better priority. "
            "By default HTCondor job priority=0. "
        ),
    )
    submission_parser.add(
        "--log-directory",
        type=nonestr,
        default=None,
        help="If given, an alternative path for the log output",
    )
    submission_parser.add(
        "--osg",
        action="store_true",
        default=False,
        help="If true, format condor submission for running on OSG, default is False",
    )

    sampler_parser = parser.add_argument_group(title="Sampler arguments")
    sampler_parser.add(
        "--model",
        type=str,
        # required=True,
        help="Neural network model for generating posterior samples."
    )
    sampler_parser.add(
        "--init-model",
        type=nonestr,
        default=None,
        help="Neural network model for generating samples to initialize Gibbs sampling."
             "Must be provided if the main model is a GNPE model. "
    )
    sampler_parser.add(
        "--num-samples",
        type=int,
        # required=True,
        help="Number of desired samples."
    )
    sampler_parser.add(
        "--num-gnpe-iterations",
        type=int,
        default=30,
        help="Number of GNPE iterations to perform when using a GNPE model. Default 30."
    )

    output_parser = parser.add_argument_group(
        title="Output arguments", description="What kind of output/summary to generate."
    )
    output_parser.add("--email", type=nonestr, help="Email for notifications")
    output_parser.add(
        "--notification",
        type=nonestr,
        default="Never",
        help=(
            "Notification setting for HTCondor jobs. "
            "One of 'Always','Complete','Error','Never'. "
            "If defined by 'Always', "
            "the owner will be notified whenever the job "
            "produces a checkpoint, as well as when the job completes. "
            "If defined by 'Complete', "
            "the owner will be notified when the job terminates. "
            "If defined by 'Error', "
            "the owner will only be notified if the job terminates abnormally, "
            "or if the job is placed on hold because of a failure, "
            "and not by user request. "
            "If defined by 'Never' (the default), "
            "the owner will not receive e-mail, regardless to what happens to the job. "
            "Note, an `email` arg is also required for notifications to be emailed. "
        ),
    )
    output_parser.add(
        "--queue",
        type=nonestr,
        help="Condor job queue. Use Online_PE for online parameter estimation runs.",
    )

    return parser
