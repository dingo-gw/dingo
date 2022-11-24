#
#  Adapted from bilby_pipe. In particular, uses the bilby_pipe data generation code.
#
import os

import numpy as np
import pandas as pd
from bilby_pipe.input import Input
from bilby_pipe.main import MainInput as BilbyMainInput
from bilby_pipe.utils import (
    parse_args,
    get_command_line_arguments,
    BilbyPipeError,
    logger,
)

from .dag_creator import generate_dag
from .parser import create_parser

from ..domains import build_domain_from_model_metadata
from ...core.models import PosteriorModel


def fill_in_arguments_from_model(args):
    # FIXME: It would be better if we did not have to load an entire model just to
    #  gain access to the metadata. Store a copy of metadata separately?
    logger.info(f"Loading dingo model from {args.model} in order to access settings.")
    model = PosteriorModel(args.model, device="cpu", load_training_info=False)
    model_metadata = model.metadata

    domain = build_domain_from_model_metadata(model_metadata)

    data_settings = model_metadata["train_settings"]["data"]

    model_args = {
        "duration": domain.duration,
        "minimum_frequency": domain.f_min,
        "maximum_frequency": domain.f_max,
        "detectors": data_settings["detectors"],
        "sampling_frequency": data_settings["window"]["f_s"],
        "tukey_roll_off": data_settings["window"]["roll_off"],
    }

    changed_args = {}
    for k, v in model_args.items():
        if vars(args)[k] is not None:

            # Convert type from str to enable comparison.
            args_v = vars(args)[k]
            try:
                if isinstance(v, float):
                    args_v = float(args_v)
                elif isinstance(v, int):
                    args_v = int(args_v)
            except ValueError:
                pass

            if args_v != v:
                logger.warning(
                    f"Argument {k} provided to dingo_pipe as {vars(args)[k]} "
                    f"does not match value {v} in model file. Using {k} = "
                    f"{v} for inference, and will attempt to change this "
                    f"during importance sampling."
                )
                changed_args[k] = vars(args)[k]

        vars(args)[k] = v

    # TODO: Also check consistency between model and init_model settings.

    return changed_args


class MainInput(BilbyMainInput):
    def __init__(self, args, unknown_args):

        # Settings added for dingo.

        self.model = args.model
        self.model_init = args.model_init
        self.num_gnpe_iterations = args.num_gnpe_iterations

        Input.__init__(self, args, unknown_args, print_msg=False)

        self.known_args = args
        self.unknown_args = unknown_args
        self.ini = args.ini
        self.submit = args.submit
        self.condor_job_priority = args.condor_job_priority
        self.create_summary = args.create_summary

        self.outdir = args.outdir
        self.label = args.label
        self.log_directory = args.log_directory
        self.accounting = args.accounting
        self.accounting_user = args.accounting_user
        # self.sampler = args.sampler
        self.detectors = args.detectors
        self.coherence_test = False  # dingo mod: Cannot use different sets of detectors.
        self.n_parallel = 1
        # self.transfer_files = args.transfer_files
        self.osg = args.osg
        self.desired_sites = args.desired_sites
        # self.analysis_executable = args.analysis_executable
        # self.analysis_executable_parser = args.analysis_executable_parser
        # self.result_format = args.result_format
        self.final_result = args.final_result
        self.final_result_nsamples = args.final_result_nsamples

        # self.webdir = args.webdir
        self.email = args.email
        self.notification = args.notification
        self.queue = args.queue
        # self.existing_dir = args.existing_dir

        self.scheduler = args.scheduler
        self.scheduler_args = args.scheduler_args
        self.scheduler_module = args.scheduler_module
        self.scheduler_env = args.scheduler_env
        self.scheduler_analysis_time = args.scheduler_analysis_time

        # self.waveform_approximant = args.waveform_approximant
        #
        # self.time_reference = args.time_reference
        # self.reference_frame = args.reference_frame
        # self.likelihood_type = args.likelihood_type
        self.duration = args.duration
        # self.phase_marginalization = args.phase_marginalization
        # self.prior_file = args.prior_file
        # self.prior_dict = args.prior_dict
        # self.default_prior = args.default_prior
        self.minimum_frequency = args.minimum_frequency
        # self.enforce_signal_duration = args.enforce_signal_duration

        self.run_local = args.local
        self.local_generation = args.local_generation
        self.local_plot = args.local_plot

        self.post_trigger_duration = args.post_trigger_duration

        # self.ignore_gwpy_data_quality_check = args.ignore_gwpy_data_quality_check
        self.trigger_time = args.trigger_time
        # self.deltaT = args.deltaT
        # self.gps_tuple = args.gps_tuple
        # self.gps_file = args.gps_file
        self.timeslide_file = args.timeslide_file
        # self.gaussian_noise = args.gaussian_noise
        # self.zero_noise = args.zero_noise
        # self.n_simulation = args.n_simulation
        #
        # self.injection = args.injection
        # self.injection_numbers = args.injection_numbers
        self.injection_file = args.injection_file
        # self.injection_dict = args.injection_dict
        # self.injection_waveform_arguments = args.injection_waveform_arguments
        # self.injection_waveform_approximant = args.injection_waveform_approximant
        # self.generation_seed = args.generation_seed
        # if self.injection:
        #     self.check_injection()

        self.request_disk = args.request_disk
        self.request_memory = args.request_memory
        self.request_memory_generation = args.request_memory_generation
        self.request_cpus = args.request_cpus
        self.request_cpus_importance_sampling = args.request_cpus_importance_sampling
        # self.sampler_kwargs = args.sampler_kwargs
        # self.mpi_samplers = ["pymultinest"]
        # self.use_mpi = (self.sampler in self.mpi_samplers) and (self.request_cpus > 1)
        #
        # # Set plotting options when need the plot node
        # self.plot_node_needed = False
        # for plot_attr in [
        #     "calibration",
        #     "corner",
        #     "marginal",
        #     "skymap",
        #     "waveform",
        # ]:
        #     attr = f"plot_{plot_attr}"
        #     setattr(self, attr, getattr(args, attr))
        #     if getattr(self, attr):
        #         self.plot_node_needed = True
        #
        # # Set all other plotting options
        # for plot_attr in [
        #     "trace",
        #     "data",
        #     "injection",
        #     "spectrogram",
        #     "format",
        # ]:
        #     attr = f"plot_{plot_attr}"
        #     setattr(self, attr, getattr(args, attr))
        #
        # self.postprocessing_executable = args.postprocessing_executable
        # self.postprocessing_arguments = args.postprocessing_arguments
        # self.single_postprocessing_executable = args.single_postprocessing_executable
        # self.single_postprocessing_arguments = args.single_postprocessing_arguments
        #
        # self.summarypages_arguments = args.summarypages_arguments
        #
        self.psd_dict = args.psd_dict

        # self.check_source_model(args)

        self.extra_lines = []
        self.requirements = []

    @property
    def request_cpus_importance_sampling(self):
        return self._request_cpus_importance_sampling

    @request_cpus_importance_sampling.setter
    def request_cpus_importance_sampling(self, request_cpus_importance_sampling):
        logger.info(f"Setting analysis request_cpus_importance_sampling = "
                    f"{request_cpus_importance_sampling}")
        self._request_cpus_importance_sampling = request_cpus_importance_sampling


def write_complete_config_file(parser, args, inputs, input_cls=MainInput):
    args_dict = vars(args).copy()
    for key, val in args_dict.items():
        if key == "label":
            continue
        if isinstance(val, str):
            if os.path.isfile(val) or os.path.isdir(val):
                setattr(args, key, os.path.abspath(val))
        if isinstance(val, list):
            if isinstance(val[0], str):
                setattr(args, key, f"[{', '.join(val)}]")
    # args.sampler_kwargs = str(inputs.sampler_kwargs)
    args.submit = False
    parser.write_to_file(
        filename=inputs.complete_ini_file,
        args=args,
        overwrite=False,
        include_description=False,
    )

    # Verify that the written complete config is identical to the source config
    # complete_args = parser.parse([inputs.complete_ini_file])
    # complete_inputs = input_cls(complete_args, "")
    # ignore_keys = ["scheduler_module", "submit"]
    # differences = []
    # for key, val in inputs.__dict__.items():
    #     if key in ignore_keys:
    #         continue
    #     if key not in complete_args:
    #         continue
    #     if isinstance(val, pd.DataFrame) and all(val == complete_inputs.__dict__[key]):
    #         continue
    #     if isinstance(val, np.ndarray) and all(
    #         np.array(val) == np.array(complete_inputs.__dict__[key])
    #     ):
    #         continue
    #     if isinstance(val, str) and os.path.isfile(val):
    #         # Check if it is relpath vs abspath
    #         if os.path.abspath(val) == os.path.abspath(complete_inputs.__dict__[key]):
    #             continue
    #     if val == complete_inputs.__dict__[key]:
    #         continue
    #
    #     differences.append(key)
    #
    # if len(differences) > 0:
    #     for key in differences:
    #         print(key, f"{inputs.__dict__[key]} -> {complete_inputs.__dict__[key]}")
    # raise BilbyPipeError(
    #     "The written config file {} differs from the source {} in {}".format(
    #         inputs.ini, inputs.complete_ini_file, differences
    #     )
    # )


def main():
    parser = create_parser(top_level=True)
    args, unknown_args = parse_args(get_command_line_arguments(), parser)

    changed_args = fill_in_arguments_from_model(args)
    inputs = MainInput(args, unknown_args)
    write_complete_config_file(parser, args, inputs)

    # TODO: Use two sets of inputs! The first must match the network; the second is
    #  used in importance sampling.

    generate_dag(inputs)

    if len(unknown_args) > 0:
        print(f"Unrecognized arguments {unknown_args}")
