#
#  Adapted from bilby_pipe. In particular, uses the bilby_pipe data generation code.
#
import os

import numpy as np
import pandas as pd
from bilby_pipe.input import Input
from bilby_pipe.main import MainInput as BilbyMainInput
from bilby_pipe.utils import parse_args, get_command_line_arguments, BilbyPipeError

from .dag_creator import generate_dag
from .parser import create_parser

from ..domains import build_domain_from_model_metadata
from ...core.models import PosteriorModel


class MainInput(BilbyMainInput):

    def __init__(self, args, unknown_args):
        Input.__init__(self, args, unknown_args, print_msg=False)

        self.known_args = args
        self.unknown_args = unknown_args
        self.ini = args.ini
        self.condor_job_priority = args.condor_job_priority

        self.outdir = args.outdir
        self.label = args.label
        self.log_directory = args.log_directory
        self.accounting = args.accounting
        self.accounting_user = args.accounting_user
        self.osg = args.osg

        self.email = args.email
        self.notification = args.notification
        self.queue = args.queue

        self.scheduler = args.scheduler
        self.scheduler_args = args.scheduler_args
        self.scheduler_module = args.scheduler_module
        self.scheduler_env = args.scheduler_env
        self.scheduler_analysis_time = args.scheduler_analysis_time

        self.trigger_time = args.trigger_time
        self.timeslide_file = None
        self.channel_dict = args.channel_dict

        self.model = args.model
        self.init_model = args.init_model
        self.num_samples = args.num_samples
        self.num_gnpe_iterations = args.num_gnpe_iterations
        self.n_parallel = 1

        self.run_local = args.local
        self.local_generation = args.local_generation
        self.local_plot = args.local_plot

        self.injection_file = None

        self.request_disk = args.request_disk
        self.request_memory_generation = args.request_memory_generation

        self.extra_lines = []
        self.requirements = []

        self.fill_in_data_generation_arguments_from_model()

    def fill_in_data_generation_arguments_from_model(self):
        # FIXME: It would be better if we did not have to load an entire model just to
        #  gain access to the metadata. Store a copy of metadata separately?
        model = PosteriorModel(self.model, device='cpu', load_training_info=False)
        model_metadata = model.metadata

        domain = build_domain_from_model_metadata(model_metadata)

        self.duration = domain.duration
        self.minimum_frequency = domain.f_min
        self.maximum_frequency = domain.f_max


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
    complete_args = parser.parse([inputs.complete_ini_file])
    complete_inputs = input_cls(complete_args, "")
    ignore_keys = ["scheduler_module", "submit"]
    differences = []
    for key, val in inputs.__dict__.items():
        if key in ignore_keys:
            continue
        if key not in complete_args:
            continue
        if isinstance(val, pd.DataFrame) and all(val == complete_inputs.__dict__[key]):
            continue
        if isinstance(val, np.ndarray) and all(
                np.array(val) == np.array(complete_inputs.__dict__[key])
        ):
            continue
        if isinstance(val, str) and os.path.isfile(val):
            # Check if it is relpath vs abspath
            if os.path.abspath(val) == os.path.abspath(complete_inputs.__dict__[key]):
                continue
        if val == complete_inputs.__dict__[key]:
            continue

        differences.append(key)

    if len(differences) > 0:
        for key in differences:
            print(key, f"{inputs.__dict__[key]} -> {complete_inputs.__dict__[key]}")
        raise BilbyPipeError(
            "The written config file {} differs from the source {} in {}".format(
                inputs.ini, inputs.complete_ini_file, differences
            )
        )


def main():
    parser = create_parser(top_level=True)
    args, unknown_args = parse_args(get_command_line_arguments(), parser)

    inputs = MainInput(args, unknown_args)
    write_complete_config_file(parser, args, inputs)

    # TODO: Use two sets of inputs! The first must match the network; the second is
    #  used in importance sampling.

    generate_dag(inputs)

    if len(unknown_args) > 0:
        print(f"Unrecognized arguments {unknown_args}")
