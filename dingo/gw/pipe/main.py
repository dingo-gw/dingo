#
#  Adapted from bilby_pipe. In particular, uses the bilby_pipe data generation code.
#
import os

from bilby_pipe.input import Input

from .dag_creator import generate_dag
from .parser import create_parser
from bilby_pipe.utils import parse_args, get_command_line_arguments


class MainInput(Input):

    def __init__(self, args, unknown_args):
        super().__init__(args, unknown_args, print_msg=False)

        self.known_args = args
        self.unknown_args = unknown_args
        self.ini = args.ini

        self.outdir = args.outdir
        self.label = args.label
        self.scheduler = args.scheduler

        self.trigger_time = args.trigger_time
        self.channel_dict = args.channel_dict

        self.model = args.model
        self.init_model = args.init_model
        self.num_samples = args.num_samples
        self.num_gnpe_iterations = args.num_gnpe_iterations
        self.n_parallel = 1

        self.run_local = args.local

    @property
    def ini(self):
        return self._ini

    @ini.setter
    def ini(self, ini):
        if os.path.isfile(ini) is False:
            raise FileNotFoundError(f"No ini file {ini} found")
        self._ini = os.path.relpath(ini)


def write_complete_config_file(parser, args, inputs, input_cls=MainInput):
    pass


def main():
    parser = create_parser(top_level=True)
    args, unknown_args = parse_args(get_command_line_arguments(), parser)

    inputs = MainInput(args, unknown_args)
    write_complete_config_file(parser, args, inputs)

    generate_dag(inputs)

    if len(unknown_args) > 0:
        print(f"Unrecognized arguments {unknown_args}")
