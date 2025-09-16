#
#  Adapted from bilby_pipe. In particular, uses the bilby_pipe data generation code.
#
import copy
import json
import os

import numpy as np

import dingo.pipe.create_injections

from bilby.core.prior import PriorDict
from bilby_pipe.input import Input
from bilby_pipe.main import MainInput as BilbyMainInput
from bilby_pipe.utils import (
    convert_string_to_dict,
    get_command_line_arguments,
    logger,
    parse_args,
    convert_prior_string_input,
    BilbyPipeError,
)

from .dag_creator import generate_dag
from .parser import create_parser
from .utils import dict_to_string

from ..gw.domains.build_domain import build_domain_from_model_metadata
from dingo.core.posterior_models.build_model import build_model_from_kwargs
from ..gw.injection import Injection
from ..gw.noise.asd_dataset import ASDDataset

logger.name = "dingo_pipe"


def fill_in_arguments_from_model(args, perform_arg_checks=True):
    if perform_arg_checks:
        if args.prior_dict is not None:
            raise ValueError(
                "Do not specify prior-dict in INI file. This is obtained from "
                "the DINGO model. To update the prior, specify "
                "prior-dict-updates."
            )
        if args.model_reference_time is not None:
            raise ValueError(
                "Do not specify model-reference-time in INI file. This is obtained from the "
                "DINGO model."
            )

    logger.info(f"Loading dingo model from {args.model} in order to access settings.")

    try:
        model = build_model_from_kwargs(
            filename=args.model, device="meta", load_training_info=False
        )
    except RuntimeError:
        # 'meta' is not supported by older version of python / torch
        model = build_model_from_kwargs(
            filename=args.model, device=args.device, load_training_info=False
        )

    model_metadata = model.metadata

    domain = build_domain_from_model_metadata(model_metadata, base=True)

    prior = Injection.from_posterior_model_metadata(model_metadata).prior
    geocent_time_prior = prior["geocent_time"]
    deltaT = geocent_time_prior.maximum - geocent_time_prior.minimum

    # Offset needed in case of DeltaFunction priors not peaked at 0.0.
    Toffset = 0.5 * (geocent_time_prior.minimum + geocent_time_prior.maximum)

    # Dingo and Bilby have different conventions for the geocent_time prior. Dingo
    # always centers it around something close to 0.0, whereas Bilby centers it around
    # the trigger time. Drop the geocent_time parameter here so that bilby_pipe
    # re-builds it as needed for Bilby.
    del prior["geocent_time"]

    data_settings = model_metadata["train_settings"]["data"]

    model_args = {
        "duration": domain.duration,
        "minimum_frequency": domain.f_min,
        "maximum_frequency": domain.f_max,
        "detectors": data_settings["detectors"],
        "sampling_frequency": data_settings["window"]["f_s"],
        "tukey_roll_off": data_settings["window"]["roll_off"],
        "waveform_approximant": model_metadata["dataset_settings"][
            "waveform_generator"
        ]["approximant"],
        "reference_frequency": model_metadata["dataset_settings"]["waveform_generator"][
            "f_ref"
        ],
        "deltaT": deltaT,
        "Toffset": Toffset,
        "prior_dict": dict_to_string(prior),
        "model_reference_time": model_metadata["train_settings"]["data"]["ref_time"],
    }

    changed_args = {}
    for k, v in model_args.items():
        args_v = getattr(args, k)
        if args_v is not None:
            # Convert type from str to enable comparison.
            try:
                # it's possible that the attribute is a dict 
                # or list.
                # for example, the minimum-frequency could be 
                # {H1: 20, L1:20}
                if "{" and "}" in args_v:
                    args_v = convert_string_to_dict(args_v)
                elif isinstance(args_v, list):
                    # if it's a list we need to strip quotes
                    # e.g. detectors = ["'H1'", "'L1'"]
                    # should become ["H1", "L1"] for the comparison
                    args_v = [l.strip("'") for l in args_v]
                elif isinstance(v, float):
                    args_v = float(args_v)
                elif isinstance(v, int):
                    args_v = int(args_v)
            except ValueError:
                pass

            if args_v != v:
                # this is for when the minimum-frequency is a float 
                # but args_v is a dict 
                if isinstance(args_v, dict) and all(val == v for val in args_v.values()):
                    continue

                if isinstance(args_v, list):
                    continue

                if k in ["waveform_approximant"]:
                    raise NotImplementedError(
                        "Cannot change waveform approximant during importance sampling."
                    )  # TODO: Implement this. Also no error if passed explicitly as an update.
                logger.warning(
                    f"Argument {k} provided to dingo_pipe as {args_v} "
                    f"does not match value {v} in model file. Using model value for "
                    f"inference, and will attempt to change this during importance "
                    f"sampling."
                )
                changed_args[k] = args_v

        setattr(args, k, v)

    # TODO: Also check consistency between model and init_model settings.

    # If an ASDDataset is specified, pull out PSDs and save them as .txt files. Use
    # these for downstream tasks by specifying the psd_dict.
    if args.asd_dataset:
        asd_dataset = ASDDataset(file_name=args.asd_dataset)
        domain_dict = copy.deepcopy(domain.domain_dict)
        if "window_factor" in domain_dict:
            del domain_dict["window_factor"]
        asd_dataset.update_domain(domain_dict)
        psd_dict = {}
        rng = np.random.default_rng(args.generation_seed)
        for ifo_name in args.detectors:
            psd_path = asd_dataset.save_psd(args.outdir, ifo_name, rng=rng)
            psd_dict[ifo_name] = str(psd_path)
        args.asd_dataset = None
        args.psd_dict = str(psd_dict)

    # Updates that are explicitly provided take priority.
    if args.importance_sampling_updates is None:
        importance_sampling_updates = {}
    else:
        importance_sampling_updates = convert_string_to_dict(
            args.importance_sampling_updates
        )
        importance_sampling_updates = {
            k.replace("-", "_"): v for k, v in importance_sampling_updates.items()
        }
    return {**changed_args, **importance_sampling_updates}, model_args


class MainInput(BilbyMainInput):
    def __init__(
        self, args, unknown_args, importance_sampling_updates, perform_checks=True
    ):
        # Settings added for dingo.

        self.model = args.model
        self.model_init = args.model_init
        self.num_gnpe_iterations = args.num_gnpe_iterations
        self.importance_sampling_updates = importance_sampling_updates
        self.prior_dict_updates = args.prior_dict_updates
        self.model_reference_time = args.model_reference_time

        Input.__init__(self, args, unknown_args, print_msg=False)

        self.known_args = args
        self.unknown_args = unknown_args
        self.ini = args.ini
        self.submit = args.submit
        self.condor_job_priority = args.condor_job_priority
        self.create_summary = args.create_summary
        self.scitoken_issuer = args.scitoken_issuer
        self.container = args.container

        self.outdir = args.outdir
        self.label = args.label
        self.log_directory = args.log_directory
        self.accounting = args.accounting
        self.accounting_user = args.accounting_user
        # self.sampler = args.sampler
        self.detectors = args.detectors
        self.coherence_test = (
            False  # dingo mod: Cannot use different sets of detectors.
        )
        self.data_dict = args.data_dict
        self.channel_dict = args.channel_dict
        self.frame_type_dict = args.frame_type_dict
        self.data_find_url = args.data_find_url
        self.data_find_urltype = args.data_find_urltype
        self.n_parallel = args.n_parallel
        # useful when condor nodes don't have access to submit filesystem
        self.transfer_files = args.transfer_files
        self.additional_transfer_paths = args.additional_transfer_paths
        self.osg = args.osg
        self.desired_sites = (
            args.cpu_desired_sites
        )  # Dummy variable so bilby_pipe doesn't complain.
        self.cpu_desired_sites = args.cpu_desired_sites
        self.gpu_desired_sites = args.gpu_desired_sites
        # self.analysis_executable = args.analysis_executable
        # self.analysis_executable_parser = args.analysis_executable_parser
        self.result_format = "hdf5"
        self.final_result = args.final_result
        self.final_result_nsamples = args.final_result_nsamples

        self.webdir = args.webdir
        self.email = args.email
        self.notification = args.notification
        self.queue = args.queue
        self.existing_dir = args.existing_dir

        self.scheduler = args.scheduler
        self.scheduler_args = args.scheduler_args
        self.scheduler_module = args.scheduler_module
        self.scheduler_env = args.scheduler_env
        self.scheduler_analysis_time = args.scheduler_analysis_time
        self.disable_hdf5_locking = args.disable_hdf5_locking
        self.environment_variables = args.environment_variables
        self.getenv = args.getenv

        self.waveform_approximant = args.waveform_approximant
        #
        # self.time_reference = args.time_reference
        self.time_reference = "geocent"
        # self.reference_frame = args.reference_frame
        # self.likelihood_type = args.likelihood_type
        self.duration = args.duration
        # self.phase_marginalization = args.phase_marginalization
        self.prior_file = None  # Dingo update. To change prior use the prior_dict.
        self.prior_dict = args.prior_dict
        self.default_prior = "BBHPriorDict"
        self.minimum_frequency = args.minimum_frequency
        self.enforce_signal_duration = args.enforce_signal_duration

        self.run_local = args.local
        self.generation_pool = args.generation_pool
        self.local_generation = args.local_generation
        self.local_plot = args.local_plot

        self.post_trigger_duration = args.post_trigger_duration

        # self.ignore_gwpy_data_quality_check = args.ignore_gwpy_data_quality_check
        self.trigger_time = args.trigger_time
        self.deltaT = args.deltaT
        self.Toffset = args.Toffset
        self.gps_tuple = args.gps_tuple
        self.gps_file = args.gps_file
        self.timeslide_file = args.timeslide_file
        self.gaussian_noise = args.gaussian_noise
        self.zero_noise = args.zero_noise
        self.n_simulation = args.n_simulation

        self.injection = args.injection
        self.injection_numbers = args.injection_numbers
        self.injection_file = args.injection_file
        self.injection_dict = args.injection_dict
        self.save_bilby_data_dump = args.save_bilby_data_dump
        # self.injection_waveform_arguments = args.injection_waveform_arguments
        # self.injection_waveform_approximant = args.injection_waveform_approximant
        # self.injection_frequency_domain_source_model = (
        #     args.injection_frequency_domain_source_model
        # )
        self.generation_seed = args.generation_seed
        if self.importance_sampling_updates and self.gaussian_noise:
            raise ValueError(
                "Cannot update data for importance sampling if using "
                "simulated Gaussian noise. This risks inconsistent noise "
                "realizations."
            )

        self.importance_sample = args.importance_sample

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
        self.plot_node_needed = False
        for plot_attr in [
            "corner",
            "weights",
            "log_probs",
        ]:
            attr = f"plot_{plot_attr}"
            setattr(self, attr, getattr(args, attr))
            if getattr(self, attr):
                self.plot_node_needed = True

        self.plot_pp = args.plot_pp
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
        self.summarypages_arguments = args.summarypages_arguments

        self.psd_dict = args.psd_dict
        self.psd_maximum_duration = args.psd_maximum_duration
        self.psd_length = args.psd_length
        self.psd_fractional_overlap = args.psd_fractional_overlap
        self.psd_start_time = args.psd_start_time
        self.spline_calibration_envelope_dict = args.spline_calibration_envelope_dict

        if perform_checks:
            # self.check_source_model(args)
            # self.check_calibration_prior_boundary(args)
            # self.check_cpu_parallelisation()
            if self.injection:
                self.check_injection()

        self.requirements = []
        self.device = args.device
        self.simple_submission = args.simple_submission

        if args.extra_lines:
            self.extra_lines = args.extra_lines
        else:
            self.extra_lines = []

        if args.sampling_requirements:
            self.sampling_requirements = args.sampling_requirements
        else:
            self.sampling_requirements = []

    @property
    def request_cpus_importance_sampling(self):
        return self._request_cpus_importance_sampling

    @request_cpus_importance_sampling.setter
    def request_cpus_importance_sampling(self, request_cpus_importance_sampling):
        logger.info(
            f"Setting analysis request_cpus_importance_sampling = "
            f"{request_cpus_importance_sampling}"
        )
        self._request_cpus_importance_sampling = request_cpus_importance_sampling

    @property
    def prior_dict_updates(self):
        """The input prior_dict from the ini (if given)

        Note, this is not the bilby prior (see self.priors for that), this is
        a key-val dictionary where the val's are strings which are converting
        into bilby priors in `_get_prior
        """
        return self._prior_dict_updates

    @prior_dict_updates.setter
    def prior_dict_updates(self, prior_dict_updates):
        if isinstance(prior_dict_updates, dict):
            prior_dict_updates = prior_dict_updates
        elif isinstance(prior_dict_updates, str):
            prior_dict_updates = convert_prior_string_input(prior_dict_updates)
        elif prior_dict_updates is None:
            self._prior_dict_updates = None
            return
        else:
            raise BilbyPipeError(
                f"prior_dict_updates={prior_dict_updates} not " f"understood"
            )

        self._prior_dict_updates = {
            self._convert_prior_dict_key(key): val
            for key, val in prior_dict_updates.items()
        }

    def _get_priors(self, add_time=True):
        priors = super()._get_priors(add_time=add_time)
        priors.update(PriorDict(self.prior_dict_updates))
        return priors

    def check_injection(self):
        """Check injection behaviour

        If injections are requested, either use the injection-dict,
        injection-file, or create an injection-file

        """
        default_injection_file_name = "{}/{}_injection_file.dat".format(
            self.data_directory, self.label
        )
        if self.injection_dict is not None:
            logger.debug(
                "Using injection dict from ini file {}".format(
                    json.dumps(self.injection_dict, indent=2)
                )
            )
        elif self.injection_file is not None:
            logger.debug(f"Using injection file {self.injection_file}")
        elif os.path.isfile(default_injection_file_name):
            # This is done to avoid overwriting the injection file
            logger.debug(f"Using injection file {default_injection_file_name}")
            self.injection_file = default_injection_file_name
        else:
            logger.debug("No injection file found, generating one now")

            if self.gps_file is not None or self.gps_tuple is not None:
                if self.n_simulation > 0 and self.n_simulation != len(self.gpstimes):
                    raise BilbyPipeError(
                        "gps_file/gps_tuple option and n_simulation are not matched"
                    )
                gpstimes = self.gpstimes
                n_injection = len(gpstimes)
            else:
                gpstimes = None
                n_injection = self.n_simulation

            if self.trigger_time is None:
                # Use the model reference time as the trigger time by default. This
                # avoids needing to adjust sky position in post-postprocessing,
                # and it avoids complications due to DeltaFunction ra priors.
                trigger_time_injections = self.model_reference_time
            else:
                trigger_time_injections = self.trigger_time

            dingo.pipe.create_injections.create_injection_file(
                filename=default_injection_file_name,
                prior_file=self.prior_file,
                prior_dict=self._get_priors(
                    add_time=False
                ),  # ensure prior-dict-updates incorporated
                n_injection=n_injection,
                trigger_time=trigger_time_injections,
                deltaT=self.deltaT,
                Toffset=self.Toffset,  # dingo-pipe mod
                gpstimes=gpstimes,
                duration=self.duration,
                post_trigger_duration=self.post_trigger_duration,
                generation_seed=self.generation_seed,
                extension="dat",
                default_prior=self.default_prior,
            )
            self.injection_file = default_injection_file_name

        # Check the gps_file has the sample length as number of simulation
        if self.gps_file is not None:
            if len(self.gpstimes) != len(self.injection_df):
                raise BilbyPipeError("Injection file length does not match gps_file")

        if self.n_simulation > 0:
            if self.n_simulation != len(self.injection_df):
                raise BilbyPipeError(
                    "n-simulation does not match the number of injections: "
                    "please check your ini file"
                )
        elif self.n_simulation == 0 and self.gps_file is None:
            self.n_simulation = len(self.injection_df)
            logger.debug(
                f"Setting n_simulation={self.n_simulation} to match injections"
            )


def write_complete_config_file(parser, args, inputs, input_cls=MainInput):
    args_dict = vars(args).copy()
    for key, val in args_dict.items():
        if key == "label":
            continue
        if isinstance(val, str):
            if os.path.isfile(val) or os.path.isdir(val):
                if not args.osg:
                    setattr(args, key, os.path.abspath(val))
                else:
                    setattr(args, key, val)
        if isinstance(val, list):
            if isinstance(val[0], str):
                setattr(args, key, f"[{', '.join(val)}]")
    # args.sampler_kwargs = str(inputs.sampler_kwargs)
    args.importance_sampling_updates = str(inputs.importance_sampling_updates)
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

    importance_sampling_updates, model_args = fill_in_arguments_from_model(args)
    inputs = MainInput(args, unknown_args, importance_sampling_updates)
    write_complete_config_file(parser, args, inputs)

    # TODO: Use two sets of inputs! The first must match the network; the second is
    #  used in importance sampling.

    generate_dag(inputs)

    if len(unknown_args) > 0:
        print(f"Unrecognized arguments {unknown_args}")
