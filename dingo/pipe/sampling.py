#!/usr/bin/env python
""" Script to sample from a Dingo model. Based on bilby_pipe data analysis script. """
import sys
from pathlib import Path
import os

from bilby_pipe.input import Input
from bilby_pipe.utils import parse_args, logger, convert_string_to_dict

from dingo.core.posterior_models.build_model import build_model_from_kwargs
from dingo.gw.data.event_dataset import EventDataset
from dingo.gw.inference.gw_samplers import GWSampler, GWSamplerGNPE
from dingo.gw.inference.inference_utils import prepare_log_prob
from dingo.pipe.default_settings import DENSITY_RECOVERY_SETTINGS
from dingo.pipe.parser import create_parser

logger.name = "dingo_pipe"


class SamplingInput(Input):
    def __init__(self, args, unknown_args):
        super().__init__(args, unknown_args)

        # Generic initialisation
        self.meta_data = dict()
        self.result = None

        # Admin arguments
        self.ini = args.ini
        self.scheduler = args.scheduler
        # self.periodic_restart_time = args.periodic_restart_time
        self.request_cpus = args.request_cpus
        self.n_parallel = args.n_parallel

        # Naming arguments
        self.outdir = args.outdir
        self.label = args.label
        self.result_format = args.result_format

        # Event file to run on
        self.event_data_file = args.event_data_file

        # Choices for running
        self.detectors = args.detectors

        # if running on the OSG, the network has been transferred to
        # the local directory, replace osdf string
        if args.osg:
            self.model = os.path.basename(args.model)
            self.model_init = os.path.basename(args.model_init)
        else:
            self.model = args.model
            self.model_init = args.model_init
        self.recover_log_prob = args.recover_log_prob
        self.device = args.device
        self.num_gnpe_iterations = args.num_gnpe_iterations
        self.num_samples = args.num_samples
        self.batch_size = args.batch_size
        self.density_recovery_settings = args.density_recovery_settings

        # self.sampler = args.sampler
        # self.sampler_kwargs = args.sampler_kwargs
        # self.sampling_seed = args.sampling_seed

        # Frequencies
        # self.sampling_frequency = args.sampling_frequency
        # self.minimum_frequency = args.minimum_frequency
        # self.maximum_frequency = args.maximum_frequency
        # self.reference_frequency = args.reference_frequency

        # # Waveform, source model and likelihood
        # self.waveform_generator_class = args.waveform_generator
        # self.waveform_approximant = args.waveform_approximant
        # self.catch_waveform_errors = args.catch_waveform_errors
        # self.pn_spin_order = args.pn_spin_order
        # self.pn_tidal_order = args.pn_tidal_order
        # self.pn_phase_order = args.pn_phase_order
        # self.pn_amplitude_order = args.pn_amplitude_order
        # self.mode_array = args.mode_array
        # self.waveform_arguments_dict = args.waveform_arguments_dict
        # self.numerical_relativity_file = args.numerical_relativity_file
        # self.frequency_domain_source_model = args.frequency_domain_source_model
        # self.conversion_function = args.conversion_function
        # self.generation_function = args.generation_function
        # self.likelihood_type = args.likelihood_type
        # self.reference_frame = args.reference_frame
        # self.time_reference = args.time_reference
        # self.extra_likelihood_kwargs = args.extra_likelihood_kwargs
        # self.enforce_signal_duration = args.enforce_signal_duration
        #
        # # ROQ
        # self.roq_folder = args.roq_folder
        # self.roq_scale_factor = args.roq_scale_factor
        #
        # # Calibration
        # self.calibration_model = args.calibration_model
        # self.spline_calibration_nodes = args.spline_calibration_nodes
        # self.spline_calibration_envelope_dict = args.spline_calibration_envelope_dict

        # # Marginalization
        # self.distance_marginalization = args.distance_marginalization
        # self.distance_marginalization_lookup_table = None
        # self.phase_marginalization = args.phase_marginalization
        # self.time_marginalization = args.time_marginalization
        # self.jitter_time = args.jitter_time

        self._load_event()
        self._load_sampler()

    def _load_event(self):
        event_dataset = EventDataset(file_name=self.event_data_file)
        self.context = event_dataset.data
        self.event_metadata = event_dataset.settings

    def _load_sampler(self):
        """Load the sampler and set its context based on event data."""
        model = build_model_from_kwargs(
            filename=self.model, device=self.device, load_training_info=False
        )

        if self.model_init is not None:
            self.gnpe = True
            init_model = build_model_from_kwargs(
                filename=self.model_init, device=self.device, load_training_info=False
            )
            init_sampler = GWSampler(model=init_model)
            self.dingo_sampler = GWSamplerGNPE(
                model=model,
                init_sampler=init_sampler,
                num_iterations=self.num_gnpe_iterations,
            )

        else:
            self.gnpe = False
            self.dingo_sampler = GWSampler(model=model)

        self.dingo_sampler.context = self.context
        self.dingo_sampler.event_metadata = self.event_metadata

    @property
    def density_recovery_settings(self):
        return self._density_recovery_settings

    @density_recovery_settings.setter
    def density_recovery_settings(self, settings):
        if self.recover_log_prob:
            self._density_recovery_settings = DENSITY_RECOVERY_SETTINGS[
                "ProxyRecoveryDefault"
            ]
        else:
            self._density_recovery_settings = dict()

        if settings is not None:
            if settings.lower() in ["default", "proxyrecoverydefault"]:
                self._density_recovery_settings.update(
                    DENSITY_RECOVERY_SETTINGS["ProxyRecoveryDefault"]
                )
            else:
                self._density_recovery_settings.update(convert_string_to_dict(settings))

        # If there is only one detector, and no context, we cannot use a coupling transform. In this case, we use an
        # autoregressive transform for the density estimator.
        # FIXME: If there are proxies other than time, the condition needs to be updated.
        if len(self.detectors) == 1:
            model_settings = self._density_recovery_settings["nde_settings"]["model"]
            if model_settings["posterior_model_type"] == "normalizing_flow":
                base_transform_kwargs = model_settings["posterior_kwargs"][
                    "base_transform_kwargs"
                ]
                if base_transform_kwargs["base_transform_type"] == "rq-coupling":
                    logger.info(
                        "Using autoregressive transform for density estimator since there is only one GNPE proxy "
                        "parameter because there is only one detector."
                    )
                    base_transform_kwargs["base_transform_type"] = "rq-autoregressive"

    # @property
    # def result_directory(self):
    #     result_dir = os.path.join(self.outdir, "result")
    #     return os.path.relpath(result_dir)

    def run_sampler(self):
        if self.gnpe and self.recover_log_prob:
            logger.info(
                "GNPE network does not provide log probability. Generating "
                "samples and training a new network to recover it."
            )

            # Note that this will not save any low latency samples at present.
            prepare_log_prob(
                self.dingo_sampler,
                batch_size=self.batch_size,
                **self.density_recovery_settings,
            )

        self.dingo_sampler.run_sampler(self.num_samples, batch_size=self.batch_size)
        self.dingo_sampler.to_hdf5(label=self.label, outdir=self.result_directory)

        if self.n_parallel > 1:
            logger.info(f"Splitting Result into {self.n_parallel} parts.")
            result = self.dingo_sampler.to_result()
            sub_results = result.split(self.n_parallel)
            outdir = Path(self.result_directory)
            outdir.mkdir(parents=True, exist_ok=True)
            for n, r in enumerate(sub_results):
                file_name = self.label + f"_part{n}.hdf5"
                r.to_file(file_name=outdir / file_name)


def create_sampling_parser():
    """Data analysis parser creation"""
    return create_parser(top_level=False)


def main():
    """Data analysis main logic"""
    args, unknown_args = parse_args(sys.argv[1:], create_sampling_parser())
    # log_version_information()
    analysis = SamplingInput(args, unknown_args)
    analysis.run_sampler()
    sys.exit(0)
