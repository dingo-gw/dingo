#!/usr/bin/env python
"""Script to importance sample based on Dingo samples. Based on bilby_pipe data
analysis script."""

import os
import sys

import bilby
import numpy as np
import torch
import yaml
from bilby_pipe.input import Input
from bilby_pipe.utils import (
    parse_args,
    logger,
    convert_string_to_dict,
    convert_prior_string_input,
    resolve_filename_with_transfer_fallback,
    BilbyPipeError,
)

from dingo.gw.data.event_dataset import EventDataset
from dingo.gw.domains import MultibandedFrequencyDomain, build_domain
from dingo.gw.waveform_generator import (
    NewInterfaceWaveformGenerator,
    WaveformGenerator,
)
from dingo.pipe.default_settings import IMPORTANCE_SAMPLING_SETTINGS
from dingo.pipe.parser import create_parser
from dingo.gw.result import Result

logger.name = "dingo_pipe"


class ImportanceSamplingInput(Input):
    def __init__(self, args, unknown_args):
        super().__init__(args, unknown_args)

        # Generic initialisation
        self.meta_data = dict()
        self.result = None

        # Admin arguments
        self.ini = args.ini
        self.scheduler = args.scheduler
        # self.periodic_restart_time = args.periodic_restart_time
        self.request_cpus = args.request_cpus_importance_sampling

        # Naming arguments
        self.outdir = args.outdir
        self.label = args.label
        self.result_format = args.result_format

        # Samples to run on
        self.proposal_samples_file = args.proposal_samples_file
        self.event_data_file = args.event_data_file

        # Prior
        self.prior_dict = args.prior_dict
        self.default_prior = "PriorDict"
        self.time_reference = "geocent"
        self.prior_dict_updates = args.prior_dict_updates

        # Choices for running
        self.detectors = args.detectors

        # self.sampler = args.sampler
        # self.sampler_kwargs = args.sampler_kwargs
        self.sampling_seed = args.sampling_seed

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
        # Calibration
        self.calibration_model = args.calibration_model
        self.calibration_mode = args.calibration_mode
        self.spline_calibration_nodes = args.spline_calibration_nodes
        self.spline_calibration_envelope_dict = args.spline_calibration_envelope_dict
        self.spline_calibration_curves = args.spline_calibration_curves
        self.calibration_correction_type = args.calibration_correction_type

        # # Marginalization
        # self.distance_marginalization = args.distance_marginalization
        # self.distance_marginalization_lookup_table = None
        # self.phase_marginalization = args.phase_marginalization
        # self.time_marginalization = args.time_marginalization
        # self.jitter_time = args.jitter_time

        self._load_proposal()
        self._load_event()  # Must be called after _load_proposal().
        self.importance_sampling_settings = args.importance_sampling_settings

    @property
    def request_memory(self):
        return self.inputs.request_memory_importance_sampling

    @property
    def sampling_seed(self):
        return self._sampling_seed

    @sampling_seed.setter
    def sampling_seed(self, sampling_seed):
        """Mirrors bilby_pipe's DataAnalysisInput, plus torch. The Pool workers of
        importance sampling and the synthetic phase are not re-seeded: under fork
        they copy one stream, under spawn they start unseeded (#408)."""
        if sampling_seed is None:
            sampling_seed = np.random.randint(1, 1e6)
        self._sampling_seed = int(sampling_seed)
        torch.manual_seed(self._sampling_seed)
        np.random.seed(self._sampling_seed)
        bilby.core.utils.random.seed(self._sampling_seed)
        logger.info(f"Sampling seed set to {self._sampling_seed}")

    def _load_proposal(self):
        self.result = Result(file_name=self.proposal_samples_file)
        if "log_prob" not in self.result.samples.columns:
            raise KeyError(
                "log_prob is not present in proposal samples. This is "
                "required for importance sampling."
            )

    def _load_event(self):
        event_dataset = EventDataset(file_name=self.event_data_file)
        self.result.reset_event(event_dataset)

    @property
    def calibration_marginalization_kwargs(self):
        if (
            self.calibration_model == "CubicSpline"
            and self.calibration_mode == "marginalize"
        ):
            return {
                "calibration_envelope": {
                    ifo: resolve_filename_with_transfer_fallback(path)
                    for ifo, path in self.spline_calibration_envelope_dict.items()
                },
                "num_calibration_nodes": self.spline_calibration_nodes,
                "num_calibration_curves": self.spline_calibration_curves,
                "correction_type": self.calibration_correction_type,
            }
        elif self.calibration_model is None or self.calibration_mode in [
            "sample",
            None,
        ]:
            return None
        else:
            raise ValueError(
                "The only calibration model which is supported is 'CubicSpline'"
            )

    @property
    def importance_sampling_settings(self):
        return self._importance_sampling_settings

    @importance_sampling_settings.setter
    def importance_sampling_settings(self, settings):
        # Set up defaults: recover the phase if the network does not infer it, and
        # psi along with it if the network infers neither.
        if "phase" not in self.result.samples.columns:
            default = (
                "PhaseRecoveryDefault"
                if "psi" in self.result.samples.columns
                else "PhasePsiRecoveryDefault"
            )
            # Copied, since the settings are updated below.
            self._importance_sampling_settings = dict(
                IMPORTANCE_SAMPLING_SETTINGS[default]
            )
        else:
            self._importance_sampling_settings = dict()

        if isinstance(self.result.domain, MultibandedFrequencyDomain):
            self._importance_sampling_settings.update(
                IMPORTANCE_SAMPLING_SETTINGS["MultibandingDefault"]
            )

        if settings is not None:
            if settings.lower() == "default":
                pass
            elif settings.lower() == "phaserecoverydefault":
                self._importance_sampling_settings.update(
                    IMPORTANCE_SAMPLING_SETTINGS["PhaseRecoveryDefault"]
                )
            elif settings.lower() == "phasepsirecoverydefault":
                self._importance_sampling_settings.update(
                    IMPORTANCE_SAMPLING_SETTINGS["PhasePsiRecoveryDefault"]
                )
            else:
                user_settings = convert_string_to_dict(settings)
                if "synthetic_phase" in user_settings:
                    raise ValueError(
                        "importance-sampling-settings: synthetic_phase has been renamed "
                        "to synthetic_parameters."
                    )
                self._importance_sampling_settings.update(user_settings)
            if "phase_marginalization" in self._importance_sampling_settings:
                self._importance_sampling_settings.pop("synthetic_parameters", None)
        else:
            self._importance_sampling_settings = dict()

        # Add calibration sampling if mode is "sample"
        if self.calibration_mode == "sample":
            if self.calibration_model == "CubicSpline":
                self._importance_sampling_settings["calibration_sampling_settings"] = {
                    "calibration_envelope": {
                        ifo: resolve_filename_with_transfer_fallback(path)
                        for ifo, path in self.spline_calibration_envelope_dict.items()
                    },
                    "num_calibration_nodes": self.spline_calibration_nodes,
                    "correction_type": self.calibration_correction_type,
                }
            else:
                raise NotImplementedError(
                    "The only calibration model which is supported is 'CubicSpline' "
                    "with calibration_mode set to 'sample'"
                )

    def run_sampler(self):
        self.result.use_base_domain = self.importance_sampling_settings.get(
            "use_base_domain", False
        )

        if self.prior_dict_updates:
            logger.info("Updating prior from network prior. Changes:")
            logger.info(
                yaml.dump(
                    self.prior_dict_updates,
                    default_flow_style=False,
                    sort_keys=False,
                )
            )
            self.result.update_prior(self.prior_dict_updates)

        likelihood_kwargs = dict(
            time_marginalization_kwargs=self.importance_sampling_settings.get(
                "time_marginalization"
            ),
            phase_marginalization_kwargs=self.importance_sampling_settings.get(
                "phase_marginalization"
            ),
            calibration_marginalization_kwargs=self.calibration_marginalization_kwargs,
        )

        # Calibration parameters and synthetic phase are drawn in one chain, the
        # calibration first, so that the phase is conditioned on it.
        synthetic_parameters_kwargs = None
        use_cached_log_likelihood = False
        if "synthetic_parameters" in self.importance_sampling_settings:
            synthetic_parameters_kwargs = {
                **self.importance_sampling_settings["synthetic_parameters"],
                "num_processes": self.request_cpus,
            }
            # The synthetic phase can cache the log likelihood at the drawn phase for
            # importance sampling, unless it uses the (2, 2)-mode approximation,
            # importance sampling uses a marginalized likelihood, or the cached value
            # would not be exact.
            use_cached_log_likelihood = synthetic_parameters_kwargs.get(
                "cache_log_likelihood", True
            )
            # When psi is drawn, SyntheticPhasePsiFactor always uses the exact mode
            # sum; approximation_22_mode (default True) only applies to phase only.
            if use_cached_log_likelihood and (
                (
                    self.result.psi_prior is None
                    and synthetic_parameters_kwargs.get("approximation_22_mode", True)
                )
                or any(likelihood_kwargs.values())
                or not self._synthetic_parameters_modes_exact(
                    synthetic_parameters_kwargs
                )
            ):
                logger.info(
                    "Not caching the synthetic phase log likelihood (incompatible "
                    "with approximation_22_mode, a marginalized likelihood, or the "
                    "waveform model's mode decomposition)."
                )
                use_cached_log_likelihood = False
            synthetic_parameters_kwargs["cache_log_likelihood"] = (
                use_cached_log_likelihood
            )
        calibration_sampling_kwargs = self.importance_sampling_settings.get(
            "calibration_sampling_settings"
        )
        if synthetic_parameters_kwargs or calibration_sampling_kwargs:
            if synthetic_parameters_kwargs:
                logger.info("Sampling synthetic phase")
            elif calibration_sampling_kwargs: 
                logger.info("Sampling calibration parameters")
            self.result.sample_proposal_extensions(
                calibration_sampling_kwargs=calibration_sampling_kwargs,
                synthetic_parameters_kwargs=synthetic_parameters_kwargs,
            )

        self.result.importance_sample(
            num_processes=self.request_cpus,
            use_cached_log_likelihood=use_cached_log_likelihood,
            **likelihood_kwargs,
        )

        self.result.print_summary()
        self.result.to_file(os.path.join(self.result_directory, self.label + ".hdf5"))

    def _synthetic_parameters_modes_exact(
        self, synthetic_parameters_kwargs: dict
    ) -> bool:
        """
        Whether the cached synthetic phase log likelihood is exact: it is computed
        from the m-components of the waveform, which sum to exactly the waveform the
        direct likelihood uses only if the waveform generator uses the DFT phase
        decomposition.
        """
        dataset_settings = self.result.base_metadata["dataset_settings"]
        wfg_settings = dict(dataset_settings["waveform_generator"])
        if "use_dft_phase_decomposition" in synthetic_parameters_kwargs:
            wfg_settings["use_dft_phase_decomposition"] = synthetic_parameters_kwargs[
                "use_dft_phase_decomposition"
            ]
        if wfg_settings.get("new_interface", False):
            wfg_class = NewInterfaceWaveformGenerator
        else:
            wfg_class = WaveformGenerator
        waveform_generator = wfg_class(
            domain=build_domain(dataset_settings["domain"]), **wfg_settings
        )
        return waveform_generator.uses_dft_phase_decomposition

    @property
    def priors(self):
        """Read in and compose the prior at run-time"""
        if getattr(self, "_priors", None) is None:
            self._priors = self._get_priors(add_time=False)
        return self._priors

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


def create_sampling_parser():
    """Data analysis parser creation"""
    return create_parser(top_level=False)


def main():
    """Data analysis main logic"""
    args, unknown_args = parse_args(sys.argv[1:], create_sampling_parser())
    # log_version_information()
    analysis = ImportanceSamplingInput(args, unknown_args)
    analysis.run_sampler()
    sys.exit(0)
