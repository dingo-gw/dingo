import os
import sys
import ast

from bilby_pipe.input import Input
from bilby_pipe.main import parse_args
from bilby_pipe.utils import logger, convert_string_to_dict
from bilby_pipe.data_generation import DataGenerationInput as BilbyDataGenerationInput
import numpy as np
import lalsimulation as LS

from dingo.gw.data.event_dataset import EventDataset
from dingo.gw.domains import FrequencyDomain
from dingo.pipe.parser import create_parser
from dingo.gw.injection import Injection
from dingo.core.models import PosteriorModel
from dingo.gw.noise.asd_dataset import ASDDataset
from dingo.gw.data.data_preparation import (
    load_raw_data,
    data_to_domain,
    build_domain_from_model_metadata,
    parse_settings_for_raw_data,
)


logger.name = "dingo_pipe"


class DataGenerationInput(BilbyDataGenerationInput):
    def __init__(self, args, unknown_args, create_data=True):
        Input.__init__(self, args, unknown_args)

        # Generic initialisation
        self.meta_data = dict(
            command_line_args=args.__dict__,
            unknown_command_line_args=unknown_args,
            injection_dict=args.injection_dict,
            # bilby_version=bilby.__version__,
            # bilby_pipe_version=get_version_information(),
        )

        # Admin arguments
        self.ini = args.ini

        # Run index arguments
        self.idx = args.idx
        # self.generation_seed = args.generation_seed
        self.trigger_time = args.trigger_time

        # Naming arguments
        self.outdir = args.outdir
        self.label = args.label

        # If creating an injection no need for real data generation
        if args.injection_dict is not None:
            self.injection_dict = ast.literal_eval(args.injection_dict)
            self.injection_dict = {
                k.replace("-", "_"): v for k, v in self.injection_dict.items()
            }
            self.generate_injection(args)
            return

        # Prior arguments
        # self.reference_frame = args.reference_frame
        # self.time_reference = args.time_reference
        # self.phase_marginalization = args.phase_marginalization
        # self.prior_file = args.prior_file
        # self.prior_dict = args.prior_dict
        # self.deltaT = args.deltaT
        # self.default_prior = args.default_prior

        # Whether to generate data for importance sampling. This must be done when
        # desired data settings differ from those used for network training. If this is
        # the case, save the new data to a different file name.

        self.importance_sampling = args.importance_sampling_generation
        self.importance_sampling_updates = args.importance_sampling_updates
        if self.importance_sampling:
            vars(args).update(self.importance_sampling_updates)

        # Data arguments
        self.ignore_gwpy_data_quality_check = args.ignore_gwpy_data_quality_check
        self.detectors = args.detectors
        self.channel_dict = args.channel_dict
        self.data_dict = args.data_dict
        self.data_format = args.data_format
        self.allow_tape = args.allow_tape
        self.tukey_roll_off = args.tukey_roll_off
        self.zero_noise = False  # dingo mod
        self.resampling_method = args.resampling_method

        # if args.timeslide_dict is not None:
        #     self.timeslide_dict = convert_string_to_dict(args.timeslide_dict)
        #     logger.info(f"Read-in timeslide dict directly: {self.timeslide_dict}")
        # elif args.timeslide_file is not None:
        #     self.gps_file = args.gps_file
        #     self.timeslide_file = args.timeslide_file
        #     self.timeslide_dict = self.get_timeslide_dict(self.idx)

        # Data duration arguments
        self.duration = args.duration
        self.post_trigger_duration = args.post_trigger_duration

        # Frequencies
        self.sampling_frequency = args.sampling_frequency
        self.minimum_frequency = args.minimum_frequency
        self.maximum_frequency = args.maximum_frequency
        # self.reference_frequency = args.reference_frequency

        # Waveform, source model and likelihood
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
        # self.extra_likelihood_kwargs = args.extra_likelihood_kwargs
        # self.enforce_signal_duration = args.enforce_signal_duration

        # PSD
        self.psd_maximum_duration = args.psd_maximum_duration
        self.psd_dict = args.psd_dict
        if self.psd_dict is None:
            self.psd_length = args.psd_length
            self.psd_fractional_overlap = args.psd_fractional_overlap
            self.psd_start_time = args.psd_start_time
            self.psd_method = args.psd_method

        # # ROQ
        # self.roq_folder = args.roq_folder
        # self.roq_linear_matrix = args.roq_linear_matrix
        # self.roq_quadratic_matrix = args.roq_quadratic_matrix
        # self.roq_weights = args.roq_weights
        # self.roq_weight_format = args.roq_weight_format
        # self.roq_scale_factor = args.roq_scale_factor
        #
        # # Calibration
        # self.calibration_model = args.calibration_model
        # self.spline_calibration_envelope_dict = args.spline_calibration_envelope_dict
        # self.spline_calibration_amplitude_uncertainty_dict = (
        #     args.spline_calibration_amplitude_uncertainty_dict
        # )
        # self.spline_calibration_phase_uncertainty_dict = (
        #     args.spline_calibration_phase_uncertainty_dict
        # )
        # self.spline_calibration_nodes = args.spline_calibration_nodes
        # self.calibration_prior_boundary = args.calibration_prior_boundary
        #
        # # Marginalization
        # self.distance_marginalization = args.distance_marginalization
        # self.distance_marginalization_lookup_table = (
        #     args.distance_marginalization_lookup_table
        # )
        # self.phase_marginalization = args.phase_marginalization
        # self.time_marginalization = args.time_marginalization
        # self.jitter_time = args.jitter_time
        #
        # # Plotting
        self.plot_data = args.plot_data
        self.plot_spectrogram = args.plot_spectrogram
        # self.plot_injection = args.plot_injection

        if create_data:
            # TODO what happens if we remove this??
            args.injection = False
            args.injection_numbers = None
            args.injection_file = None
            args.injection_dict = None
            args.gaussian_noise = False
            args.injection_waveform_arguments = None
            self.create_data(args)

    def generate_injection(self, args):
        """Generate injection consistent with trained dingo model"""
        # loading posterior model for which we want to generate injections
        pm = PosteriorModel(model_filename=args.model, device="cpu")
        injection_generator = Injection.from_posterior_model_metadata(pm.metadata)

        # selecting PSD
        if args.use_psd_of_trigger:
            trigger_time = float(args.trigger_time)
            domain = build_domain_from_model_metadata(pm.metadata)
            settings_raw_data = parse_settings_for_raw_data(
                pm.metadata, args.psd_length, args.psd_fractional_overlap
            )
            raw_data = load_raw_data(trigger_time, settings=settings_raw_data)

            # Converting data to frequency series
            event_data = data_to_domain(
                raw_data,
                settings_raw_data,
                domain,
                window=pm.metadata["train_settings"]["data"]["window"],
            )
            injection_generator.asd = event_data["asds"]
        else:
            asd_dataset = ASDDataset(args.asd_dataset)
            randint = np.random.randint(
                0, [v for v in asd_dataset.length_info.values()][0]
            )
            injection_generator.asd = {
                k: v[randint]
                for k, v in asd_dataset.asds.items()
                if k in [ifo.name for ifo in injection_generator.ifo_list]
            }

        # allowing for changing waveform approximant injection
        if args.injection_waveform_approximant is not None:
            injection_generator.waveform_generator.approximant = (
                LS.GetApproximantFromString(args.injection_waveform_approximant)
            )
            injection_generator.waveform_generator.approximant_str = (
                args.injection_waveform_approximant
            )
            self.injection_waveform_approximant = args.injection_waveform_approximant
        else:
            self.injection_waveform_approximant = (
                injection_generator.waveform_generator.approximant_str
            )

        # the trigger time determines how the waveform is injected based on the rotation of earth
        injection_generator.t_ref = trigger_time

        self.detectors = [ifo.name for ifo in injection_generator.ifo_list]
        self.sampling_frequency = injection_generator.waveform_generator.domain.sampling_rate
        self.duration = injection_generator.data_domain.duration
        self.minimum_frequency = injection_generator.data_domain.f_min
        self.maximum_frequency = injection_generator.data_domain.f_max
        self.window_type = pm.metadata["train_settings"]["data"]["window"]["type"]
        self.tukey_roll_off = pm.metadata["train_settings"]["data"]["window"][
            "roll_off"
        ]

        self.asd = injection_generator.asd
        self.strain_data = injection_generator.injection(
            self.injection_dict, seed=args.injection_random_seed
        )

    def save_hdf5(self):
        """Save frequency-domain strain and ASDs as DingoDataset HDF5 format."""

        # if the data is created via an injection, we don't need to convert anything
        # from the Bilby format
        if hasattr(self, "strain_data"):
            settings = {
                "detectors": self.detectors,
                "f_s": self.sampling_frequency,
                "detectors": self.detectors,
                "T": self.duration,
                "f_min": self.minimum_frequency,
                "f_max": self.maximum_frequency,
                "window_type": self.window_type,
                "roll_off": self.tukey_roll_off,
            }
            dataset = EventDataset(
                dictionary={
                    "data": self.strain_data,
                    "injection_waveform_approximant": self.injection_waveform_approximant,
                    "injection_dict": self.injection_dict,
                    "settings": settings,
                }
            )

        else:
            # PSD and strain data.
            data = {"waveform": {}, "asds": {}}  # TODO: Rename these keys.
            for ifo in self.interferometers:
                strain = ifo.strain_data.frequency_domain_strain
                frequency_array = ifo.strain_data.frequency_array
                asd = ifo.power_spectral_density.get_amplitude_spectral_density_array(
                    frequency_array
                )

                # These arrays extend up to self.sampling_frequency. Truncate them to
                # self.maximum_frequency, and also set the asd to 1.0 below
                # self.minimum_frequency.
                domain = FrequencyDomain(
                    f_min=self.minimum_frequency,
                    f_max=self.maximum_frequency,
                    delta_f=1 / self.duration,
                )
                strain = domain.update_data(strain)
                asd = domain.update_data(asd, low_value=1.0)

                # Dingo expects data to have trigger time 0, so we apply a cyclic time shift
                # by the post-trigger duration.
                strain = domain.time_translate_data(strain, self.post_trigger_duration)

                # Note that the ASD estimated by the Bilby Interferometer differs ever so
                # slightly from the ASDs we computed before using pycbc. In addition,
                # there is no handling of NaNs in the strain data from which the ASD is
                # estimated.

                data["waveform"][ifo.name] = strain
                data["asds"][ifo.name] = asd

            # Data conditioning settings.
            settings = {
                "time_event": self.trigger_time,
                "time_buffer": self.post_trigger_duration,
                "detectors": self.detectors,
                "f_s": self.sampling_frequency,
                "T": self.duration,
                "f_min": self.minimum_frequency,
                "f_max": self.maximum_frequency,
                "window_type": "tukey",
                "roll_off": self.tukey_roll_off,
            }

            for k in [
                "psd_duration",
                "psd_dict",
                "psd_fractional_overlap",
                "psd_start_time",
                "psd_method",
                "channel_dict",
                "data_dict",
            ]:
                try:
                    v = getattr(self, k)
                except AttributeError:
                    continue
                if v is not None:
                    settings[k] = v

            dataset = EventDataset(
                dictionary={
                    "data": data,
                    # "event_metadata": event_metadata,
                    "settings": settings,
                }
            )

        dataset.to_file(self.event_data_file)

    @property
    def event_data_file(self):
        return os.path.join(
            self.data_directory, "_".join([self.label, "event_data.hdf5"])
        )

    @property
    def importance_sampling_updates(self):
        return self._importance_sampling_updates

    @importance_sampling_updates.setter
    def importance_sampling_updates(self, setting):
        if setting is not None:
            self._importance_sampling_updates = convert_string_to_dict(
                setting, "importance-sampling-updates"
            )
        else:
            self._importance_sampling_updates = None


def create_generation_parser():
    """Data generation parser creation"""
    return create_parser(top_level=False)


def main():
    """Data generation main logic"""
    args, unknown_args = parse_args(sys.argv[1:], create_generation_parser())
    # log_version_information()
    data = DataGenerationInput(args, unknown_args)
    data.save_hdf5()
    logger.info("Completed data generation")
