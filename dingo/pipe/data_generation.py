import os
import sys

from bilby_pipe.input import Input
from bilby_pipe.main import parse_args
from bilby_pipe.utils import logger, convert_string_to_dict
from bilby_pipe.data_generation import DataGenerationInput as BilbyDataGenerationInput

from dingo.gw.data.event_dataset import EventDataset
from dingo.gw.domains import FrequencyDomain
from dingo.pipe.parser import create_parser

logger.name = "dingo_pipe"


class DataGenerationInput(BilbyDataGenerationInput):
    def __init__(self, args, unknown_args, create_data=True):
        Input.__init__(self, args, unknown_args)

        # Generic initialisation
        self.meta_data = dict(
            command_line_args=args.__dict__,
            unknown_command_line_args=unknown_args,
            injection_parameters=None,
            # bilby_version=bilby.__version__,
            # bilby_pipe_version=get_version_information(),
        )
        self.injection_parameters = None

        # Admin arguments
        self.ini = args.ini

        # Run index arguments
        self.idx = args.idx
        # self.generation_seed = args.generation_seed
        self.trigger_time = args.trigger_time

        # Naming arguments
        self.outdir = args.outdir
        self.label = args.label

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

        if args.timeslide_dict is not None:
            self.timeslide_dict = convert_string_to_dict(args.timeslide_dict)
            logger.info(f"Read-in timeslide dict directly: {self.timeslide_dict}")
        elif args.timeslide_file is not None:
            self.gps_file = args.gps_file
            self.timeslide_file = args.timeslide_file
            self.timeslide_dict = self.get_timeslide_dict(self.idx)

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
        # self.injection_waveform_approximant = args.injection_waveform_approximant
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
            # Added for dingo so that create_data runs. TODO: enable injections
            args.injection = False
            args.injection_numbers = None
            args.injection_file = None
            args.injection_dict = None
            args.injection_waveform_arguments = None
            args.injection_frequency_domain_source_model = None
            self.frequency_domain_source_model = None
            self.gaussian_noise = False

            self.create_data(args)

    def save_hdf5(self):
        """Save frequency-domain strain and ASDs as DingoDataset HDF5 format."""

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
