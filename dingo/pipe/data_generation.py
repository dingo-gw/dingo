import os
import sys

from bilby_pipe.input import Input
from bilby_pipe.data_generation import DataGenerationInput as BilbyDataGenerationInput
from bilby.core.prior import PriorDict
from bilby_pipe.utils import (
    parse_args,
    logger,
    convert_string_to_dict,
    convert_prior_string_input,
    BilbyPipeError,
)
import lalsimulation as LS
import numpy as np

from dingo.core.posterior_models.build_model import build_model_from_kwargs
from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.gwutils import get_extrinsic_prior_dict
from dingo.gw.data.event_dataset import EventDataset
from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.injection import Injection
from dingo.pipe.parser import create_parser
from dingo.pipe.main import fill_in_arguments_from_model

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
        self.transfer_files = args.transfer_files

        # Run index arguments
        self.idx = args.idx
        self.generation_seed = args.generation_seed
        self.trigger_time = args.trigger_time

        # Naming arguments
        self.outdir = args.outdir
        self.label = args.label

        # Prior arguments
        # self.reference_frame = args.reference_frame
        # self.time_reference = "geocent" # DINGO mod used for saving data dump
        # self.phase_marginalization = args.phase_marginalization
        self.prior_dict = args.prior_dict
        self.default_prior = "BBHPriorDict"
        self.time_reference = "geocent"
        self.prior_dict_updates = args.prior_dict_updates

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
        self.gaussian_noise = args.gaussian_noise
        self.zero_noise = args.zero_noise
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
        self.reference_frequency = args.reference_frequency

        # Waveform, source model and likelihood
        self.waveform_generator_class = (
            "bilby.gw.waveform_generator.LALCBCWaveformGenerator"
        )
        self.waveform_approximant = args.waveform_approximant
        self.catch_waveform_errors = args.catch_waveform_errors
        # TODO: These are set to parser defaults. Fix to set from model.
        self.pn_spin_order = -1
        self.pn_tidal_order = -1
        self.pn_phase_order = -1
        self.pn_amplitude_order = 0
        self.mode_array = None
        # don't set self.waveform_arguments_dict, it will be updated later by injection_waveform_arguments
        self.waveform_arguments_dict = None 
        self.injection_waveform_arguments = args.injection_waveform_arguments
        self.numerical_relativity_file = args.numerical_relativity_file
        self.dingo_injection = args.dingo_injection
        self.injection_waveform_approximant = args.injection_waveform_approximant
        if args.injection_waveform_approximant in ["SEOBNRv5PHM", "SEOBNRv5EHM", "SEOBNRv5HM"]:
            self.injection_frequency_domain_source_model = "gwsignal_binary_black_hole"
            self.frequency_domain_source_model = "gwsignal_binary_black_hole"
        else:
            self.injection_frequency_domain_source_model = "lal_binary_black_hole"
            self.frequency_domain_source_model = "lal_binary_black_hole" 

        # DINGO mod
        self.save_bilby_data_dump = args.save_bilby_data_dump
        if self.save_bilby_data_dump:
            self.time_reference = args.time_reference
            self.deltaT = args.deltaT

        # self.conversion_function = args.conversion_function
        # self.generation_function = args.generation_function
        # self.likelihood_type = args.likelihood_type
        # self.extra_likelihood_kwargs = args.extra_likelihood_kwargs
        self.enforce_signal_duration = args.enforce_signal_duration

        # PSD
        self.psd_maximum_duration = args.psd_maximum_duration
        self.psd_dict = args.psd_dict
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
        self.plot_injection = args.plot_injection

        if create_data:
            if self.dingo_injection:
                self.create_data_dingo_injection(args)
            else:
                self.create_data(args)

    def create_data_dingo_injection(self, args):
        """Adaptation of create_data to use Dingo signal models rather than Bilby.

        First, executes create_data but without any requested injections. This creates
        a  noise-only dataset.

        Second, calls _inject_dingo_signal to generate the Dingo signal waveform and
        add it to the noisy data within the interferometers.
        """
        # Save values of relevant args.
        injection = args.injection
        injection_file = args.injection_file
        injection_dict = args.injection_dict

        args.injection = False
        args.injection_file = None
        args.injection_dict = None

        # Create noise.
        self.create_data(args)

        # Reset args.
        args.injection = injection
        args.injection_file = injection_file
        args.injection_dict = injection_dict
        self.injection = args.injection
        self.injection_file = args.injection_file
        self.injection_dict = args.injection_dict

        if self.injection:
            self._inject_dingo_signal(args)

    def _inject_dingo_signal(self, args):
        """Generate a GW signal using the dingo.gw.injection class and add it to the
        interferometer strain data. Also compute SNRs and store them."""
        try:
            model = build_model_from_kwargs(
                filename=args.model, device="meta", load_training_info=False
            )
        except RuntimeError:
            # 'meta' is not supported by older version of python / torch
            model = build_model_from_kwargs(
                filename=args.model, device="cpu", load_training_info=False
            )

        injection = Injection.from_posterior_model_metadata(model.metadata)
        injection.use_base_domain = True  # Do not generate MFD signals.
        injection.t_ref = self.trigger_time
        injection._initialize_transform()

        # Possibly update waveform generator based on supplied settings. Note that
        # default values will have been set by dingo_pipe from the DINGO model.
        waveform_arguments = self.get_injection_waveform_arguments()
        injection.f_ref = waveform_arguments["reference_frequency"]
        injection.waveform_generator.approximant = LS.GetApproximantFromString(
            waveform_arguments["waveform_approximant"]
        )
        injection.waveform_generator.approximant_str = waveform_arguments[
            "waveform_approximant"
        ]

        logger.info("Injecting waveform from DINGO with ")
        logger.info(f"data_domain = {injection.data_domain.domain_dict}")
        for prop in [
            "t_ref",
        ]:
            logger.info(f"{prop} = {getattr(injection, prop)}")
        for prop in [
            "approximant_str",
            "f_ref",
            "f_start",
        ]:
            logger.info(f"{prop} = {getattr(injection.waveform_generator, prop)}")

        # Generate signal
        self.injection_parameters = self.injection_df.iloc[self.idx].to_dict()
        theta = self.injection_parameters.copy()
        theta["geocent_time"] -= self.trigger_time
        signal = injection.signal(theta)

        # Add signal to interferometer data
        domain = injection.data_domain
        for ifo in self.interferometers:
            s = signal["waveform"][ifo.name]
            s = domain.time_translate_data(s, -self.post_trigger_duration)
            # Interferometer data extends to Nyquist = f_max / 2, so pad signal. Will
            # be truncated when saved as HDF5.
            s = np.pad(
                s, (0, len(ifo.strain_data.frequency_domain_strain) - len(domain))
            )
            ifo.strain_data.frequency_domain_strain += s
            ifo.meta_data["optimal_SNR"] = np.sqrt(
                ifo.optimal_snr_squared(signal=s).real
            )
            ifo.meta_data["matched_filter_SNR"] = ifo.matched_filter_snr(signal=s)

    def save_hdf5(self):
        """
        Save frequency-domain strain and ASDs as DingoDataset HDF5 format.

        This method will also save the PSDs as .txt files in the data directory
        for easy reading by pesummary and Bilby.
        """

        if self.save_bilby_data_dump:
            # this is needed because we want bilby to use the updated DINGO 
            # prior
            self.waveform_arguments_dict = self.injection_waveform_arguments
            self.prior_dict = self.priors 
            self.likelihood_type = "GravitationalWaveTransient"
            self.calibration_marginalization = False
            self.phase_marginalization = False
            self.time_marginalization = False
            self.distance_marginalization = False
            self.number_of_response_curves = 0
            self._distance_marginalization_lookup_table = None
            self.reference_frame = "sky"
            self.fiducial_parameters = None
            self.update_fiducial_parameters = None 
            self.epsilon = None 
            self.jitter_time = True
            self.save_data_dump()

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
            domain = UniformFrequencyDomain(
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
            "injection",
            "generation_seed",
            "gaussian_noise",
            "zero_noise",
            "numerical_relativity_file",
            "injection_waveform_approximant",
            "injection_frequency_domain_source_model",
        ]:
            try:
                v = getattr(self, k)
            except AttributeError:
                continue
            if v is not None and v is not False:
                settings[k] = v

        if self.injection:
            settings["injection_parameters"] = self.injection_parameters.copy()
            settings["dingo_injection"] = self.dingo_injection
            # Dingo and Bilby have different geocent_time conventions.
            settings["injection_parameters"]["geocent_time"] -= self.trigger_time
            settings["optimal_SNR"] = {
                k: v["optimal_SNR"] for k, v in self.interferometers.meta_data.items()
            }
            settings["matched_filter_SNR"] = {
                k: v["matched_filter_SNR"]
                for k, v in self.interferometers.meta_data.items()
            }

        dataset = EventDataset(
            dictionary={
                "data": data,
                "settings": settings,
            }
        )
        dataset.to_file(self.event_data_file)

        # also saving the psd as a .txt file which can be read in
        # easily by pesummary or bilby
        for ifo in self.interferometers:
            np.savetxt(
                os.path.join(self.data_directory, f"{ifo.name}_psd.txt"),
                np.vstack([domain(), data["asds"][ifo.name] ** 2]).T,
            )

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

    def _set_interferometers_from_gaussian_noise(self):
        super()._set_interferometers_from_gaussian_noise()
        # Scale the FD strain appropriately by the window factor (not done by default
        # in Bilby / bilby_pipe). This is to ensure that the injection is consistent
        # with TD data with a given PSD, making it consistent also with DINGO network
        # training.
        for ifo in self.interferometers:
            # This is a hack to set the window factor. It ensures also that the SNRs
            # are calculated correctly.
            td_strain = ifo.time_domain_strain
            # TODO: correct for window factor changes https://git.ligo.org/pe/pe-group-coordination/-/issues/1
            ifo.strain_data.time_domain_window(roll_off=self.tukey_roll_off)
            ifo.strain_data.frequency_domain_strain = (
                ifo.strain_data.frequency_domain_strain
                * np.sqrt(ifo.strain_data.window_factor)
            )

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
