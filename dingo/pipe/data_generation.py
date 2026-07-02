import os
import sys

from bilby_pipe.input import Input
from bilby_pipe.data_generation import DataGenerationInput as BilbyDataGenerationInput
from bilby_pipe.utils import (
    parse_args,
    logger,
    convert_string_to_dict,
    resolve_filename_with_transfer_fallback,
)
import lalsimulation as LS
import numpy as np

from dingo.gw.data.event_dataset import EventDataset
from dingo.gw.domains import UniformFrequencyDomain, build_domain_from_model_metadata
from dingo.gw.injection import Injection
from dingo.pipe.parser import create_parser
from dingo.core.utils.backward_compatibility import torch_load_with_fallback

logger.name = "dingo_pipe"


class DataGenerationInput(Input):

    def __init__(self, args, unknown_args, create_data=True):
        super().__init__(args, unknown_args)
        self.meta_data = dict(
            command_line_args=args.__dict__,
            unknown_command_line_args=unknown_args,
            injection_parameters=None,
        )

        self.model = resolve_filename_with_transfer_fallback(args.model) or args.model
        self.model_init = args.model_init and (
            resolve_filename_with_transfer_fallback(args.model_init) or args.model_init
        )
        self.trigger_time = args.trigger_time

        model, _ = torch_load_with_fallback(self.model, preferred_map_location="meta")
        self.model_metadata = model["metadata"]

        # ###############################
        # TODO: AUTOCOMPLETE FROM MODEL #
        # ###############################
        args.detectors = self.model_metadata["train_settings"]["data"]["detectors"]
        self.domain = build_domain_from_model_metadata(self.model_metadata, base=True)
        assert isinstance(self.domain, UniformFrequencyDomain)
        args.duration = int(np.round(1 / self.domain.delta_f))

        # Whether to generate data for importance sampling. This must be done when
        # desired data settings differ from those used for network training. If this is
        # the case, save the new data to a different file name.
        self.importance_sampling = args.importance_sampling_generation
        self.importance_sampling_updates = args.importance_sampling_updates
        if self.importance_sampling:
            # Updates to frequency range should not affect the data generation for importance sampling
            if "minimum_frequency" in self.importance_sampling_updates:
                self.importance_sampling_updates.pop("minimum_frequency")
            if "maximum_frequency" in self.importance_sampling_updates:
                self.importance_sampling_updates.pop("maximum_frequency")
            vars(args).update(self.importance_sampling_updates)

        self.save_bilby_data_dump = args.save_bilby_data_dump

        # Build the base domain from model metadata to ensure bilby generates
        # data with the full available frequency range. This way it won't set zeros
        # outside the range, and DINGO can mask it later as needed.
        self.bilby_generation_input = BilbyDataGenerationInput(
            args, unknown_args, create_data=False
        )
        self.bilby_generation_input.minimum_frequency = self.domain.f_min
        self.bilby_generation_input.maximum_frequency = self.domain.f_max
        self.bilby_generation_input.sampling_frequency = (
            self.lowest_sampling_frequency_from_domain()
        )

        if create_data:
            if self.dingo_injection:
                self.create_data_dingo_injection(args)
            else:
                self.bilby_generation_input.create_data(args)

    def __getattr__(self, name):
        try:
            return getattr(self.bilby_generation_input, name)
        except AttributeError:
            pass
        return None

    def create_data_dingo_injection(self, args):
        """Adaptation of create_data to use Dingo signal models rather than Bilby.

        First, executes create_data but without any requested injections. This creates
        a  noise-only dataset.

        Second, calls _inject_dingo_signal to generate the Dingo signal waveform and
        add it to the noisy data within the interferometers.
        """
        self.injection = args.injection
        self.injection_file = args.injection_file
        self.injection_dict = args.injection_dict

        args.injection = False
        args.injection_file = None
        args.injection_dict = None

        # Create noise with bilby_pipe and inject with DINGO
        self.bilby_generation_input.create_data(args)
        if self.injection:
            self._inject_dingo_signal()

    def _inject_dingo_signal(self):
        """Generate a GW signal using the dingo.gw.injection class and add it to the
        interferometer strain data. Also compute SNRs and store them."""
        injection = Injection.from_posterior_model_metadata(self.model_metadata)
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

            # These arrays extend up to self.sampling_frequency / 2. Truncate them to
            # the base domain maximum frequency, and set the ASD to 1.0 below base domain
            # minimum frequency (which should already be the case from bilby generation).
            strain = self.domain.update_data(strain)
            asd = self.domain.update_data(asd, low_value=1.0)

            # Dingo expects data to have trigger time 0, so we apply a cyclic time shift
            # by the post-trigger duration.
            strain = self.domain.time_translate_data(strain, self.post_trigger_duration)

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
            "window_type": "tukey",
            "roll_off": self.tukey_roll_off,
            "minimum_frequency": self.minimum_frequency_dict,
            "maximum_frequency": self.maximum_frequency_dict,
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
                k: (
                    v["optimal_SNR"].item()
                    if hasattr(v["optimal_SNR"], "item")
                    else v["optimal_SNR"]
                )
                for k, v in self.interferometers.meta_data.items()
            }
            settings["matched_filter_SNR"] = {
                k: (
                    v["matched_filter_SNR"].item()
                    if hasattr(v["matched_filter_SNR"], "item")
                    else v["matched_filter_SNR"]
                )
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
                np.vstack([self.domain(), data["asds"][ifo.name] ** 2]).T,
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

    def lowest_sampling_frequency_from_domain(self):
        """Set the sampling frequency as the lowest power of two greater than
        self.domain.f_max that, when divided by two, is greater or equal to
        self.domain.f_max."""
        target = 2 * self.domain.f_max
        return 2 ** np.ceil(np.log2(target))


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
