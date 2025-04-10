import numpy as np
from bilby.gw.detector import InterferometerList
from torchvision.transforms import Compose

from dingo.gw.noise.asd_dataset import ASDDataset
from dingo.gw.domains import (
    UniformFrequencyDomain,
    MultibandedFrequencyDomain,
)
from dingo.gw.domains import build_domain, build_domain_from_model_metadata
from dingo.gw.gwutils import get_extrinsic_prior_dict
from dingo.gw.prior import build_prior_with_defaults, split_off_extrinsic_parameters
from dingo.gw.transforms import (
    GetDetectorTimes,
    ProjectOntoDetectors,
    WhitenAndScaleStrain,
    ApplyCalibrationUncertainty,
)
from dingo.gw.waveform_generator.waveform_generator import (
    WaveformGenerator,
    NewInterfaceWaveformGenerator,
)


class GWSignal(object):
    """
    Base class for generating gravitational wave signals in interferometers. Generates
    waveform polarizations based on provided parameters, and then projects to detectors.

    Includes option for whitening the signal based on a provided ASD.
    """

    def __init__(
        self,
        wfg_kwargs: dict,
        wfg_domain: UniformFrequencyDomain | MultibandedFrequencyDomain,
        data_domain: UniformFrequencyDomain | MultibandedFrequencyDomain,
        ifo_list: list,
        t_ref: float,
    ):
        """
        Parameters
        ----------
        wfg_kwargs : dict
            Waveform generator parameters [approximant, f_ref, and (optionally) f_start].
        wfg_domain : UniformFrequencyDomain | MultibandedFrequencyDomain
            Domain used for waveform generation. This can potentially deviate from the
            final domain, having a wider frequency range needed for waveform generation.
        data_domain : UniformFrequencyDomain | MultibandedFrequencyDomain
            Domain object for final signal.
        ifo_list : list
            Names of interferometers for projection.
        t_ref : float
            Reference time that specifies ifo locations.
        """
        self._use_base_domain = False
        self._check_domains(wfg_domain, data_domain)
        self.data_domain = data_domain

        # The waveform generator potentially has a larger frequency range than the
        # domain of the trained network / requested injection / etc. This is typically
        # the case for EOB waveforms, which require the larger range to generate
        # robustly. For this reason we have two domains.

        new_interface_flag = wfg_kwargs.get("new_interface", False)
        if new_interface_flag:
            self.waveform_generator = NewInterfaceWaveformGenerator(
                domain=wfg_domain, **wfg_kwargs
            )
        else:
            self.waveform_generator = WaveformGenerator(domain=wfg_domain, **wfg_kwargs)

        self.t_ref = t_ref
        self.ifo_list = InterferometerList(ifo_list)

        # When we set self.whiten, the projection transforms are automatically prepared.
        self._calibration_envelope = None
        self._calibration_marginalization_kwargs = None
        self.whiten = False

        self.asd = None

    @staticmethod
    def _check_domains(
        domain_in: UniformFrequencyDomain | MultibandedFrequencyDomain,
        domain_out: UniformFrequencyDomain | MultibandedFrequencyDomain,
    ):
        if domain_in.f_min > domain_out.f_min or domain_in.f_max < domain_out.f_max:
            raise ValueError(
                "Output domain is not contained within WaveformGenerator domain."
            )
        if (
            domain_in.domain_dict["type"] == "UniformFrequencyDomain"
            and domain_out.domain_dict["type"] == "UniformFrequencyDomain"
        ):
            if domain_in.delta_f != domain_out.delta_f:
                raise ValueError("Domains must have same delta_f.")

    @property
    def use_base_domain(self):
        return self._use_base_domain

    @use_base_domain.setter
    def use_base_domain(self, value: bool):
        if value != self._use_base_domain:
            if value:
                if hasattr(self.data_domain, "base_domain"):
                    self.waveform_generator.domain = (
                        self.waveform_generator.full_domain.base_domain
                    )
                    self.data_domain = self.data_domain.base_domain
                    self._use_base_domain = True
                    self._initialize_transform()
                else:
                    print(
                        f"{type(self.data_domain)} has no base domain. Nothing to do."
                    )
            else:
                raise NotImplementedError(
                    "Cannot recover original domain from base domain alone."
                )

    @property
    def whiten(self):
        """
        Bool specifying whether to whiten (and scale) generated signals.
        """
        return self._whiten

    @whiten.setter
    def whiten(self, value):
        self._whiten = value
        self._initialize_transform()

    @property
    def calibration_marginalization_kwargs(self):
        """
        Dictionary with the following keys:

        calibration_envelope
            Dictionary of the form {"H1": filepath, "L1": filepath, ...} with locations of
            lookup tables for the calibration uncertainty curves.

        num_calibration_nodes
            Number of nodes for the calibration model.

        num_calibration_curves
            Number of calibration curves to use in marginalization.
        """
        return self._calibration_marginalization_kwargs

    @calibration_marginalization_kwargs.setter
    def calibration_marginalization_kwargs(self, value):
        self._calibration_marginalization_kwargs = value
        self._initialize_transform()

    def _initialize_transform(self):
        transforms = [
            GetDetectorTimes(self.ifo_list, self.t_ref),
            ProjectOntoDetectors(self.ifo_list, self.data_domain, self.t_ref),
        ]
        if self.calibration_marginalization_kwargs:
            transforms.append(
                ApplyCalibrationUncertainty(
                    self.ifo_list,
                    self.data_domain,
                    **self.calibration_marginalization_kwargs,
                )
            )
        if self.whiten:
            transforms.append(WhitenAndScaleStrain(self.data_domain.noise_std))
        self.projection_transforms = Compose(transforms)

    def signal(self, theta):
        """
        Compute the GW signal for parameters theta.

        Step 1: Generate polarizations
        Step 2: Project polarizations onto detectors; optionally (depending on
        self.whiten) whiten and scale.

        Parameters
        ----------
        theta: dict
            Signal parameters. Includes intrinsic parameters to be passed to waveform
            generator, and extrinsic parameters for detector projection.

        Returns
        -------
        dict
            keys:
                waveform: GW strain signal for each detector.
                extrinsic_parameters: {}
                parameters: waveform parameters
                asd (if set): amplitude spectral density for each detector
        """
        theta_intrinsic, theta_extrinsic = split_off_extrinsic_parameters(theta)
        theta_intrinsic = {k: float(v) for k, v in theta_intrinsic.items()}

        # Step 1: generate polarizations h_plus and h_cross
        polarizations = self.waveform_generator.generate_hplus_hcross(theta_intrinsic)
        polarizations = {  # truncation, in case wfg has a larger frequency range
            k: self.data_domain.update_data(v) for k, v in polarizations.items()
        }

        # Step 2: project h_plus and h_cross onto detectors
        sample = {
            "parameters": theta_intrinsic,
            "extrinsic_parameters": theta_extrinsic,
            "waveform": polarizations,
        }

        asd = self.asd
        if asd is not None:
            sample["asds"] = asd

        return self.projection_transforms(sample)

    # It would be good to have an ASD class to handle all of this functionality,
    # namely storing ASDs from numpy arrays, from ASDDatasets, loading from files,
    # etc. For now this functionality is partially implemented here.

    def signal_m(self, theta):
        """
        Compute the GW signal for parameters theta. Same as self.signal(theta) method,
        but it does not sum the contributions of the individual modes, and instead
        returns a dict {m: pol_m for m in [-l_max,...,0,...,l_max]} where each
        contribution pol_m transforms as exp(-1j * m * phase_shift) under phase shifts.

        Step 1: Generate polarizations
        Step 2: Project polarizations onto detectors;
                optionally (depending on self.whiten) whiten and scale.

        Parameters
        ----------
        theta: dict
            Signal parameters. Includes intrinsic parameters to be passed to waveform
            generator, and extrinsic parameters for detector projection.

        Returns
        -------
        dict
            keys:
                waveform:
                    GW strain signal for each detector, with individual contributions
                    {m: pol_m for m in [-l_max,...,0,...,l_max]}
                extrinsic_parameters: {}
                parameters: waveform parameters
                asd (if set): amplitude spectral density for each detector
        """
        theta_intrinsic, theta_extrinsic = split_off_extrinsic_parameters(theta)
        theta_intrinsic = {k: float(v) for k, v in theta_intrinsic.items()}

        # Step 1: generate m-contributions to polarizations h_plus and h_cross
        pol_m = self.waveform_generator.generate_hplus_hcross_m(theta_intrinsic)
        # truncation, in case wfg has a larger frequency range
        pol_m = {
            k_m: {
                k_pol: self.data_domain.update_data(v_pol)
                for k_pol, v_pol in v_m.items()
            }
            for k_m, v_m in pol_m.items()
        }

        # Step 2: project m-contributions to h_plus and h_cross onto detectors
        sample_out = {}
        for m, pol in pol_m.items():
            sample = {
                "parameters": theta_intrinsic,
                "extrinsic_parameters": theta_extrinsic,
                "waveform": pol,
            }
            if self.asd is not None:
                sample["asds"] = self.asd
            sample_out[m] = self.projection_transforms(sample)

        return sample_out

    @property
    def asd(self):
        """
        Amplitude spectral density.

        Either a single array, a dict (for individual interferometers),
        or an ASDDataset, from which random ASDs are drawn.
        """
        if isinstance(self._asd, np.ndarray):
            asd = {ifo.name: self._asd for ifo in self.ifo_list}
        elif isinstance(self._asd, dict):
            asd = self._asd
        elif isinstance(self._asd, ASDDataset):
            asd = self._asd.sample_random_asds()
        elif self._asd is None:
            return None
        else:
            raise TypeError("Invalid ASD type.")
        asd = {
            k: self.data_domain.update_data(v, low_value=1e-20) for k, v in asd.items()
        }
        return asd

    @asd.setter
    def asd(self, asd):
        ifo_names = [ifo.name for ifo in self.ifo_list]
        if isinstance(asd, ASDDataset):
            if set(asd.asds.keys()) != set(ifo_names):
                raise KeyError("ASDDataset ifos do not match signal.")
            if asd.domain.domain_dict != self.data_domain.domain_dict:
                print("Updating ASDDataset domain to match data domain.")
                domain_dict = self.data_domain.domain_dict
                if "window_factor" in domain_dict:
                    print("Dropping window factor for update.")
                    del domain_dict["window_factor"]
                if (
                    "base_domain" in domain_dict
                    and "window_factor" in domain_dict["base_domain"]
                ):
                    print("Dropping window factor for update.")
                    del domain_dict["base_domain"]["window_factor"]
                asd.update_domain(domain_dict)
        elif isinstance(asd, dict):
            if set(asd.keys()) != set(ifo_names):
                raise KeyError("ASD ifos do not match signal.")
        elif isinstance(asd, str):
            raise NotImplementedError(
                "Still need to implement injections with ASDs defined by file names."
            )
        self._asd = asd


class Injection(GWSignal):
    """
    Produces injections of signals (with random or specified parameters) into stationary
    Gaussian noise. Output is not whitened.
    """

    def __init__(self, prior, **gwsignal_kwargs):
        """
        Parameters
        ----------
        prior : PriorDict
            Prior used for sampling random parameters.
        gwsignal_kwargs
            Arguments to be passed to GWSignal base class.
        """
        super().__init__(**gwsignal_kwargs)
        self.prior = prior

    @classmethod
    def from_posterior_model_metadata(cls, metadata):
        """
        Instantiate an Injection based on a posterior model. The prior, waveform
        settings, etc., will all be consistent with what the model was trained with.

        Parameters
        ----------
        metadata : dict
            Dict which you can get via PosteriorModel.metadata
        """
        intrinsic_prior = metadata["dataset_settings"]["intrinsic_prior"]
        extrinsic_prior = get_extrinsic_prior_dict(
            metadata["train_settings"]["data"]["extrinsic_prior"]
        )
        prior = build_prior_with_defaults({**intrinsic_prior, **extrinsic_prior})

        return cls(
            prior=prior,
            wfg_kwargs=metadata["dataset_settings"]["waveform_generator"],
            wfg_domain=build_domain(metadata["dataset_settings"]["domain"]),
            data_domain=build_domain_from_model_metadata(metadata),
            ifo_list=metadata["train_settings"]["data"]["detectors"],
            t_ref=metadata["train_settings"]["data"]["ref_time"],
        )

    def injection(self, theta):
        """
        Generate an injection based on specified parameters.

        This is a signal + noise  consistent with the amplitude spectral density in
        self.asd. If self.asd is an ASDDataset, then it uses a random ASD from this
        dataset.

        Data are not whitened.

        Parameters
        ----------
        theta : dict
            Parameters used for injection.

        Returns
        -------
        dict
            keys:
                waveform: data (signal + noise) in each detector
                extrinsic_parameters: {}
                parameters: waveform parameters
                asd (if set): amplitude spectral density for each detector
        """
        signal = self.signal(theta)
        try:
            # Be careful to use the ASD included with the signal, since each time
            # self.asd is accessed it gives a different ASD (if using an ASD dataset).
            asd = signal["asds"]
        except KeyError:
            raise ValueError("self.asd must be set in order to produce injections.")

        if self.whiten:
            print("self.whiten was set to True. Resetting to False.")
            self.whiten = False

        data = {}
        for ifo, s in signal["waveform"].items():
            noise = (
                (np.random.randn(len(s)) + 1j * np.random.randn(len(s)))
                * asd[ifo]
                * self.data_domain.noise_std
            )
            d = s + noise
            data[ifo] = self.data_domain.update_data(d, low_value=0.0)

        signal["waveform"] = data
        return signal

    def random_injection(self):
        """
        Generate a random injection.

        This is a signal + noise  consistent with the amplitude spectral density in
        self.asd. If self.asd is an ASDDataset, then it uses a random ASD from this
        dataset.

        Data are not whitened.

        Returns
        -------
        dict
            keys:
                waveform: data (signal + noise) in each detector
                extrinsic_parameters: {}
                parameters: waveform parameters
                asd (if set): amplitude spectral density for each detector
        """
        theta = self.prior.sample()
        theta = {
            k: float(v) for k, v in theta.items()
        }  # Some parameters are np.float64
        return self.injection(theta)
