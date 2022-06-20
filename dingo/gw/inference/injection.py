import numpy as np
from bilby.gw.detector import InterferometerList
from torchvision.transforms import Compose

from dingo.gw.ASD_dataset.noise_dataset import ASDDataset
from dingo.gw.domains import (
    FrequencyDomain,
    build_domain,
    build_domain_from_model_metadata,
)
from dingo.gw.gwutils import get_extrinsic_prior_dict
from dingo.gw.prior import build_prior_with_defaults, split_off_extrinsic_parameters
from dingo.gw.transforms import (
    GetDetectorTimes,
    ProjectOntoDetectors,
    WhitenAndScaleStrain,
)
from dingo.gw.waveform_generator.waveform_generator import (
    WaveformGenerator,
    sum_fd_mode_contributions,
    sum_over_l,
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
        wfg_domain: FrequencyDomain,
        data_domain: FrequencyDomain,
        ifo_list: list,
        t_ref: float,
    ):
        """
        Parameters
        ----------
        wfg_kwargs : dict
            Waveform generator parameters [approximant, f_ref, and (optionally) f_start].
        wfg_domain : FrequencyDomain
            Domain used for waveform generation. This can potentially deviate from the
            final domain, having a wider frequency range needed for waveform generation.
        data_domain : FrequencyDomain
            Domain object for final signal.
        ifo_list : list
            Names of interferometers for projection.
        t_ref : float
            Reference time that specifies ifo locations.
        """

        self._check_domains(wfg_domain, data_domain)
        self.data_domain = data_domain

        # The waveform generator potentially has a larger frequency range than the
        # domain of the trained network / requested injection / etc. This is typically
        # the case for EOB waveforms, which require the larger range to generate
        # robustly. For this reason we have two domains.
        self.waveform_generator = WaveformGenerator(domain=wfg_domain, **wfg_kwargs)

        self.t_ref = t_ref
        self.ifo_list = InterferometerList(ifo_list)

        # When we set self.whiten, the projection transforms are automatically prepared.
        self.whiten = False

        self.asd = None

    @staticmethod
    def _check_domains(domain_in, domain_out):
        if domain_in.f_min > domain_out.f_min or domain_in.f_max < domain_out.f_max:
            raise ValueError(
                "Output domain is not contained within WaveformGenerator domain."
            )
        if domain_in.delta_f != domain_out.delta_f:
            raise ValueError("Domains must have same delta_f.")

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

    def _initialize_transform(self):
        transforms = [
            GetDetectorTimes(self.ifo_list, self.t_ref),
            ProjectOntoDetectors(self.ifo_list, self.data_domain, self.t_ref),
        ]
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

    def signal_from_cached_polarizations(self, theta, polarizations_cached=None):
        """
        Compute the GW signal for parameters theta. Same as self.signal method,
        but allows for caching of the polarizations.

        Step 1: Generate polarizations
        Step 2: Project polarizations onto detectors;
                optionally (depending on self.whiten) whiten and scale.

        Parameters
        ----------
        theta: dict
            Signal parameters. Includes intrinsic parameters to be passed to waveform
            generator, and extrinsic parameters for detector projection.
        polarizations_cached: dict = None

        Returns
        -------
        GW_strain: dict
            keys:
                waveform: GW strain signal for each detector.
                extrinsic_parameters: {}
                parameters: waveform parameters
                asd (if set): amplitude spectral density for each detector
        polarizations_cached: dict
            dict with cached polarizations, for next call of this method
        """
        theta_intrinsic, theta_extrinsic = split_off_extrinsic_parameters(theta)
        theta_intrinsic = {k: float(v) for k, v in theta_intrinsic.items()}

        # Step 1: generate polarizations h_plus and h_cross for all modes if they are not cached
        polarizations_modes = polarizations_cached
        if polarizations_modes is None:
            # compute modes with phase=0, phase will be treated as extrinsic parameter
            polarizations_modes = self.waveform_generator.generate_hplus_hcross_modes(
                {**theta_intrinsic, "phase": 0}
            )
            polarizations_modes = {  # truncation, in case wfg has a larger frequency
                # range
                k_mode: {
                    k_pol: self.data_domain.update_data(v_pol)
                    for k_pol, v_pol in v_mode.items()
                }
                for k_mode, v_mode in polarizations_modes.items()
            }

        # Step 2: sum different modes for h_plus and h_cross according to phase parameter
        polarizations = sum_fd_mode_contributions(
            polarizations_modes, delta_phi=theta["phase"]
        )

        # Step 2: project h_plus and h_cross onto detectors
        sample = {
            "parameters": theta_intrinsic,
            "extrinsic_parameters": {
                **theta_extrinsic,
                # Need to subtract phase from psi to account for the effect that phase
                # parameter rotates spin in sx-sy plane, which is not accounted for by
                # the spherical harmonics in sum_polarization_modes.
                "psi": (theta_extrinsic["psi"] + 0.0 * theta["phase"]) % np.pi,
            },
            "waveform": polarizations,
        }

        asd = self.asd
        if asd is not None:
            sample["asds"] = asd

        return self.projection_transforms(sample), polarizations_modes

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

    def signal_modes(self, theta, keep_l=True):
        """
        Compute the GW signal for parameters theta. Same as self.signal method,
        but it returns the individual contributions of the modes instead of the sum.

        Step 1: Generate polarizations
        Step 2: Project polarizations onto detectors;
                optionally (depending on self.whiten) whiten and scale.

        Parameters
        ----------
        theta: dict
            Signal parameters. Includes intrinsic parameters to be passed to waveform
            generator, and extrinsic parameters for detector projection.
        keep_l: bool = True
            If False, then modes with different l but identical |m| will be summed. This
            is useful, since only |m| determines the transformation behaviour under the
            spherical harmonics when changing the phase.

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
        polarizations_modes = self.waveform_generator.generate_hplus_hcross_modes(
            theta_intrinsic
        )
        # truncation, in case wfg has a larger frequency range
        polarizations_modes = {
            k_mode: {
                k_pol: self.data_domain.update_data(v_pol)
                for k_pol, v_pol in v_mode.items()
            }
            for k_mode, v_mode in polarizations_modes.items()
        }
        # sum modes with different l but identical |m| to (-1, |m|)
        if not keep_l:
            polarizations_modes = sum_over_l(polarizations_modes)

        # Step 2: project h_plus and h_cross onto detectors
        sample_out = {}
        for mode, mode_polarizations in polarizations_modes.items():
            sample = {
                "parameters": theta_intrinsic,
                "extrinsic_parameters": theta_extrinsic,
                "waveform": mode_polarizations,
            }
            if self.asd is not None:
                sample["asds"] = self.asd
            sample_out[mode] = self.projection_transforms(sample)

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
    def from_posterior_model(cls, pm):
        """
        Instantiate an Injection based on a posterior model. The prior, waveform
        settings, etc., will all be consistent with what the model was trained with.

        Parameters
        ----------
        pm : PosteriorModel
        """
        metadata = pm.metadata
        intrinsic_prior = metadata["dataset_settings"]["intrinsic_prior"]
        extrinsic_prior = get_extrinsic_prior_dict(
            metadata["train_settings"]["data"]["extrinsic_prior"]
        )
        prior = build_prior_with_defaults({**intrinsic_prior, **extrinsic_prior})

        return cls(
            prior=prior,
            wfg_kwargs=pm.metadata["dataset_settings"]["waveform_generator"],
            wfg_domain=build_domain(pm.metadata["dataset_settings"]["domain"]),
            data_domain=build_domain_from_model_metadata(pm.metadata),
            ifo_list=pm.metadata["train_settings"]["data"]["detectors"],
            t_ref=pm.metadata["train_settings"]["data"]["ref_time"],
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
        asd = self.asd

        if asd is None:
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
        Generate an random injection.

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
        return self.injection(theta)
