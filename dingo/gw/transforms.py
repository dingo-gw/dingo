from typing import Dict, Any, List, Tuple, Union, Set
import numpy as np
import pandas as pd
import torch

from dingo.gw.domains import Domain, UniformFrequencyDomain
from dingo.gw.detector_network import DetectorNetwork, RandomProjectToDetectors
from dingo.gw.noise import AddNoiseAndWhiten, noise_summary_function
from dingo.gw.waveform_generator import WaveformGenerator

"""
Collect transforms which do not naturally belong with other classes,
such as RandomProjectToDetectors and AddNoiseAndWhiten.
"""


class StandardizeParameters:
    """
    Standardize parameters according to the transform (x - mu) / std.
    """
    def __init__(self, mu: Dict[str, float], std: Dict[str, float]):
        """
        Initialize the standardization transform with means
        and standard deviations for each parameter

        Parameters
        ----------
        mu : Dict[str, float]
            The (estimated) means
        std : Dict[str, float]
            The (estimated) standard deviations
        """
        self.mu = mu
        self.std = std
        if not set(mu.keys()) == set(std.keys()):
            raise ValueError('The keys in mu and std disagree:'
                             f'mu: {mu.keys()}, std: {std.keys()}')

    def __call__(self, samples: Dict[str, Dict[str, Union[float, np.ndarray]]]) \
            -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """Standardize the parameter array according to the
        specified means and standard deviations.

        Parameters
        ----------
        samples: Dict[Dict, Dict]
            A nested dictionary with keys 'parameters', 'waveform',
            'noise_summary'.

        Only parameters included in mu, std get transformed.
        """
        x = samples['parameters']
        y = {k: (x[k] - self.mu[k]) / self.std[k] for k in self.mu.keys()}
        samples_out = samples.copy()
        samples_out['parameters'] = y
        return samples_out

    def inverse(self, samples: Dict[str, Dict[str, Union[float, np.ndarray]]]) \
            -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """De-standardize the parameter array according to the
        specified means and standard deviations.

        Parameters
        ----------
        samples: Dict[Dict, Dict]
            A nested dictionary with keys 'parameters', 'waveform',
            'noise_summary'.

        Only parameters included in mu, std get transformed.
        """
        y = samples['parameters']
        x = {k: self.mu[k] + y[k] * self.std[k] for k in self.mu.keys()}
        samples_out = samples.copy()
        samples_out['parameters'] = x
        return samples_out


class ToNetworkInput:
    """
    Format data for neural network.

    Restrict waveform data to its support (trim off zeros).
    Convert data to torch tensors which can be passed to a NN.
    """
    def __init__(self, domain: Domain):
        """

        Parameters
        ----------
        domain : Domain
            The physical domain on which strains and ASDs are defined.
        """
        self.domain = domain

    def _check_data(self, waveform_dict: Dict[str, Dict[str, np.ndarray]]):
        """
        Check consistency between waveform and ASD data.
        """
        strain_keys = waveform_dict['waveform'].keys()
        noise_summary_keys = waveform_dict['noise_summary'].keys()
        if set(strain_keys) != set(noise_summary_keys):
            raise ValueError('Strains and noise summary must have the same interferometer keys.'
                             f'But got strain: {strain_keys}, asd: {noise_summary_keys}')

        k = list(strain_keys)[0]
        strain_shape = waveform_dict['waveform'][k].shape
        noise_summary_shape = waveform_dict['noise_summary'][k].shape
        if not (strain_shape == noise_summary_shape):
            raise ValueError('Shape of strain and ASD arrays must be the same.'
                             f'But got strain: {strain_shape}, ASD: {noise_summary_shape}')

    def get_output_dimensions(self, waveform_dict: Dict[str, Dict[str, np.ndarray]]) \
            -> Tuple[Tuple, Tuple]:
        """
        Return size of output tensors given input data.

        Parameters
        ----------
        waveform_dict :
            Nested data dictionary with keys 'parameters',
            'waveform', and 'asd' at top level.
        """
        x = pd.DataFrame(waveform_dict['parameters'], index=[0])
        x_shape = x.to_numpy().shape

        n_freq_bins = self.domain.frequency_mask_length
        self._check_data(waveform_dict)
        strain_keys = waveform_dict['waveform'].keys()
        n_ifos = len(strain_keys)
        y_shape = (n_ifos, 3, n_freq_bins)
        return x_shape, y_shape

    def __call__(self, waveform_dict: Dict[str, Dict[str, np.ndarray]]) \
            -> Tuple[torch.tensor, torch.tensor]:
        """
        Transform nested data dictionary into torch tensors.

        Parameters
        ----------
        waveform_dict :
            Nested data dictionary with keys 'parameters',
            'waveform', and 'noise_summary' at top level.
        """
        self._check_data(waveform_dict)
        domain = self.domain

        # 1. Convert binary parameters
        x = pd.DataFrame(waveform_dict['parameters'], index=[0])
        x = x.to_numpy()

        # 2. Repackage detector waveform strains and noise info for network
        if domain.domain_type == 'uFD':
            mask = domain.frequency_mask
            strains = waveform_dict['waveform']
            noise = waveform_dict['noise_summary']
            y = np.array([np.vstack([h[mask].real, h[mask].imag, noise[ifo][mask]])
                          for ifo, h in strains.items()])
        else:
            raise ValueError('Unsupported domain type', domain.domain_type)

        # Sanity check output shapes
        x_shape, y_shape = self.get_output_dimensions(waveform_dict)
        assert (x.shape == x_shape) and (y.shape == y_shape)

        return torch.from_numpy(x), torch.from_numpy(y)


class Compose:
    """Compose several transforms together.

    E.g. for y = f( g( h(x) ) ), defines a transform T(x) := f( g( h(x) ) )
    and its inverse T^{-1}(y) = h^{-1}( g^{-1}( f^{-1}(y) ) ) if it exists.

    A transforms implements __call__ and consumes a particular data object.
    (See torchvision.transforms.)
    """
    def __init__(self, transforms: List):
        """
        Parameters
        ----------
        transforms: List
            A list of transforms which implement the __call__ method.
        """
        self.transforms = transforms

    def __call__(self, data: Any):
        for tr in self.transforms:
            data = tr(data)
        return data

    def inverse(self, data: Any):
        for tr in self.transforms[::-1]:
            if not callable(getattr(tr, 'inverse', None)):
                raise AttributeError(f'Transformation {tr} does not implement an inverse.')
            data = tr.inverse(data)
        return data


class WaveformTransformationTraining:
    """
    Generate a standard chain of transformations from keyword arguments
    to generate input data for training a neural network.

    It consists of the following steps:
    1. Sample in extrinsic parameters and project waveform polarizations
       onto the detector network to compute the GW strain.
    2. Whiten strain data and add zero-mean, white Gaussian noise.
    3. Standardize waveform parameters using reference means and
       standard deviations.
    4. Convert parameters, strain and noise summary information to
       torch tensors.

    This is an alternative to explicitly deserializing the chain of
    transform classes from a pickle file. We allow some of the transformation
    classes and classes the transformation classes depend on, to be specified,
    along with their arguments.
    """
    def __init__(self, *,
                 domain_class: str = 'UniformFrequencyDomain',
                 domain_kwargs: Dict[str, float],
                 prior_class: str = 'GWPriorDict',
                 prior_kwargs: Dict[str, float],
                 detector_network_class: str = 'DetectorNetwork',
                 ifo_list: List):
        """
        Some transformation classes and classes which they depend on are
        specified as strings. They need to be imported into this module
        in order to retrieve their class references.

        Arguments that go with specified classes are in some cases to be
        given as dictionaries so that they can be passed as kwargs into
        the respective class constructor. In other cases, some arguments
        only become available when other classes are instantiated here.

        Parameters
        ----------
        domain_class: str
            Name of the domain class to be used
        domain_kwargs: dict
            kwargs for instantiating the domain class
        prior_class: str
            Name of the prior class to be used
        prior_kwargs: dict
            kwargs for instantiating the prior class
        detector_network_class: str
            Name of the detector network class to be used
        ifo_list: List
            List of interferometer names
        """
        # Assume that classes for all options we want to support are imported into this module,
        self.domain = globals()[domain_class](**domain_kwargs)
        self.priors = globals()[prior_class](**prior_kwargs)
        det_network_kwargs = {'ifo_list': ifo_list, 'domain': self.domain,
                              'start_time': self.priors.reference_geocentric_time}
        self.det_network = globals()[detector_network_class](**det_network_kwargs)

        # Initialize prior means and standard deviations
        # Will contain valid values for simple analytical distributions
        # and NaNs for complicated ones.
        self.mu_dict = {k: v.mean() for k, v in self.priors.items()}
        self.std_dict = {k: v.std() for k, v in self.priors.items()}

    def set_prior_means_stdevs(self, mu_dict: Dict[str, float], std_dict: Dict[str, float]):
        """
        Set the means and standard deviations of the prior distributions.

        Analytical means and standard deviations have been set for simple
        priors in the constructor, while for complicated priors they have
        been set to NaN and need to be overridden.
        THis can be done either by calling this method or by calling
        `set_standardization_values_from_samples()` to calculate
        the sample mean and standard deviations from prior draws.

        Parameters
        ----------
        mu_dict: dict
            Dictionary of parameter means used along with a
            waveform data set
        std_dict: dict
            Dictionary of parameter standard deviations used
            along with a waveform data set
        """
        self.mu_dict.update(mu_dict)
        self.std_dict.update(std_dict)

    def get_prior_means_stdevs(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Return the means and standard deviations of the prior distributions.
        """
        return self.mu_dict, self.std_dict

    def check_standardization_values(self) -> Set:
        """
        Return set of parameters for which either the means or stdevs are NaN.
        """
        nan_means = {k for k, v in self.mu_dict.items() if np.isnan(v)}
        nan_stds = {k for k, v in self.std_dict.items() if np.isnan(v)}
        return nan_means | nan_stds

    def set_standardization_values_from_samples(self, n_samples: int):
        """
        Helper method to set reference standardization values from samples.
        Purely draw parameter samples without generating any waveforms.

        Note: This could be a function, just pass in the priors instance

        Parameters
        ----------
        n_samples:
            Number of samples to draw
        """
        # Draw prior samples and fill in non-sampling parameters
        par_dict = self.priors.sample(size=n_samples)
        par_dict = self.priors.default_conversion_function(par_dict)
        par_dict = {k: par_dict[k] for k in self.check_standardization_values()}

        # Calculate and set sample means and standard deviations
        mu_dict = {k: np.mean(v) for k, v in par_dict.items()}
        std_dict = {k: np.std(v) for k, v in par_dict.items()}
        print(f'Calculated standardization values from {n_samples} samples:'
              f'\nmeans: {mu_dict}\nstdev: {std_dict}')
        print('Updating standardization dictionaries.')
        self.set_prior_means_stdevs(mu_dict, std_dict)

    def __call__(self):
        """
        Return chain of waveform transforms
        """
        # Note:
        # * This does not need to be __call__, it could also be some named method
        #  or there could be a different method for each type of transform
        # * The transformation classes could be specified via arguments
        #   if we find that this feature is needed. I.e. could pass a dict of transform
        #   class names and kwargs to the constructor and then create the
        #   transformation object here
        # * Could also output all transforms we might need
        nan_pars = self.check_standardization_values()
        if len(nan_pars) > 0:
            raise ValueError(f'Please set proper reference values for {nan_pars} which'
                             f' are NaN. \nmeans: {self.mu_dict} \nstdevs: {self.std_dict}')

        return Compose([
            RandomProjectToDetectors(self.det_network, self.priors),
            AddNoiseAndWhiten(self.det_network),
            StandardizeParameters(mu=self.mu_dict, std=self.std_dict),
            ToNetworkInput(self.domain)
        ])

    def get_parameter_list(self, waveform_generator: WaveformGenerator) -> List:
        """
        List of parameter labels which is stripped from the parameter
        dictionary in ToNetworkInput().
        """
        rp_det = RandomProjectToDetectors(self.det_network, self.priors)
        wd = WaveformDataset(prior=F.priors, waveform_generator=waveform_generator,
                             transform=rp_det)
        # Generate intrinsic parameters and waveform polarizations
        wd.generate_dataset(size=1)
        # Sample extrinsic parameters (and compute strain)
        return list(wd[0]['parameters'].keys())


class GenerateObservationTransform:
    """
    Undo whitening and convert torch tensors to dicts of arrays
    """
    def __call__(self, x: torch.tensor, y: torch.tensor,
                 asd_dict: Dict[str, np.ndarray]):
        """
        Parameters
        ----------
        x: torch.tensor
            Parameter samples
        y: torch.tensor
            Real and imaginary strains and noise summary
            for each detector. Shape: (n_detectors, 3, n_bins)
        asd_dict: dict
            A dictionary of ASDs for all detectors in the network.
        """
        # Note:
        # We discard the normalized noise information generated by
        # AddNoiseAndWhiten._noise_summary_function()
        # We could use it if we stored the discarded scale somewhere
        # or if _noise_summary_function() was just 1/asd
        data = y.numpy()
        strains = data[:, 0] + 1j * data[:, 1]
        strain_dict = {k: h * asd_dict[k] for k, h in zip(asd_dict.keys(), strains)}
        return {'strains': strain_dict, 'asds': asd_dict}


class InferenceTransform:
    """
    Observation ---> Network Input

    Prepare observed data for neural network:
    1. Whiten strains
    2. Compute noise summary data from ASDs
    3. Transform to a single torch tensor
    """
    def __call__(self, obs_dict: Dict[Dict[str, np.ndarray], Dict[str, np.ndarray]]) -> torch.tensor:
        """
        Convert strain and ASD data to neural network input.

        Parameters
        ----------
        obs_dict: dict
            Nested dictionary containing 'strains' and 'asds'.
        """
        strains = obs_dict['strains']
        asds = obs_dict['asds']
        strains_white = {ifo: h / asds[ifo] for ifo, h in strains.items()}
        noise_summary = {ifo: noise_summary_function(asd)
                         for ifo, asd in asds.items()}
        y = np.array([np.vstack([h.real, h.imag, noise_summary[ifo]])
                      for ifo, h in strains_white.items()])
        return torch.from_numpy(y)


class PostprocessParameters:
    """
    Convert raw posterior samples to userfriendly format

    1. Undo standardization
    2. Convert to a dict of arrays?
    TODO:
    """
    def __init__(self, mu_dict: Dict[str, float], std_dict: Dict[str, float]):
        pass

    def __call__(self, parameter_samples: torch.tensor, parameter_names: List) -> np.ndarray:
        pass


if __name__ == "__main__":
    """
    Example for setting up a WaveformTransformationTraining and 
    using it with a WaveformDataset and WaveformGenerator.
    """
    from dingo.gw.waveform_generator import WaveformGenerator
    from dingo.gw.waveform_dataset import WaveformDataset
    from dingo.gw.parameters import generate_default_prior_dictionary

    domain_kwargs = {'f_min': 20.0, 'f_max': 4096.0, 'delta_f': 1.0 / 4.0, 'window_factor': 1.0}
    parameter_prior_dict = generate_default_prior_dictionary()
    prior_kwargs = {'parameter_prior_dict': parameter_prior_dict, 'geocent_time_ref': 1126259642.413,
                    'luminosity_distance_ref': 500.0, 'reference_frequency': 20.0}
    ifo_list = ["H1", "L1"]

    F = WaveformTransformationTraining(
        domain_class='UniformFrequencyDomain', domain_kwargs=domain_kwargs,
        prior_class='GWPriorDict', prior_kwargs=prior_kwargs,
        detector_network_class='DetectorNetwork', ifo_list=ifo_list
    )
    # For mass_1, mass_2, luminosity_distance:
    F.set_standardization_values_from_samples(n_samples=10000)

    approximant = 'IMRPhenomXPHM'
    waveform_generator = WaveformGenerator(approximant, F.domain)
    wd = WaveformDataset(prior=F.priors, waveform_generator=waveform_generator, transform=F())
    print('F.priors', F.priors)
    n_waveforms = 17
    wd.generate_dataset(size=n_waveforms)
    print(wd[9])

    # parameter labels for "x" tensor data
    print(F.get_parameter_list(waveform_generator))


    # Test observation transform
    import matplotlib.pyplot as plt
    O = GenerateObservationTransform()
    # Grab some ASDs
    psd_dict = F.det_network.power_spectral_densities
    asd_dict = {ifo: np.sqrt(psd)[F.domain.frequency_mask] for ifo, psd in psd_dict.items()}
    obs_data = O(*wd[9], asd_dict)
    # for ifo, h in obs_data['strains'].items():
    #     plt.loglog(np.abs(h), label=ifo)
    # plt.legend()
    # plt.show()

    # Test inference transform
    print(InferenceTransform()(obs_data))
