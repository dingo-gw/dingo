from typing import Dict, Any, List, Tuple, Union
import numpy as np
import pandas as pd
import torch

from dingo.gw.domains import Domain, UniformFrequencyDomain
from dingo.gw.detector_network import DetectorNetwork, RandomProjectToDetectors
from dingo.gw.noise import AddNoiseAndWhiten
from dingo.gw.parameters import GWPriorDict

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
            A nested dictionary with keys 'parameters', 'waveform'.

        Only parameters included in mu, std get transformed.
        """
        x = samples['parameters']
        print('d_L in', x['luminosity_distance'])
        y = {k: (x[k] - self.mu[k]) / self.std[k] for k in self.mu.keys()}
        print('d_L tr', y['luminosity_distance'])
        # samples['parameters'] = y
        # return samples
        return {'parameters': y, 'waveform': samples['waveform'], 'asd': samples['asd']}

    def inverse(self, samples: Dict[str, Dict[str, Union[float, np.ndarray]]]) \
            -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """De-standardize the parameter array according to the
        specified means and standard deviations.

        Parameters
        ----------
        samples: Dict[Dict, Dict]
            A nested dictionary with keys 'parameters', 'waveform'.

        Only parameters included in mu, std get transformed.
        """
        y = samples['parameters']
        print('d_L inv', y['luminosity_distance'])
        x = {k: self.mu[k] + y[k] * self.std[k] for k in self.mu.keys()}
        print('d_L back', x['luminosity_distance'])
        return {'parameters': x, 'waveform': samples['waveform'], 'asd': samples['asd']}


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
        asd_keys = waveform_dict['asd'].keys()
        if set(strain_keys) != set(asd_keys):
            raise ValueError('Strains and ASDs must have the same interferometer keys.'
                             f'But got strain: {strain_keys}, asd: {asd_keys}')

        k = list(strain_keys)[0]
        strain_shape = waveform_dict['waveform'][k].shape
        asd_shape = waveform_dict['asd'][k].shape
        if not (strain_shape == asd_shape):
            raise ValueError('Shape of strain and ASD arrays must be the same.'
                             f'But got strain: {strain_shape}, ASD: {asd_shape}')

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
            'waveform', and 'asd' at top level.
        """
        self._check_data(waveform_dict)
        domain = self.domain

        # 1. Convert binary parameters
        x = pd.DataFrame(waveform_dict['parameters'], index=[0])
        x = x.to_numpy()

        # 2. Repackage detector waveform strains and ASDs for entire network
        if domain.domain_type == 'uFD':
            mask = domain.frequency_mask
            strains = waveform_dict['waveform']
            asds = waveform_dict['asd']
            y = np.array([np.vstack([h[mask].real, h[mask].imag, asds[ifo][mask]])
                          for ifo, h in strains.items()])

            # y = np.zeros((n_ifos, 3, n_freq_bins))
            # # y = np.empty((n_ifos, 3, n_freq_bins)) # not so safe, but perhaps a little bit faster
            # for ind, (ifo, d) in enumerate(waveform_dict['waveform'].items()):
            #     d = waveform_dict['waveform'][ifo][mask]
            #     asd = waveform_dict['asd'][ifo][mask]
            #     y[ind, 0, :] = d.real
            #     y[ind, 1, :] = d.imag
            #     y[ind, 2, :] = asd

            # TODO: move this to a unit test
            x_shape, y_shape = self.get_output_dimensions(waveform_dict)
            assert (x.shape == x_shape) and (y.shape == y_shape)
        else:
            raise ValueError('Unsupported domain type', domain.domain_type)

        # FIXME: how will the NN know which entries are which parameters and which rows are which detectors?
        #  Must return this additional label data for later
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


class WaveformTransformationFactory:
    """
    Generate a standard chain of transformations from keyword arguments.

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


    def __call__(self, mu_dict: Dict[str, float], std_dict: Dict[str, float]):
        """
        Return chain of waveform transforms

        Parameters
        ----------
        mu_dict: dict
            Dictionary of parameter means used along with a
            waveform data set
        std_dict: dict
            Dictionary of parameter standard deviations used
            along with a waveform data set

        TODO: The data from which these dicts can be computed only becomes
          available after all waveform parameters have been sampled,
          i.e. for the extrinsic parameters.
          We could instead set them to some reference values.
          Alternatively, we could use the analytical means and stdevs
          for each of the parameter distributions; these would naturally
          belong in subclasses of the bilby priors.
        """
        # The transformation classes could be specified via arguments
        # if we find that this feature is needed.
        return Compose([
            RandomProjectToDetectors(self.det_network, self.priors),
            AddNoiseAndWhiten(self.det_network),
            StandardizeParameters(mu=mu_dict, std=std_dict),
            ToNetworkInput(self.domain)
        ])




if __name__ == "__main__":
    """
    Example for setting up a WaveformTransformationFactory and 
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

    F = WaveformTransformationFactory(
        domain_class='UniformFrequencyDomain', domain_kwargs=domain_kwargs,
        prior_class='GWPriorDict', prior_kwargs=prior_kwargs,
        detector_network_class='DetectorNetwork', ifo_list=ifo_list
    )

    mu_dict = {'phi_jl': 1.0, 'tilt_1': 1.0, 'theta_jn': 2.0, 'tilt_2': 1.0, 'mass_1': 54.0, 'phi_12': 0.5,
               'chirp_mass': 40.0, 'phase': np.pi, 'a_2': 0.5, 'mass_2': 39.0, 'mass_ratio': 0.5,
               'a_1': 0.5, 'f_ref': 20.0, 'luminosity_distance': 1000.0, 'geocent_time': 1126259642.413,
               'ra': 2.5, 'dec': 1.0, 'psi': np.pi}
    std_dict = mu_dict
    wf_transforms = F(mu_dict, std_dict)

    approximant = 'IMRPhenomXPHM'
    waveform_generator = WaveformGenerator(approximant, F.domain)
    wd = WaveformDataset(priors=F.priors, waveform_generator=waveform_generator, transform=wf_transforms)
    print(F.priors)
    n_waveforms = 17
    wd.generate_dataset(size=n_waveforms)
    print(wd[9])

