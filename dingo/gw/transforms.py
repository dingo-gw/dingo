from typing import Dict, Any, List, Tuple, Union
import numpy as np
import pandas as pd
import torch

from dingo.gw.domains import Domain

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
