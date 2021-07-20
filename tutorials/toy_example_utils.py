import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from typing import Union


class HarmonicOscillator:
    """
    Simulator for the harmonic oscillator forward model. It is initialised
    with an axis for the time samples T where the oscillator is evaluated.
    """

    def __init__(self, t_lower=0, t_upper=10, N_bins=2000):
        # Time axis
        self.t_upper = t_upper
        self.t_lower = t_lower
        self.N_bins = N_bins
        self.t_axis = np.linspace(self.t_lower, self.t_upper, self.N_bins)
        self.delta_t = (self.t_upper - self.t_lower) / (self.N_bins - 1)

    def simulate(self, parameters):
        if parameters.shape[1] == 3:
            t, omega0, beta = parameters[:, 0], parameters[:, 1], \
                              parameters[:, 2]
        elif parameters.shape[1] == 2:
            omega0, beta = parameters[:, 0], parameters[:, 1]
            t = np.zeros_like(omega0)
        else:
            raise ValueError('Unexpected number of parameters')
        A = np.ones(len(t))
        a = A[:, np.newaxis] \
            * np.exp(-np.outer(beta * omega0, self.t_axis) \
                     + (beta * omega0 * t)[:, np.newaxis])
        b = np.outer(np.sqrt(1 - beta ** 2) * omega0, self.t_axis) \
            - (np.sqrt(1 - beta ** 2) * omega0 * t)[:, np.newaxis]
        c = (np.sqrt(1 - beta ** 2) * omega0)
        x = a * np.sin(b) / c[:, np.newaxis] * (self.t_axis > t[:, np.newaxis])
        return x


def sample_from_uniform_prior(prior_ranges, num_samples=10_000):
    parameters = np.zeros((num_samples, len(prior_ranges)))
    for idx, (low, high) in enumerate(prior_ranges):
        parameters[:, idx] = np.random.uniform(low=low, high=high,
                                               size=num_samples)
    return parameters


class SimulationsDataset(Dataset):
    """Dataset with parameters and corresponding raw simulations."""

    def __init__(self,
                 parameters: np.array,
                 simulations: np.array,
                 transform: callable = None):
        """
        Parameters
        ----------

        parameters: np.array
        Args:
            parameters : np.array
                array with parameters for simulation
            simulations : np.array
                array with corresponding simulated observations
            transform : callable = None
                optional transform to be applied on a sample.
        """
        self.parameters = parameters
        self.simulations = simulations
        self.transform = transform

    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'parameters': self.parameters[idx],
                  'simulations': self.simulations[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


def shift_time_series_by_t(x, t, t_axis_delta):
    return np.roll(x, int(t / t_axis_delta))


class NoiseModule:
    """
    Module for performing noise-related operation. Required methods:

    Methods
    -------
    get_random_noise_distribution:
        returns a random noise distribution used to generate noise
    whiten_signal_and_get_noise_summary:
        whiten a given signal, and also provide noise context summary
    add_noise:
        add white noise to whitened signal data
    """

    def __init__(self,
                 num_bins: int,
                 std: float = 1,
                 ):
        """
        Parameters
        ----------

        num_bins: int
            number of noise bins
        std: float = 1
            standard deviation of added white noise
        """
        self.num_bins = num_bins
        self.std = std

    def get_random_noise_distribution(self):
        """
        Generates a random noise distribution.

        Returns
        -------
        noise_distribution: np.ndarray
            array of the noise distribution
        """
        n = self.num_bins
        noise_distribution = np.ones(n)
        noise_distribution[:n // 4] *= 2.0
        noise_distribution[:n // 16] *= 4.0
        noise_distribution[:n // 64] *= 4.0
        np.random.shuffle(noise_distribution)
        return noise_distribution

    def whiten_signal_and_get_noise_summary(self,
                                            signal: np.ndarray,
                                            noise_distribution: np.ndarray =
                                            None,
                                            ):
        """
        Sample a random noise distribution (unless provided), whiten the signal
        accordingly. Return the whitened signal and the summary information
        of the noise distribution.

        Parameters
        ----------
        signal: np.ndarray
            the signal to be whitened
        noise_distribution: np.ndarray = None
            noise distribution; if not provided, random one is sampled

        Returns
        -------
        signal: np.ndarray
            the whitened signal
        noise_summary: np.ndarray
             summary information about the sampled noise distribution
        """
        if noise_distribution is None:
            noise_distribution = self.get_random_noise_distribution()
        signal = signal / noise_distribution
        noise_summary = self.get_noise_summary(noise_distribution)
        return signal, noise_summary

    def get_noise_summary(self, noise_distribution: np.ndarray):
        """
        Get the summary information for the noise distribution.

        Parameters
        ----------
        noise_distribution: np.ndarray
            array with noise distribution
        Returns
        -------
        noise_summary: np.ndarray
            summary information for the noise distribution
        """
        return 1. / noise_distribution

    def get_noise_distribution_from_summary(self, noise_summary: np.ndarray):
        """
        Get the noise distribution from its summary information. This inverts
        the method get_noise_summary.

        Parameters
        ----------
        noise_summary: np.ndarray
            array with noise summary
        Returns
        -------
        noise_distribution: np.ndarray
            noise distribution
        """
        return 1. / noise_summary

    def add_white_noise(self, signal: np.ndarray):
        """
        Add white gaussian noise with standard deviation self.std to signal.

        Parameters
        ----------
        signal: np.ndarray
            whitened signal without noise

        Returns
        -------
        signal: np.ndarray
            signal with added noise
        """
        signal[:2] += np.random.normal(loc=0, scale=self.std,
                                       size=(signal[:2].shape))


##################################################
# Transformation classes for harmonic oscillator #
##################################################

class ProjectOntoDetectors(object):
    """
    This takes the intrinsic simulator parameters and the raw oscillation
    data x as input. It samples the extrinsic parameter t1, the time of the
    pulse in Detector 1. It returns the projection of the raw oscillation x
    onto the detectors in a single array x_out of shape
    (num_detectors, num_channels, len(x)).

    The signal for detector i is stored as x_out[i, 0, :] = x.real and
    x_out[i, 1, :] = x.imag. Channels with indeces > 1 are used for noise
    context information, and initialized with zeros.
    """

    def __init__(self,
                 num_detectors: int = 2,
                 num_channels: int = 2,
                 t_axis_delta: int = None,
                 ti_priors_list: list = None,
                 ):
        if ti_priors_list is None:
            ti_priors_list = [[0, 5]] * num_detectors
        self.num_detectors = num_detectors
        self.num_channels = num_channels
        self.t_axis_delta = t_axis_delta
        self.ti_priors_list = ti_priors_list

    def __call__(self, sample):
        theta, x = sample['parameters'], sample['simulations']
        x_mod = np.zeros((self.num_detectors, self.num_channels, len(x)))

        ti_list = []
        for idx in range(self.num_detectors):
            ti = np.random.uniform(*self.ti_priors_list[idx])
            ti_list.append(ti)
            x_idx = shift_time_series_by_t(x, ti,
                                           t_axis_delta=self.t_axis_delta)
            x_mod[idx, 0, :] = x_idx.real
            x_mod[idx, 1, :] = x_idx.imag

        theta = np.concatenate((theta, ti_list))

        return {'parameters': theta, 'simulations': x_mod}


class WhitenSignalAndGetNoiseSummary(object):
    """
    This takes the signal data, whitens it and adds the noise context to the
    context dimension. If no noise_distribution is provided in the sample,
    a random one is sampled.
    """

    def __init__(self, noise_module: NoiseModule):
        self.noise_module = noise_module

    def __call__(self, sample):
        x = sample.pop('simulations')
        try:
            noise_distribution = sample['noise_distributions']
        except KeyError:
            noise_distribution = [None] * x.shape[0]
        for idx in range(x.shape[0]):
            x_det, noise_summary = \
                self.noise_module.whiten_signal_and_get_noise_summary(
                    x[idx], noise_distribution[idx])
            x[idx] = x_det
            x[idx, 2:] = noise_summary

        sample['simulations'] = x
        return sample


class AddWhiteNoise(object):
    """
    This takes the signal data and adds white noise.
    """

    def __init__(self, noise_module: NoiseModule):
        self.noise_module = noise_module

    def __call__(self, sample):
        theta, x = sample['parameters'], sample['simulations']
        for idx in range(x.shape[0]):
            self.noise_module.add_white_noise(x[idx])

        return {'parameters': theta, 'simulations': x}


class NormalizeParameters(object):
    """
    Normalization transform. If inverse=False, the sample parameters are
    normalized by subtracting means and dividing by standard deviations. If
    inverse=True, the parameters are denormalized by multiplying by standard
    deviations and adding means.

    Parameters
    ----------
    means: np.ndarray
        array with means of sample parameters
    stds: np.ndarray
        array with standard deviations of sample parameters
    inverse: bool = False
        if True, apply normalization; if False, apply denormalization
    """

    def __init__(self,
                 means: np.ndarray,
                 stds: np.ndarray,
                 inverse: bool = False,
                 ):
        self.means = means
        self.stds = stds
        self.inverse = inverse

    def __call__(self, sample):
        theta, x = sample['parameters'], sample['simulations']
        if not self.inverse:
            theta = (theta - self.means) / self.stds
        else:
            theta = theta * self.stds + self.means

        return {'parameters': theta, 'simulations': x}


class DictElementsToTensor(object):
    """Convert all numpy arrays in sample dict to torch tensors."""

    def __call__(self, sample):
        for key, item in sample.items():
            if type(item) is np.ndarray:
                sample[key] = torch.from_numpy(item)
        return sample


class ConvertDictsToArray(object):
    """
    This iterates through all elements of the sample dict and checks, whether
    the element is a dict itself. In that case, it converts this dictionary
    to a numpy array by concatenating the elements.
    """

    def __call__(self, sample):
        for name, item in sample.items():
            if type(item) == dict:
                shape = next(iter(item.values())).shape
                x = np.zeros((len(item), *shape))
                for idx, (_, item_) in enumerate(item.items()):
                    x[idx] = item_
                sample[name] = x
        return sample


class AddContextChannels(object):
    """
    Add num_channels additional context channels along axis 1 to sample[key].
    """

    def __init__(self, key, num_channels=1):
        self.key = key
        self.num_channels = num_channels

    def __call__(self, sample):
        x = sample[self.key]
        shape_new = list(x.shape)
        shape_new[1] += self.num_channels
        x_new = np.zeros(shape_new)
        x_new[:, :x.shape[1], :] = x
        sample[self.key] = x_new
        return sample


class NetworkInputToObservation(object):
    """
    This takes simulated data that is processed for training input and
    transforms this to a non-whitened observation.

    Parameters
    ----------

    noise_module: NoiseModule
        noise module used to generate noise distribution from noise summary
    """

    def __init__(self, noise_module: NoiseModule):
        self.noise_module = noise_module

    def __call__(self, sample):
        # separate whitened observation from simulation
        observations_white = np.array(sample['simulations'][:, :2, :])
        Nd = len(observations_white)
        # get noise distribution from context data
        noise_distributions = np.array(
            self.noise_module.get_noise_distribution_from_summary(
                sample['simulations'][:, 2, :]
            )
        )
        # undo the whitening
        observations = np.zeros_like(observations_white)
        for idx in range(Nd):
            observations[idx] = \
                observations_white[idx] * noise_distributions[idx]
        # store observed data in dicts
        observations = {
            'Detector{:}'.format(idx): observations[idx] for idx in range(Nd)
        }
        noise_distributions = {
            'Detector{:}'.format(idx): noise_distributions[idx] for idx in
            range(Nd)
        }
        sample = {
            'simulations': observations,
            'noise_distributions': noise_distributions,
        }
        return sample


def get_train_transformations_composite(projection_kwargs: dict,
                                        noise_module_kwargs: dict,
                                        normalization_kwargs: dict,
                                        ):
    """
    Build the full preprocessing transformation for the training data,
    the transforms the raw data from the dataset to input tensors for the
    neural network.

    Parameters
    ----------
    projection_kwargs: dict
        kwargs for projection onto the detectors
    noise_module_kwargs: dict
        kwargs for the noise module, which is used for whitening the signals
        and adding noise
    normalization_kwargs: dict
        kwargs for normalization of model parameters; contains means and stds

    Returns
    -------
    train_transformation
        transformation object for preprocessing training data
    """
    transform_projection = ProjectOntoDetectors(**projection_kwargs)

    noise_module = NoiseModule(**noise_module_kwargs)
    transform_whiten_signal_and_add_noise_summary = \
        WhitenSignalAndGetNoiseSummary(noise_module)
    transform_add_white_noise = AddWhiteNoise(noise_module)

    transform_normalize_parameters = NormalizeParameters(
        **normalization_kwargs, inverse=False)

    transform_to_tensors = DictElementsToTensor()

    train_transformation = transforms.Compose([
        transform_projection,
        transform_whiten_signal_and_add_noise_summary,
        transform_add_white_noise,
        transform_normalize_parameters,
        transform_to_tensors,
    ])

    return train_transformation


def get_transformations_composites(projection_kwargs: dict,
                                   noise_module_kwargs: dict,
                                   normalization_kwargs: dict,
                                   ):
    """
    Build the full preprocessing transformation for the training data,
    the transforms the raw data from the dataset to input tensors for the
    neural network.

    Parameters
    ----------
    projection_kwargs: dict
        kwargs for projection onto the detectors
    noise_module_kwargs: dict
        kwargs for the noise module, which is used for whitening the signals
        and adding noise
    normalization_kwargs: dict
        kwargs for normalization of model parameters; contains means and stds

    Returns
    -------
    train_transformation
        transformation object for preprocessing training data
    """
    noise_module = NoiseModule(**noise_module_kwargs)

    # train transformation
    transform_list = []
    transform_list.append(ProjectOntoDetectors(**projection_kwargs))
    transform_list.append(WhitenSignalAndGetNoiseSummary(noise_module))
    transform_list.append(AddWhiteNoise(noise_module))
    transform_list.append(NormalizeParameters(**normalization_kwargs,
                                              inverse=False))
    transform_list.append(DictElementsToTensor())
    train_transformation = transforms.Compose(transform_list)

    # generate observations transformation
    gen_obs_transformation = NetworkInputToObservation(noise_module)

    # inference transformation
    transform_list = []
    transform_list.append(ConvertDictsToArray())
    transform_list.append(AddContextChannels('simulations', 1))
    transform_list.append(WhitenSignalAndGetNoiseSummary(noise_module))
    transform_list.append(DictElementsToTensor())
    inference_transformation = transforms.Compose(transform_list)

    return train_transformation, gen_obs_transformation, \
           inference_transformation


def get_means_and_stds_from_uniform_prior_ranges(
        prior_ranges: Union[list, np.ndarray],
):
    """
    Compute and return means and standard deviations of uniform distributions
    with specified prior ranges.

    Parameters
    ----------
    prior_ranges: Union[list, np.ndarray]
        ranges of uniform priors in format [[low, high], [low, high], ...]

    Returns
    -------
    means, stds: (np.ndarray, np.ndarray)
        numpy arrays with means and standard deviations
    """
    prior_ranges = np.array(prior_ranges)
    means = np.mean(prior_ranges, axis=1)
    stds = np.abs(prior_ranges[:, 1] - prior_ranges[:, 0]) / np.sqrt(12.)
    return means, stds


def main_old():
    from scipy import linalg

    t_axis = [0, 10, 1000]
    simulator = HarmonicOscillator(*t_axis)

    prior_ranges = [[0, 5], [3.0, 10.0], [0.2, 0.5]]
    parameters = sample_from_uniform_prior(prior_ranges, num_samples=5_000)
    simulations = simulator.simulate(parameters)
    U, s, Vh = linalg.svd(simulations)
    V = Vh.T.conj()

    prior_ranges_GNPE = [[5, 5], [3.0, 10.0], [0.2, 0.5]]
    parameters_GNPE = sample_from_uniform_prior(prior_ranges_GNPE,
                                                num_samples=5_000)
    parameters_GNPE[:, 0] += np.random.normal(loc=0, scale=0.1,
                                              size=len(parameters_GNPE))
    simulations_GNPE = simulator.simulate(parameters_GNPE)
    U_GNPE, s_GNPE, Vh_GNPE = linalg.svd(simulations_GNPE)
    V_GNPE = Vh.T.conj()

    print('Effective dimension.\nThreshold\tNPE\t\tGNPE')
    for thr in [0.1, 0.01, 0.001]:
        print('{:.3f}\t\t{:}\t\t{:}' \
              .format(thr, np.argmax(s < thr) - 1, np.argmax(s_GNPE < thr) - 1))

    plt.ylabel('singular value')
    plt.xlabel('index of singular value')
    plt.xlim((0, 300))
    plt.ylim((1e-5, 1e2))
    plt.yscale('log')
    plt.plot(s, label='NPE')
    plt.plot(s_GNPE, label='GNPE')
    plt.legend()
    plt.show()

    for idx, data in enumerate(dataset):
        print(data['parameters'])
        plt.plot(simulator.t_axis, data['simulations'])
        plt.show()
        if idx == 2: break

    nrb = 20
    plt.plot(simulator.t_axis, simulations[0].real)
    plt.plot(simulator.t_axis, (simulations[0] @ V[:, :nrb] @ Vh[:nrb,
                                                              :]).real)
    plt.show()
    plt.plot((simulations[0] @ V[:, :100]).real)
    plt.plot((simulations[0] @ V[:, :100]).imag)
    plt.show()
