"""
This is an implementation of parameter inference for a toy example. We build
a toy model based on the harmonic oscillator with similar properties as the
GW parameter estimation problem. We then use (G)NPE to perform parameter
inference.

As forward model we use a harmonic oscillator that is initially at rest,
and then excited with an infinitesimally short pulse. After excitation,
there is a damped oscillation.

The forward model has 2 intrinsic parameters:
    omega0 :    resonance frequency of undamped oscillator
    beta :      damping factor

To build an example similar to the GW use case, the corresponding observation
is a pair of time series in Nd different detectors. The signal arrives in
Detector i at time ti, which is sampled for each detector at train time. The
model has thus Nd extrinsic parameters. In total, the forward model has 2 +
Nd parameters {t, omega0, beta, t0, t1, ...}.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from toy_example_utils import *


class HarmonicOscillator:
    """
    Simulator for the harmonic oscillator forward model. It is initialised
    with a axis for the time samples T where the oscillator is evaluated.
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


if __name__ == '__main__':


    # set up forward model
    t_axis = [0, 10, 1000]
    simulator = HarmonicOscillator(*t_axis)

    # get dataset for extrinsic parameters
    extrinsic_prior = [[3.0, 10.0], [0.2, 0.5]]
    intrinsic_prior = [[0, 3], [2, 5]]
    prior = extrinsic_prior + intrinsic_prior
    parameters = sample_from_uniform_prior(extrinsic_prior, num_samples=5_000)
    simulations = simulator.simulate(parameters)
    dataset = SimulationsDataset(parameters, simulations)

    ###################
    # transformations #
    ###################

    transform_projection = ProjectOntoDetectors(
        num_detectors=2,
        num_channels=3,
        t_axis_delta=simulator.delta_t,
        ti_priors_list=intrinsic_prior,
    )

    noise_module = NoiseModule(num_bins=simulator.N_bins, std=0.01)
    transform_whiten_signal_and_add_noise_summary = \
        WhitenSignalAndGetNoiseSummary(noise_module)
    transform_add_white_noise = AddWhiteNoise(noise_module)

    transform_normalize_parameters = NormalizeParameters(prior)

    # get data from dataset
    data = dataset[0]
    print(data['parameters'])
    plt.title('Raw data')
    plt.plot(simulator.t_axis, data['simulations'])
    plt.show()

    # sample extrinsic parameters, project onto the detectors
    data = transform_projection(data)
    print(data['parameters'])
    for idx in range(data['simulations'].shape[0]):
        plt.title('Projection onto detector {:}'.format(idx))
        plt.plot(simulator.t_axis, data['simulations'][idx, 0], label='real')
        plt.plot(simulator.t_axis, data['simulations'][idx, 1], label='imag')
        plt.plot(simulator.t_axis, data['simulations'][idx, 2])
        plt.legend()
        plt.show()

    # draw noise distribution, whiten and add context data
    data = transform_whiten_signal_and_add_noise_summary(data)
    print(data['parameters'])
    for idx in range(data['simulations'].shape[0]):
        plt.title('Whitened data in detector i {:}'.format(idx))
        plt.plot(simulator.t_axis, data['simulations'][idx, 0], label='real')
        plt.plot(simulator.t_axis, data['simulations'][idx, 1], label='imag')
        plt.legend()
        plt.show()
        plt.title('Noise summary in detector i {:}'.format(idx))
        plt.plot(simulator.t_axis, data['simulations'][idx, 2])
        plt.show()

    # add white noise to whitened signal
    data = transform_add_white_noise(data)
    print(data['parameters'])
    for idx in range(data['simulations'].shape[0]):
        plt.title('Noisy data in detector i {:}'.format(idx))
        plt.plot(simulator.t_axis, data['simulations'][idx, 0], label='real')
        plt.plot(simulator.t_axis, data['simulations'][idx, 1], label='imag')
        plt.legend()
        plt.show()
        plt.title('Noise summary in detector i {:}'.format(idx))
        plt.plot(simulator.t_axis, data['simulations'][idx, 2])
        plt.show()

    data = transform_normalize_parameters(data)
    print(data['parameters'])

    pass
