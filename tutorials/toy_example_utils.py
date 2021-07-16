import numpy as np


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
                                            ):
        """
        Sample a random noise distribution, whiten the signal accordingly.
        Return the whitened signal and the summary information of the noise
        distribution.

        Parameters
        ----------
        signal: np.ndarray
            the signal to be whitened

        Returns
        -------
        signal: np.ndarray
            the whitened signal
        noise_summary: np.ndarray
             summary information about the sampled noise distribution
        """
        noise_distribution = self.get_random_noise_distribution()
        signal = signal / noise_distribution
        noise_summary = 1. / noise_distribution
        return signal, noise_summary

    def add_white_noise(self,
                        signal: np.ndarray,
                        ):
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
            ti_priors_list = [[0,5]] * num_detectors
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
    context dimension.
    """

    def __init__(self,
                 noise_module: NoiseModule,
                 ):
        self.noise_module = noise_module

    def __call__(self, sample):
        theta, x = sample['parameters'], sample['simulations']
        for idx in range(x.shape[0]):
            x_det, noise_summary = \
                self.noise_module.whiten_signal_and_get_noise_summary(x[idx])
            x[idx] = x_det
            x[idx, 2:] = noise_summary

        return {'parameters': theta, 'simulations': x}


class AddWhiteNoise(object):
    """
    This takes the signal data and adds white noise.
    """

    def __init__(self,
                 noise_module: NoiseModule,
                 ):
        self.noise_module = noise_module

    def __call__(self, sample):
        theta, x = sample['parameters'], sample['simulations']
        for idx in range(x.shape[0]):
            self.noise_module.add_white_noise(x[idx])

        return {'parameters': theta, 'simulations': x}


class NormalizeParameters(object):
    """

    """

    def __init__(self, prior_ranges):
        prior_ranges = np.array(prior_ranges)
        self.means = np.mean(prior_ranges, axis=1)
        self.stds = np.abs(prior_ranges[:,1] - prior_ranges[:,0]) / np.sqrt(12.)

    def __call__(self, sample):
        theta, x = sample['parameters'], sample['simulations']
        theta = (theta - self.means) / self.stds

        return {'parameters': theta, 'simulations': x}

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
