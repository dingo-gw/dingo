from typing import Dict
import numpy as np

from dingo.gw.detector_network import DetectorNetwork
from bilby.gw.detector import PowerSpectralDensity

from dingo.gw.domains import UniformFrequencyDomain

# TODO:
#  * Support transformations
#    1. add noise to detector_projected wf sample
#    2. whiten
#  * Caveat: Don't use PSDs with float32!
#  * simplest case: fixed designed sensitivity PSD
#  * noise needs to know the domain
#  * more complex: database of PSDs for each detector
#    - randomly select a PSD for each detector
#  * Maybe create a PSD_DataSet class (open / non-open data), and transform
#    - at each call randomly draw a psd
#  * window_function
# bilby's create_white_noise

# TODO: Noise class needs to provide:
# 1. sample_noise()
# 2. provide_context() -- will not modify a tensor, but spit out a tensor that has the shape of expected noise summary
#     particular function of the PSD
class PSD:
    """TODO: Should a DetectorNetwork have a list of PSDs?

    This class could take a domain class and then truncate the generate PSD onto this domain
    or always use the domain grid as input to get_power_spectral_density_array.
    """
    def __init__(self, psd_file: str):
        if psd_file is not None:
            self._psd = PowerSpectralDensity.from_power_spectral_density_file(psd_file)
        else:
            raise ValueError('Please specify a psd file.')

    def get_power_spectral_density_array(self, frequency_array: np.ndarray):
        """Return the PSD for the specified frequency array.

        Parameters
        ----------

        frequency_array : np.ndarray
            Frequency array in Hz.

        For frequencies in the frequency array for which the PSD is not
        defined return np.inf.
        """
        return self._psd.get_power_spectral_density_array(frequency_array)

    # def sample_noise(self, sampling_frequency, duration):
    #     """
    #     Generate colored noise with zero mean in the Fourier domain
    #
    #     1. Generates white noise with given fs and duration.
    #     \tilde n_white(f_i) ~ N(0, sigma) + i N(0, sigma)
    #     sigma = 1/2 1/sqrt(df)  # missing window factor
    #     Set DC to zero. If number of samples is even, set Nyquist = 0.
    #
    #     2. Color by PSD:
    #     \tilde n_white(f_i) * sqrt(S_n(f_i))
    #
    #     TODO: Does not include sqrt(window_factor) which needs to be multiplied with sigma above!
    #      therefore not useful for us
    #
    #     sampling_frequency:
    #     duration:
    #     """
    #     return self._psd.get_noise_realisation(sampling_frequency, duration)

    def provide_context(self):
        pass



class PSDDataSet:
    """draw PSD object()"""
    # TODO: Look at old code: class NoiseModule
    # Where does the random choice of index come in?
    # transform class would have to call sample_index() and then sample_...
    def sample_index(self):
        pass
        # index = .... # numpy generator -- be careful about setting seed

    def sample_noise(self):
        self._sample_noise(self, self.index)

    def _sample_noise(self, index):
        pass

    def provide_context(self, index):
        pass


class AddNoiseAndWhiten:
    def __init__(self, network: DetectorNetwork):
        """
        Pass in a DetectorNetwork which can return the PSD of each detector.
        (Alternatively could pass in a dict of PSD objects for the network.)

        network: DetectorNetwork
        """
        self.network = network
        self.domain = network.domain
        self.psd_dict = network.power_spectral_densities

    def _generate_white_noise(self) -> np.ndarray:
        """
        Generate complex white noise in the Fourier domain

        Assume a given frequency grid (or sampling rate and duration)
        as defined by the domain. Then draw samples

            \tilde n_white(f_i) ~ N(0, sigma) + i N(0, sigma)

        where sigma is the standard deviation of the noise including
        the window factor.

        Set DC to zero. If number of samples is even, set Nyquist = 0.
        """
        # Assume uniform FD for now
        d = self.domain
        n_samples = len(d)
        n_white = np.random.normal(loc=0, scale=d.noise_std, size=n_samples) + \
             1j * np.random.normal(loc=0, scale=d.noise_std, size=n_samples)

        # TODO: do we want this?
        # set DC and Nyquist = 0
        n_white[0] = 0
        # no Nyquist frequency when N=odd
        if np.mod(n_samples, 2) == 0:
            n_white[-1] = 0

        return n_white

    def _whiten_waveform(self, strain: np.ndarray, psd_array: np.ndarray) -> np.ndarray:
        """
        Whiten strain with then given power spectral density.

        Parameters
        ----------
        strain : np.ndarray
            GW strain in a particular detector
        psd_array : np.ndarray
            The power spectral density in a particular detector
        """
        return strain / np.sqrt(psd_array)


    def __call__(self, waveform_dict: Dict[Dict[str, float], Dict[str, np.ndarray]]) -> Dict[Dict[str, float], Dict[str, np.ndarray]]:
        """
        Transform detector strain data transformation
        #    1. add noise to detector_projected wf sample
        #    2. whiten

        Parameters
        ----------
        waveform_dict: Dict[Dict[str, float], Dict[str, np.ndarray]]
            Nested dictionary of parameters and strains
            {'parameters': ..., 'waveform': ...}

        FIXME: I think the steps should be the other way around
        given what is done in the old code -- looking at the calls in WaveformDatasetTorch.__getitem__()
        otherwise would need to add colored noise, not white noise
        """
        strain_dict = waveform_dict['waveform']

        # 1. Whiten waveform
        strain_dict = {ifo: self._whiten_waveform(h, self.psd_dict[ifo])
                                for ifo, h in strain_dict.items()}

        # 2. Generate and add white noise
        strain_dict = {ifo: h + self._generate_white_noise()
                       for ifo, h in strain_dict.items()}

        return {'parameters': waveform_dict['parameters'], 'waveform': strain_dict}


        # Dev code:
        # # Obtain parameters and whitened waveforms
        # p, h, asd, w, snr = self.wfd.p_h_random_extrinsic(idx, self.train)
        #
        #     if whiten_with_reference_PSD:
        #         h_d = h_d / (self._get_psd(h_d.delta_f, ifo) ** 0.5)
        #
        #
        # # Add noise, reshape, standardize
        # x, y = self.wfd.x_y_from_p_h(p, h, asd, add_noise=self.add_noise)
        #
        #     # Add noise. Waveforms are assumed to be white in each detector.
        #     if add_noise:
        #         if self.domain in ('RB', 'FD'):
        #             noise = (np.random.normal(scale=self._noise_std, size=n)
        #                      + np.random.normal(scale=self._noise_std, size=n) * 1j)
        #             noise = noise.astype(np.complex64)
        #
        #         elif self.domain == 'TD':
        #             noise = np.random.normal(scale=self._noise_std, size=n)
        #             noise = noise.astype(np.float32)
        #
        #         d = d + noise


    # TODO: somewhere add method to calculate SNR
    #     snr = (np.sqrt(np.sum(np.abs(stacked)**2)) / self._noise_std)

    # TODO: look at init_relative_whitening() and whiten_relative(), whiten_absolute()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    psd = PSD(psd_file='aLIGO_ZERO_DET_high_P_psd.txt')
    f = np.logspace(np.log10(5), np.log10(20000), 1000)
    psd_array = psd.get_power_spectral_density_array(f)
    f_defined = f[np.isfinite(psd_array)]
    print(f'PSD unudefined for f<{f_defined[0]}, f>{f_defined[-1]} Hz')
    plt.loglog(f, psd_array)
    plt.show()

