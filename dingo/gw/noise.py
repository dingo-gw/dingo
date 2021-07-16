from typing import Dict, Union
import numpy as np

from dingo.gw.detector_network import DetectorNetwork
from bilby.gw.detector import PowerSpectralDensity

from dingo.gw.domains import UniformFrequencyDomain

# TODO:
#  * Caveat: Don't use PSDs with float32!
#  * simplest case: fixed designed sensitivity PSD
#  * noise needs to know the domain
#  * more complex: database of PSDs for each detector
#    - randomly select a PSD for each detector
#  * Maybe create a PSD_DataSet class (open / non-open data), and transform
#    - at each call randomly draw a psd
#  * window_function

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

    def provide_context(self):
        # TODO: What should this do in practse?
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
    def __init__(self, network: DetectorNetwork, whiten_data=True, add_noise=True):
        """
        Whiten strain data and add zero-mean, white Gaussian noise.

        network : DetectorNetwork
            An instance of DetectorNetwork which implicitly defines
            the physical domain and the power spectral density for
            each detector in the network.
        whiten_data : bool
            Whether to whiten the strain data in __call__
        add_noise : bool
            Whether to add white noise in __call__
            Caveat: If whiten_data = False, and the input strain
            data has not been whitened this may not be want you want.
        """
        self.network = network
        self.domain = network.domain
        self.psd_dict = network.power_spectral_densities
        self._compute_amplitude_spectral_densities()


    def _compute_amplitude_spectral_densities(self) -> Dict[str, np.ndarray]:
        """
        Compute the amplitude spectral density from the
        power spectral densities for all detectors in the network.
        """
        self.asd_dict = {ifo: np.sqrt(psd)
                         for ifo, psd in self.psd_dict.items()}


    def _generate_Gaussian_white_noise(self) -> np.ndarray:
        """
        Generate complex zero-mean Gaussian white noise in the Fourier domain

        Assume a given frequency grid (or sampling rate and duration)
        as defined by the domain. Then draw samples

            \tilde n_white(f_i) ~ N(0, sigma) + i N(0, sigma)

        where sigma is the standard deviation of the noise including
        the window factor, and the noise in two frequency bins i and j
        is independent for i != j.

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

    def _whiten_waveform(self, strain: np.ndarray, asd_array: np.ndarray) -> np.ndarray:
        """
        Whiten strain with then given amplitude spectral density.

        Parameters
        ----------
        strain : np.ndarray
            GW strain in a particular detector
        asd_array : np.ndarray
            The amplitude spectral density in a particular detector
        """
        return strain / asd_array

    def __call__(self, waveform_dict: Dict[str, Dict[str, Union[float, np.ndarray]]]) \
            -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """
        Whiten detector strain waveforms and add zero-mean,
        white Gaussian noise.

        Parameters
        ----------
        waveform_dict: Dict[Dict[str, float], Dict[str, np.ndarray]]
            Nested dictionary of parameters and strains
            {'parameters': ..., 'waveform': ...}

        """
        strain_dict = waveform_dict['waveform']

        # 1. Whiten waveform
        strain_dict = {ifo: self._whiten_waveform(h, self.asd_dict[ifo])
                       for ifo, h in strain_dict.items()}

        # 2. Generate and add zero-mean white Gaussian noise
        strain_dict = {ifo: h + self._generate_Gaussian_white_noise()
                       for ifo, h in strain_dict.items()}

        return {'parameters': waveform_dict['parameters'],
                'waveform': strain_dict, 'asd': self.asd_dict}


    # TODO: somewhere add method to calculate SNR
    #     snr = (np.sqrt(np.sum(np.abs(stacked)**2)) / self._noise_std)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    psd = PSD(psd_file='aLIGO_ZERO_DET_high_P_psd.txt')
    f = np.logspace(np.log10(5), np.log10(20000), 1000)
    psd_array = psd.get_power_spectral_density_array(f)
    f_defined = f[np.isfinite(psd_array)]
    print(f'PSD undefined for f<{f_defined[0]}, f>{f_defined[-1]} Hz')
    plt.loglog(f, psd_array)
    plt.show()
    # TODO: add unit tests

