from typing import Dict, Union
import numpy as np

from dingo.gw.detector_network import DetectorNetwork
from bilby.gw.detector import PowerSpectralDensity


class PSD:
    """
    Load a PSD from a file, or look up a standard PSD available from lalsuite / bilby.

    TODO:
        * Add lookup of reference PSDs
        * Take a domain class instance and then truncate the generated PSD onto this domain
          or always use the domain grid as input to get_power_spectral_density_array.
        * Should a DetectorNetwork have a list of PSDs?
        * Caveat: Don't use PSDs with float32!
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
    """draw PSD object()

    TODO:
       * implement a database of PSDs for each detector
       * Want to randomly draw a PSD for each detector
       * Consider open / non-open data
       * transform to a standard domain

    """
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

        TODO: add support for using PSD and PSDDataSet classes via
          DetectorNetwork
        """
        self.network = network
        self.domain = network.domain
        self.psd_dict = network.power_spectral_densities
        self._compute_amplitude_spectral_densities()


    def _compute_amplitude_spectral_densities(self):
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

    def _noise_summary_function(self, asd: np.ndarray) -> np.ndarray:
        """
        Map the inverse ASD into [0, 1].
        """
        z = 1.0 / asd
        return z / np.max(z)

    def noise_summary(self) -> Dict[str, np.ndarray]:
        """
        Return a dictionary of noise summary data for the given
        detector network. This serves as context data for the NN flow.
        """
        return {ifo: self._noise_summary_function(asd)
                for ifo, asd in self.asd_dict.items()}

    def _whiten_waveform(self, strain: np.ndarray, asd_array: np.ndarray) -> np.ndarray:
        """
        Whiten strain with the given amplitude spectral density.

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

        Return nested dictionary of waveform parameters, strains,
        and noise summary information for the detector network.

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
                'waveform': strain_dict,
                'noise_summary': self.noise_summary()}


    # TODO: somewhere add a method to calculate SNR
    #     snr = (np.sqrt(np.sum(np.abs(stacked)**2)) / self._noise_std)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    psd = PSD(psd_file='aLIGO_ZERO_DET_high_P_psd.txt')
    f = np.logspace(np.log10(5), np.log10(20000), 1000)
    psd_array = psd.get_power_spectral_density_array(f)
    f_defined = f[np.isfinite(psd_array)]
    print(f'PSD undefined for f<{f_defined[0]}, f>{f_defined[-1]} Hz')
    # plt.loglog(f, psd_array)
    # plt.show()
    # TODO: add unit tests



    from dingo.gw.parameters import GWPriorDict
    from dingo.gw.domains import UniformFrequencyDomain
    from dingo.gw.detector_network import DetectorNetwork, RandomProjectToDetectors
    from dingo.gw.noise import AddNoiseAndWhiten
    from dingo.gw.waveform_generator import WaveformGenerator
    from dingo.gw.waveform_dataset import WaveformDataset
    import matplotlib.pyplot as plt

    domain_kwargs = {'f_min': 20.0, 'f_max': 4096.0, 'delta_f': 1.0 / 8.0, 'window_factor': 1.0}
    domain = UniformFrequencyDomain(**domain_kwargs)
    # priors = GWPriorDict(geocent_time_ref=1126259642.413, luminosity_distance_ref=500.0,
    #                      reference_frequency=20.0)
    # approximant = 'IMRPhenomXPHM'
    # waveform_generator = WaveformGenerator(approximant, domain)


    # Generate a dataset using the first waveform dataset object
    # wd = WaveformDataset(priors=priors, waveform_generator=waveform_generator)
    # n_waveforms = 17
    # wd.generate_dataset(size=n_waveforms)

    # 1. RandomProjectToDetectors  (single waveform)
    #    in:  waveform_polarizations: Dict[str, np.ndarray], waveform_parameters: Dict[str, float]
    #    out: Dict[str, np.ndarray]      {'H1':h1_strain, ...}
    #    changes extrinsic pars

    det_network = DetectorNetwork(["H1", "L1"], domain)
    nw = AddNoiseAndWhiten(det_network)
    fake_strain_dict = {'H1': np.zeros_like(domain()), 'L1': np.ones_like(domain())}
    data_dict = {'parameters': None, 'waveform': fake_strain_dict}
    ret_dict = nw(data_dict)
    assert ret_dict['parameters'] == data_dict['parameters']
    assert len(ret_dict.keys()) == 3
    n = ret_dict['waveform']['H1']
    print(n)
    plt.plot(domain(), n.real)
    print(np.mean(n.real), np.std(n.real))
    plt.show()

    # noise summary
    # plt.clf()
    # for ifo, ns in nw.noise_summary().items():
    #     plt.plot(domain(), ns, label=ifo)
    # plt.legend()
    # plt.show()

