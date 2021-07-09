import numpy as np
from typing import Dict, List

from bilby.gw.detector import Interferometer, InterferometerList

from dingo.gw.domains import Domain
from dingo.gw.parameters import GWPriorDict



class DetectorNetwork:
    """A wrapper class around bilby's Interferometer and InterferometerList

    Coupled to our Domain classes.
    """

    def __init__(self, ifo_list: List[str],
                 domain: Domain,
                 start_time: float = 0.0):
        """
        Parameters
        ----------
        ifo_list: List[str]
            List of detector strings to constitute the network
            The available instruments are: H1, L1, V1, GEO600, CE
        domain: Domain
            The physical domain on which waveforms are defined.
        start_time: float
            The GPS start-time of the data

        Strain data is initialized to zero in each detector.
        """
        self.ifos = InterferometerList(ifo_list)
        if not issubclass(type(domain), Domain):
            raise ValueError('domain should be an instance of a subclass of Domain, but got', type(domain))
        else:
            self.domain = domain

        # Initialize strain data to zero
        # Only support Domains which define f_max and duration
        sampling_frequency = 2*domain.f_max
        self.ifos.set_strain_data_from_zero_noise(sampling_frequency, domain.duration,
                                             start_time=start_time)
        # Note: Could also set strain data from a PSD if needed by calling
        # ifos.set_strain_data_from_power_spectral_densities()


    def detector_antenna_response(self, ifo: Interferometer,
                                  parameters: Dict[str, float],
                                  mode: str) -> float:
        """
        Calculate the antenna response function for a given detector

        Parameters
        ----------
        ifo: Interferometer
            The GW Interferometer object
        parameters: Dict[str, float]
            Dictionary containing detector response parameters
                ra: right ascension in radians
                dec: declination in radians
                time: geocentric GPS time
                psi: binary polarisation angle counter-clockwise
                    about the direction of propagation
        mode: str
            polarisation mode (e.g. 'plus', 'cross')

        Returns the response for the given polarization mode.
        """
        if not isinstance(ifo, Interferometer):
            raise ValueError('ifo should be a Interferometer')

        return ifo.antenna_response(parameters['ra'], parameters['dec'],
                                    parameters['geocent_time'],
                                    parameters['psi'], mode)


    def project_onto_detector(self, ifo: Interferometer,
                              waveform_polarizations: Dict[str, np.ndarray],
                              parameters: Dict[str, float]) -> np.ndarray:
        """
        Project waveform polarizations onto the given GW interferometer

        Parameters
        ----------
        ifo: Interferometer
            The GW Interferometer object
        waveform_polarizations: Dict[str, np.ndarray]
            The waveform polarizations (e.g. for 'plus', 'cross' modes)
        parameters: Dict[str, float]
            Dictionary containing detector response parameters
                ra: right ascension in radians
                dec: declination in radians
                time: geocentric GPS time
                psi: binary polarisation angle counter-clockwise
                    about the direction of propagation

        Adapted from bilby.gw.detector.interferometer.get_detector_response().

        Return the strain array.
        """
        strain = sum([wf_pol * self.detector_antenna_response(ifo, parameters, mode)
                      for mode, wf_pol in waveform_polarizations.items()])

        mask = self.domain.frequency_mask
        frequency_array = self.domain()
        strain *= mask

        # Apply time shift
        time_shift = ifo.time_delay_from_geocenter(
            parameters['ra'], parameters['dec'], parameters['geocent_time'])

        # Be careful to first subtract the two GPS times which are ~1e9 sec.
        # And then add the time_shift which varies at ~1e-5 sec
        dt_geocent = parameters['geocent_time'] - ifo.strain_data.start_time
        dt = dt_geocent + time_shift

        strain[mask] = strain[mask] * np.exp(-1j * 2*np.pi * dt * frequency_array[mask])

        # Apply calibration correction if the calibration model has been set
        # By default, there is no correction.
        strain[mask] *= ifo.calibration_model.get_calibration_factor(
            frequency_array[mask],
            prefix='recalib_{}_'.format(ifo.name), **parameters)

        return strain


    def project_onto_network(self, waveform_polarizations: Dict[str, np.ndarray],
                                   parameters: Dict[str, float]) -> Dict[str, np.ndarray]:
        """
        Project waveform polarizations onto the GW network

        Parameters
        ----------
        waveform_polarizations: Dict[str, np.ndarray]
            The waveform polarizations (e.g. for 'plus', 'cross' modes)
        parameters: Dict[str, float]
            Dictionary containing detector response parameters
                ra: right ascension in radians
                dec: declination in radians
                time: geocentric GPS time
                psi: binary polarisation angle counter-clockwise
                    about the direction of propagation

        Return a dictionary of the strains over the detectors in the network.
        """
        return {ifo.name: self.project_onto_detector(ifo, waveform_polarizations, parameters)
                    for ifo in self.ifos}


class RandomProjectToDetectors:
    """Given a sample waveform (in terms of its polarizations, and intrinsic parameters),
    draw a sample from the extrinsic parameter prior distribution and project on the
    given detector network. Return the strain.
    """

    def __init__(self, detector_network: DetectorNetwork,
                 extrinsic_prior: GWPriorDict):
        """
        Parameters
        ----------
        detector_network: DetectorNetwork
            A DetectorNetwork object. Its Domain object needs to be
            consistent with the physical domain on which the
            waveform polarizations passed to __call__() are defined.
        extrinsic_prior: GWPriorDict
            The prior distribution from which extrinsic samples will
            be drawn. Intrinsic parameters can be present and will be
            ignored.
        """

        self.detector_network = detector_network
        self.domain = detector_network.domain
        self.extrinsic_prior = extrinsic_prior

    def __call__(self, waveform_polarizations: Dict[str, np.ndarray],
                 waveform_parameters: Dict[str, float]):
        """
        Compute strain from waveform polarizations and detector network
        for a random draw from the extrinsic parameter prior distribution.

        Parameters
        ----------
        waveform_polarizations: Dict[str, np.ndarray]
            The waveform polarizations 'h_plus', 'h_cross'.
        waveform_parameters: Dict[str, float]
            Dictionary of parameters used to generate waveform polarizations.
            This needs to include any reference values for extrinsic parameters.
            FIXME: LIST WHICH VALUES THESE ARE
                CURRENTLY UNUSED
        """
        extrinsic_parameters = self.extrinsic_prior.sample_extrinsic()
        # FIXME: combine extrinsic parameter sample with extrinsic parameter
        #  reference values used to generate the polarizations

        # Switch to naming convention of polarization modes in DetectorNetwork (and bilby)
        wf_dict = {'plus': waveform_polarizations['h_plus'],
                   'cross': waveform_polarizations['h_cross']}
        # We require that extrinsic_parameters includes ra, dec, geocent_time, psi
        return self.detector_network.project_onto_network(wf_dict, extrinsic_parameters)

    #
    # def project_to_detectors(self, h_plus, h_cross, wf_parameters, extrinsic_parameters):
    #     # ifos[0].antenna_response(ra, dec, time, psi, mode)
    #     waveform_polarizations = {'plus': h_plus, 'cross': h_cross}
    #     # Uses
    #     # extrinsic_parameters['ra'],
    #     # extrinsic_parameters['dec'],
    #     # extrinsic_parameters['geocent_time'],
    #     # extrinsic_parameters['psi']
    #
    #
    #     # get_detector_response calls:
    #     #   antenna_response()
    #     #   for i in polarizations:
    #     #       h_ifo_i *= det_response
    #     #   h_ifo = sum(h_ifo_i.values())
    #     #   time_delay_from_geocenter
    #     #   dt = dt_geocent + time_shift
    #     #   h_ifo[mask] *= np.exp(-1j * 2 * np.pi * dt * frequency_array[mask])
    #     #
    #     #   To have more control about the grid we could just call this directly
    #     for ifo in ifos:
    #         ifo.strain_data.frequency_domain_strain = ifo.get_detector_response(
    #             waveform_polarizations, extrinsic_parameters)
    #     return [ifo.strain_data.frequency_domain_strain for ifo in ifos]
    #
    #     # see WaveformDataset.get_detector_waveforms()
    #     # Given ra, dec, psi, self.ref_time and a list of detector objects
    #     # loop over detectors, and compute h+ * F+ + hx + Fx and timeshift at detector
    #
    #     # Detector objects:
    #     # see WaveformDataset.init_detectors()
    #     # So far using pycbc.detector.Detector -- get rid of this dependency? Look at structure of this class and what is used in existing code
    #


if __name__ == "__main__":
    """A visual test."""
    from dingo.gw.domains import UniformFrequencyDomain
    from dingo.gw.waveform_generator import WaveformGenerator
    import matplotlib.pyplot as plt

    approximant = 'IMRPhenomPv2'
    f_min = 20.0
    f_max = 512.0
    domain = UniformFrequencyDomain(f_min=f_min, f_max=f_max, delta_f=1.0/4.0, window_factor=1.0)
    parameters = {'chirp_mass': 34.0, 'mass_ratio': 0.35, 'chi_1': 0.2, 'chi_2': 0.1, 'theta_jn': 1.57, 'f_ref': 20.0, 'phase': 0.0, 'luminosity_distance': 1.0}
    WG = WaveformGenerator(approximant, domain)
    waveform_polarizations = WG.generate_hplus_hcross(parameters)
    # print(waveform_polarizations['h_plus'])

    # plt.loglog(domain(), np.abs(waveform_polarizations['h_plus']))
    # plt.xlim([f_min/2, 2048.0])
    # plt.axvline(f_min, c='gray', ls='--')
    # plt.show()


    # f_min = 20.0
    # domain = UniformFrequencyDomain(f_min=f_min, f_max=4096.0, delta_f=1.0/4.0, window_factor=1.0)
    # priors = GWPriorDict()
    # ifos = InterferometerList(["H1", "L1"])
    # sampling_frequency = 2*f_max
    # duration = 4.0
    #ifos.set_strain_data_from_power_spectral_densities(sampling_frequency, duration, start_time=0)
    # ifos.set_strain_data_from_zero_noise(sampling_frequency, duration, start_time=0)
    # rp_det = RandomProjectToDetectors(domain, priors, ifos)
    # strain_list = rp_det(waveform_polarizations, parameters)
    # strain = strain_list[0]
    # idx = np.nonzero(strain)
    # print(strain[idx])
    # ifo = ifos[0]
    # plt.loglog(domain(), np.abs(waveform_polarizations['h_plus'] + 1j*waveform_polarizations['h_cross']))
    # plt.loglog(ifo.strain_data.frequency_array[idx], np.abs(strain[idx]))
    # plt.show()


    det_network = DetectorNetwork(["H1", "L1"], domain, start_time=0)
    priors = GWPriorDict()
    rp_det = RandomProjectToDetectors(det_network, priors)
    strain_dict = rp_det(waveform_polarizations, parameters)
    print(strain_dict)
    plt.loglog(domain(), np.abs(waveform_polarizations['h_plus'] + 1j*waveform_polarizations['h_cross']), '--', label='hp + i*hx')
    for name, strain in strain_dict.items():
        plt.loglog(domain(), np.abs(strain), label=name)
    plt.legend()
    plt.xlim([domain.f_min/2, domain.f_max])
    plt.show()

