# from dingo.gw.domains import UniformFrequencyDomain
# from dingo.gw.waveform_generator import WaveformGenerator
# from dingo.gw.detector_network import DetectorNetwork, RandomProjectToDetectors
# from dingo.gw.parameters import GWPriorDict
import pytest
import numpy as np


@pytest.fixture
def generate_waveform_polarizations():
    approximant = 'IMRPhenomPv2'
    f_min = 20.0
    f_max = 512.0
    domain = UniformFrequencyDomain(f_min=f_min, f_max=f_max, delta_f=1.0/4.0, window_factor=1.0)
    parameters = {'chirp_mass': 34.0, 'mass_ratio': 0.35, 'chi_1': 0.2, 'chi_2': 0.1, 'theta_jn': 1.57,
                  'f_ref': 20.0, 'phase': 0.0, 'luminosity_distance': 1.0, 'geocent_time': 1126259642.413}
    wg = WaveformGenerator(approximant, domain, reference_frequency=parameters['f_ref'])
    waveform_polarizations = wg.generate_hplus_hcross(parameters)
    return waveform_polarizations, parameters, domain


@pytest.mark.obsolete
def test_detector_network(generate_waveform_polarizations):
    waveform_polarizations, wf_parameters, domain = generate_waveform_polarizations
    ifo_list = ["H1", "L1"]

    priors = GWPriorDict()
    extrinsic_parameters = priors.sample_extrinsic()
    wf_dict = {'plus': waveform_polarizations['h_plus'],
               'cross': waveform_polarizations['h_cross']}

    # dingo DetectorNetwork class wrapping bilby's Interferometer class
    det_network = DetectorNetwork(ifo_list, domain, start_time=0)
    strain_dict_dingo = det_network.project_onto_network(
        wf_dict, extrinsic_parameters)

    # use bilby's detector classes directly
    from bilby.gw.detector import InterferometerList
    ifos = InterferometerList(ifo_list)
    ifos.set_strain_data_from_zero_noise(2*domain.f_max, domain.duration, start_time=0)
    strain_dict_bilby = {ifo.name: ifo.get_detector_response(wf_dict, extrinsic_parameters)
                         for ifo in ifos}

    for ifo_name in strain_dict_dingo.keys():
        assert np.allclose(strain_dict_dingo[ifo_name], strain_dict_bilby[ifo_name])


@pytest.mark.obsolete
def test_random_project_to_detectors(generate_waveform_polarizations):
    waveform_polarizations, parameters, domain = generate_waveform_polarizations
    ifo_list = ["H1", "L1"]
    det_network = DetectorNetwork(ifo_list, domain, start_time=0)
    priors = GWPriorDict()
    rp_det = RandomProjectToDetectors(det_network, priors)
    strain_dict = rp_det({'parameters': parameters, 'waveform': waveform_polarizations})
    assert len(strain_dict) == len(ifo_list)
    assert len(list(strain_dict['waveform'].values())[0]) == len(domain)

