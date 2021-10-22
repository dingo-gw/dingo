import yaml

from bilby.gw.detector import InterferometerList
from bilby.gw.conversion import component_masses_to_chirp_mass

from dingo.gw.waveform_dataset import WaveformDataset
from dingo.gw.prior_split import default_extrinsic_dict
from dingo.gw.domains import build_domain
from dingo.gw.transforms_parameters import SampleExtrinsicParameters
from dingo.gw.transforms_detectors import GetDetectorTimes, ProjectOntoDetectors

import numpy as np





if __name__ == '__main__':
    wfd_path = '/Users/mdax//Documents/dingo/devel/dingo-devel/tutorials/02_gwpe' \
               '/datasets/waveforms/02_IMR_test/waveform_dataset.hdf5'
    wfd = WaveformDataset(wfd_path)

    with open('./train_dir/train_settings.yaml', 'r') as fp:
        train_settings = yaml.safe_load(fp)

    extrinsic_prior_dict = default_extrinsic_dict.copy()
    for k, v in train_settings['transform_settings']['extrinsic_prior'].items():
        if v.lower() != 'default':
            extrinsic_prior_dict[k] = v
    ref_time = train_settings['transform_settings']['ref_time']
    detector_list = train_settings['transform_settings']['detectors']
    domain_dict = wfd.domain.domain_dict
    # build objects
    domain = build_domain(domain_dict)
    ifo_list = InterferometerList(detector_list)

    # build transforms
    sample_extrinsic_parameters = SampleExtrinsicParameters(extrinsic_prior_dict)
    get_detector_times = GetDetectorTimes(ifo_list, ref_time)
    project_onto_detectors = ProjectOntoDetectors(ifo_list, domain, ref_time)

    d0 = wfd[0]
    d1 = sample_extrinsic_parameters(d0)
    d2 = get_detector_times(d1)
    d3 = project_onto_detectors(d2)

    import matplotlib.pyplot as plt
    plt.plot(wfd[0]['waveform']['h_cross'].real / d3['parameters'][
        'luminosity_distance'] * 100)
    plt.plot(d3['waveform']['H1'].real)
    plt.show()

    ref_data = np.load('train_dir/waveform_data.npy', allow_pickle=True).item()
    sample_in = {'parameters': ref_data['intrinsic_parameters'],
                 'waveform': {'h_cross': ref_data['hc'],
                              'h_plus': ref_data['hp']},
                 'extrinsic_parameters': ref_data['extrinsic_parameters']}

    # check that we packaged the polarizations correctly
    assert sample_in.keys() == d1.keys()
    for k in sample_in.keys():
        assert sample_in[k].keys() == d1[k].keys()

    sample_out = get_detector_times(sample_in)
    sample_out = project_onto_detectors(sample_out)

    new = sample_out['waveform']['H1'].real
    ref = ref_data['h_d_unwhitened']['H1'].real
    deviation = new - ref

    plt.xlim((0,250))
    plt.xlabel('f in Hz')
    plt.title('strain.real in H1')
    plt.plot(domain(), ref.real, '.', label='research code')
    plt.plot(domain(), new.real, label='dingo code')
    # plt.plot(domain(), deviation, label='deviation')
    plt.legend()
    plt.show()

    plt.xlim((0,1024))
    plt.xlabel('f in Hz')
    plt.title('Deviation between research code and Dingo strain (H1)')
    plt.plot(deviation)
    plt.show()

    import h5py
    with h5py.File('train_dir/waveform_data.hdf5', 'r') as f:
        hp = f['hp'][:]
        hc = f['hc'][:]
        parameters_array = f['parameters'][:]
        h_H1 = f['h_H1_unwhitened'][:]
        h_L1 = f['h_L1_unwhitened'][:]
        param_idx = {'mass_1': 0, 'mass_2': 1, 'phase': 2, 'geocent_time': 3,
                     'luminosity_distance': 4, 'a_1': 5, 'a_2': 6, 'tilt_1': 7,
                     'tilt_2': 8, 'phi_12': 9, 'phi_jl': 10, 'theta_jn': 11,
                     'psi': 12, 'ra': 13, 'dec': 14,
                     'H1_time': 15, 'L1_time': 16}
        parameters = {k: parameters_array[v] for k,v in param_idx.items()}
        parameters['chirp_mass'] = component_masses_to_chirp_mass(
            parameters['mass_1'], parameters['mass_2'])
        parameters['mass_ratio'] = parameters['mass_2'] / parameters['mass_1']

    wfd._waveform_polarizations['h_plus'][0] = hp @ wfd._Vh.T.conj()
    wfd._waveform_polarizations['h_cross'][0] = hc @ wfd._Vh.T.conj()
    for k in wfd._parameter_samples.keys():
        wfd._parameter_samples[k][0] = parameters[k]
    # Need to reset geocent time, as here it is still the reference time for
    # hp and hc, which is zero. The real hc is set for extrinsic parameters.
    wfd._parameter_samples['geocent_time'][0] = 0

    sample0 = wfd[0]

    plt.plot(hp.real)
    plt.plot(sample0['waveform']['h_plus'].real)
    plt.show()
    plt.plot(hp.imag)
    plt.plot(sample0['waveform']['h_plus'].imag)
    plt.show()
    # plt.plot(hc.real)
    # plt.plot(sample0['waveform']['h_cross'].real)
    # plt.show()
    # plt.plot(hc.imag)
    # plt.plot(sample0['waveform']['h_cross'].imag)
    # plt.show()


    sample1 = sample_extrinsic_parameters(sample0)
    # enforce same extrinsic parameters as in ref sample
    for k in sample1['extrinsic_parameters'].keys():
        sample1['extrinsic_parameters'][k] = parameters[k]
    sample2 = get_detector_times(sample1)
    sample3 = project_onto_detectors(sample2)

    # check that detector time shifts are the same
    ratio_H1 = sample2['extrinsic_parameters']['H1_time']/parameters['H1_time']
    ratio_L1 = sample2['extrinsic_parameters']['L1_time']/parameters['L1_time']
    # should not be large, but also non-zero, since different routines are used.
    assert 0 < abs(ratio_H1-1) < 1e-5
    assert 0 < abs(ratio_L1-1) < 1e-5

    # check that detector projections are the same
    plt.xlim((0,250))
    plt.xlabel('f in Hz')
    plt.title('strain.real in H1')
    plt.plot(domain(), h_H1.real, '.', label='research code')
    plt.plot(domain(), sample3['waveform']['H1'].real, label='dingo code')
    plt.legend()
    plt.show()

    plt.ylim((0,1))
    plt.plot(np.abs(h_H1 - sample3['waveform']['H1']) / np.abs(h_H1))
    plt.show()

    print('done')
