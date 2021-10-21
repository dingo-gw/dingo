import yaml

from bilby.gw.detector import InterferometerList

from dingo.gw.waveform_dataset import WaveformDataset
from dingo.gw.prior_split import BBHExtrinsicPriorDict, default_extrinsic_dict
from dingo.gw.domains import build_domain

wfd_path = '/Users/mdax//Documents/dingo/devel/dingo-devel/tutorials/02_gwpe' \
           '/datasets/waveforms/02_IMR_test/waveform_dataset.hdf5'

wfd = WaveformDataset(wfd_path)


class SampleExtrinsicParameters(object):
    """
    Sample extrinsic parameters and add them to sample in a separate dictionary.
    """
    def __init__(self, extrinsic_prior_dict):
        self.extrinsic_prior_dict = extrinsic_prior_dict
        self.prior = BBHExtrinsicPriorDict(extrinsic_prior_dict)

    def __call__(self, input_sample):
        sample = input_sample.copy()
        extrinsic_parameters = self.prior.sample()
        sample['extrinsic_parameters'] = extrinsic_parameters
        return sample

    @property
    def reproduction_dict(self):
        return {'extrinsic_prior_dict': self.extrinsic_prior_dict}


class GetDetectorTimes(object):
    """
    Compute the time shifts in the individual detectors based on the sky
    position (ra, dec), the geocent_time and the ref_time.
    """
    def __init__(self, ifo_list, ref_time):
        self.ifo_list = ifo_list
        self.ref_time = ref_time

    def __call__(self, input_sample):
        sample = input_sample.copy()
        ra = sample['extrinsic_parameters']['ra']
        dec = sample['extrinsic_parameters']['dec']
        geocent_time = sample['extrinsic_parameters']['geocent_time']
        for ifo in self.ifo_list:
            dt = ifo.time_delay_from_geocenter(ra, dec, self.ref_time)
            ifo_time = geocent_time + dt
            sample['extrinsic_parameters'][f'{ifo.name}_time'] = ifo_time
        return sample


class ProjectOntoDetectors(object):
    """
    Project the GW polarizations onto the detectors in ifo_list. This does
    not sample any new parameters, but relies on the parameters provided in
    sample['extrinsic_parameters']. Specifically, this transform applies the
    following operations:

    (1) Rescale polarizations to account for sampled luminosity distance
    (2) Project polarizations onto the antenna patterns using the ref_time and
        the extrinsic parameters (ra, dec, psi)
    (3) Time shift the strains in the individual detectors according to the
        times <ifo.name>_time provided in the extrinsic parameters.
    """
    def __init__(self, ifo_list, domain, ref_time):
        self.ifo_list = ifo_list
        self.domain = domain
        self.ref_time = ref_time

    def __call__(self, input_sample):
        sample = input_sample.copy()
        try:
            d_ref = sample['parameters']['luminosity_distance']
            d_new = sample['extrinsic_parameters']['luminosity_distance']
            ra = sample['extrinsic_parameters']['ra']
            dec = sample['extrinsic_parameters']['dec']
            psi = sample['extrinsic_parameters']['psi']
            tc_ref = sample['parameters']['geocent_time']
            tc_new = sample['extrinsic_parameters']['geocent_time']
        except:
            raise ValueError('Missing parameters.')

        # (1) rescale polarizations and set distance parameter to sampled value
        hc = sample['waveform']['h_cross'] * d_ref / d_new
        hp = sample['waveform']['h_plus'] * d_ref / d_new
        sample['parameters']['luminosity_distance'] = d_new

        strains = {}
        for ifo in self.ifo_list:
            # (2) project strains onto the different detectors
            fp = ifo.antenna_response(ra, dec, self.ref_time, psi, mode='plus')
            fc = ifo.antenna_response(ra, dec, self.ref_time, psi, mode='cross')
            strain = fp * hp + fc * hc

            # (3) time shift the strain. If polarizations are timeshifted by
            #     tc_ref != 0, undo this here by subtracting it from dt.
            dt = sample['extrinsic_parameters'][f'{ifo.name}_time'] - tc_ref
            strains[ifo.name] = self.domain.time_translate_data(strain, dt)

        # add extrinsic extrinsic parameter corresponding to the
        # transformations applied in the loop above to parameters
        sample['parameters']['ra'] = ra
        sample['parameters']['dec'] = dec
        sample['parameters']['psi'] = psi
        sample['parameters']['geocent_time'] = tc_new

        sample['waveform'] = strains

        return sample


if __name__ == '__main__':
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
    print('done')
