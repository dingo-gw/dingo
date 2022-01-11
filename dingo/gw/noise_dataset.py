import h5py
import numpy as np
from scipy.signal import tukey
import ast
from dingo.gw.domains import build_domain
from dingo.gw.gwutils import *

class ASDDataset:
    """
    Dataset of amplitude spectral densities (ASDs). The ASDs are typically
    used for whitening strain data, and additionally passed as context to the
    neural density estimator.
    """
    def __init__(self, dataset, ifos=None):
        """
        Initializes an ASD dataset by loading the corresponding hdf5 file.

        :param dataset: Path of the hdf5 dataset file.
        :param ifos:    List of detectors used for dataset, e.g. ['H1', 'L1'].
                        If not set, all available ones in the dataset are used.
        """
        with h5py.File(dataset, 'r') as f:
            if ifos is None:
                ifos = [k[5:] for k in f.keys() if k.startswith('asds_')]
            # load asds
            self.asds = {ifo: f[f'asds_{ifo}'][:] for ifo in ifos}
            # gps times are potentially relevant for continual learning tasks,
            # where one might want to only use ASDs in a particular time range
            try:
                self.gps_times = {ifo: f['gps_times'][ifo][:] for ifo in ifos}
            except KeyError:
                self.gps_times = {ifo: -1 for ifo in ifos}
            self.metadata = ast.literal_eval(f.attrs['metadata'])

        self.domain = build_domain(self.metadata['domain_dict'])
        self.is_truncated = False

    def truncate_dataset_domain(self, new_range = None):
        """
        The asd dataset provides asds in the range [0, domain._f_max]. In
        practice one may want to apply data conditioning different to that of
        the dataset by specifying a different range, and truncating this
        dataset accordingly.

        This method provides functionality for that. It truncates the dataset
        to the range specified by the domain, by calling domain.truncate_data.
        In uniform FD, this corresponds to truncating data in the range
        [0, domain._f_max] to the range [domain._f_min, domain._f_max].

        Before this truncation step, one may optionally modify the domain,
        to set a new range. This is done by domain.set_new_range(*new_range),
        which is called if new_range is not None.
        """
        if self.is_truncated:
            raise ValueError('Dataset is already truncated')
        len_domain_original = len(self.domain)

        # optionally set new data range the dataset
        if new_range is not None:
            self.domain.set_new_range(*new_range)

        # truncate the dataset
        for ifo, asds in self.asds.items():
            assert asds.shape[-1] == len_domain_original, \
                f'ASDs with shape {self._Vh.shape} are not compatible' \
                f'with the domain of length {len_domain_original}.'
            self.asds[ifo] = self.domain.truncate_data(
                asds, allow_for_flexible_upper_bound=(new_range is not None))

        self.is_truncated = True

    def sample_random_asds(self):
        """
        Sample a random asd for each detector.
        :return: Dict with a random asd from the dataset for each detector.
        """
        return {k: v[np.random.choice(len(v), 1)[0]]
                for k,v in self.asds.items()}

if __name__ == '__main__':
    # this transforms the np psd datasets from the research code to the hdf5
    # datasets used by dingo
    from os.path import join
    num_psds_max = None
    run = 'O1'
    ifos = ['H1', 'L1']
    data_dir = '../../../data/PSDs'
    f = h5py.File(join(data_dir, f'asds_{run}.hdf5'), 'w')
    gps_times = f.create_group('gps_times')
    settings = {}


    for ifo in ifos:
        psds = np.load(join(data_dir, f'{run}_{ifo}_psd.npy'))
        meta = np.load(join(data_dir, f'{run}_{ifo}_metadata.npy'),
                       allow_pickle=True).item()
        freqs = meta['sample_frequencies']
        scale_factor = meta['scale_factor']
        asds = np.sqrt(psds) / np.sqrt(scale_factor)
        f_min, f_max, delta_f = freqs[0], freqs[-1], freqs[1] - freqs[0]
        domain = build_domain({'name': 'UniformFrequencyDomain',
                               'kwargs': {'f_min': f_min, 'f_max': f_max,
                                          'delta_f': delta_f}})
        settings['domain_dict'] = domain.domain_dict
        # settings['window'] = {'window_type': 'tukey', **meta['tukey_window']}
        f.create_dataset(f'asds_{ifo}', data=asds[:num_psds_max])
        gps_times[ifo] = meta['start_times'][:num_psds_max]

    f.attrs['metadata'] = str(settings)
    f.close()

    asd_dataset = ASDDataset(join(data_dir, f'asds_{run}.hdf5'))
    asd_samples = asd_dataset.sample_random_asds()

    window_factor = get_window_factor({'window_type': 'tukey',
                                       **meta['tukey_window']})

    noise_std = np.sqrt(window_factor) / \
                np.sqrt(4*asd_dataset.metadata['domain_dict']['kwargs']['delta_f'])

    domain = build_domain(asd_dataset.metadata['domain_dict'])

    print(window_factor)
    print(noise_std**2)
    print(noise_std)

    # print(domain.window_factor)
    # print(domain.noise_std**2)

    print('done')