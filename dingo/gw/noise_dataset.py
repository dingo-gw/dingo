import h5py
import numpy as np
from scipy.signal import tukey
import ast
from dingo.gw.gwutils import *

class ASDDataset:
    def __init__(self, dataset, ifos=None):
        with h5py.File(dataset, 'r') as f:
            if ifos is None:
                ifos = [k[5:] for k in f.keys() if k.startswith('asds_')]
            # load asds
            self.asds = {ifo: f[f'asds_{ifo}'][:] for ifo in ifos}
            # gps times are potentially relevant for continual learning tasks,
            # where one might want to only use ASDs in a particular time range
            self.gps_times = {ifo: f['gps_times'][ifo][:] for ifo in ifos}
            self.metadata = ast.literal_eval(f.attrs['metadata'])

    def truncate_dataset_range(self):
        raise NotImplementedError

    def sample_random_asds(self):
        return {k: v[np.random.choice(len(v), 1)[0]]
                for k,v in self.asds.items()}

if __name__ == '__main__':
    # this transforms the np psd datasets from the research code to the hdf5
    # datasets used by dingo
    from os.path import join
    from dingo.gw.domains import build_domain
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
                                          'delta_f': delta_f,
                                          'window_kwargs': {
                                              'window_type': 'tukey',
                                              **meta['tukey_window']}
                                          },
                               })
        settings['domain_dict'] = domain.domain_dict
        # settings['window'] = {'window_type': 'tukey', **meta['tukey_window']}
        f.create_dataset(f'asds_{ifo}', data=asds)
        gps_times[ifo] = meta['start_times']

    f.attrs['metadata'] = str(settings)
    f.close()

    asd_dataset = ASDDataset(join(data_dir, f'asds_{run}.hdf5'))
    asd_samples = asd_dataset.sample_random_asds()

    window_factor = get_window_factor(
        asd_dataset.metadata['domain_dict']['kwargs']['window_kwargs'])

    noise_std = np.sqrt(window_factor) / \
                np.sqrt(4*asd_dataset.metadata['domain_dict']['kwargs']['delta_f'])

    from dingo.gw.domains import build_domain
    domain = build_domain(asd_dataset.metadata['domain_dict'])

    print(window_factor)
    print(noise_std**2)

    print(domain.window_factor)
    print(domain.noise_std**2)

    print('done')