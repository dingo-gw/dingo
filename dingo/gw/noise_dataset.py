import h5py
import numpy as np
import ast

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
                                          'delta_f': delta_f}})
        settings['domain_dict'] = domain.domain_dict
        settings['tukey_window'] = meta['tukey_window']
        f.create_dataset(f'asds_{ifo}', data=asds)
        gps_times[ifo] = meta['start_times']

    f.attrs['metadata'] = str(settings)
    f.close()

    AD = ASDDataset(join(data_dir, f'asds_{run}.hdf5'))
    asd_samples = AD.sample_random_asds()

    print('done')