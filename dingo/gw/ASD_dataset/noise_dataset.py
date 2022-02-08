import copy

import h5py
import numpy as np
from scipy.signal import tukey
import ast
from dingo.gw.domains import build_domain
from dingo.gw.gwutils import *
from dingo.gw.dataset import DingoDataset

HIGH_ASD_VALUE = 1.0


class ASDDataset(DingoDataset):
    """
    Dataset of amplitude spectral densities (ASDs). The ASDs are typically
    used for whitening strain data, and additionally passed as context to the
    neural density estimator.
    """

    def __init__(self, file_name=None, dictionary=None, ifos=None, domain_update=None):
        """
        Parameters
        ----------
        file_name : str
            HDF5 file containing a dataset
        dictionary : dict
            Contains settings and data entries. The dictionary keys should be
            'settings', 'asds', and 'gps_times'.
        ifos : List[str]
            List of detectors used for dataset, e.g. ['H1', 'L1'].
            If not set, all available ones in the dataset are used.
        domain_update : dict
            If provided, update domain from existing domain using new settings.
        """

        super().__init__(
            file_name=file_name,
            dictionary=dictionary,
            data_keys=["asds", "gps_times"],
        )

        if ifos is not None:
            for ifo in list(self.asds.keys()):
                if ifo not in ifos:
                    self.asds.pop(ifo)

        self.domain = build_domain(self.settings["domain_dict"])
        if domain_update is not None:
            self.update_domain(domain_update)

    def update_domain(self, domain_update):
        """
        Update the domain based on new configuration. Also adjust data arrays to match
        the new domain.

        The ASD dataset provides ASDs in a particular domain. In Frequency domain,
        this is [0, domain._f_max]. In practice one may want to train a network based on
        slightly different domain settings, which corresponds to truncating the likelihood
        integral.

        This method provides functionality for that. It truncates the data below a
        new f_max, and sets the ASD below f_min to a large but finite value.

        Parameters
        ----------
        domain_update : dict
            Settings dictionary. Must contain a subset of the keys contained in
            domain_dict.
        """
        len_domain_original = len(self.domain)
        self.domain.update(domain_update)
        self.settings['domain'] = copy.deepcopy(self.domain.domain_dict)

        # truncate the dataset
        for ifo, asds in self.asds.items():

            # Is there a reason this check is needed? I would think that a dataset
            # should never be saved with this not matching.
            assert asds.shape[-1] == len_domain_original, (
                f"ASDs with shape {asds.shape[-1]} are not compatible"
                f"with the domain of length {len_domain_original}."
            )
            self.asds[ifo] = self.domain.update_data(
                asds,
                low_value=HIGH_ASD_VALUE,
            )

    def sample_random_asds(self):
        """
        Sample a random asd for each detector.
        :return: Dict with a random asd from the dataset for each detector.
        """
        return {k: v[np.random.choice(len(v), 1)[0]] for k, v in self.asds.items()}


# if __name__ == '__main__':
#     this transforms the np psd datasets from the research code to the hdf5
#     datasets used by dingo
#     from os.path import join
#     num_psds_max = None
#     run = 'O1'
#     ifos = ['H1', 'L1']
#     data_dir = '../../../data/PSDs'
#     f = h5py.File(join(data_dir, f'asds_{run}.hdf5'), 'w')
#     gps_times = f.create_group('gps_times')
#     settings = {}
#
#
#     for ifo in ifos:
#         psds = np.load(join(data_dir, f'{run}_{ifo}_psd.npy'))
#         meta = np.load(join(data_dir, f'{run}_{ifo}_metadata.npy'),
#                        allow_pickle=True).item()
#         freqs = meta['sample_frequencies']
#         scale_factor = meta['scale_factor']
#         asds = np.sqrt(psds) / np.sqrt(scale_factor)
#         f_min, f_max, delta_f = freqs[0], freqs[-1], freqs[1] - freqs[0]
#         domain = build_domain({'name': 'FrequencyDomain',
#                                'kwargs': {'f_min': f_min, 'f_max': f_max,
#                                           'delta_f': delta_f}})
#         settings['domain_dict'] = domain.domain_dict
#         # settings['window'] = {'window_type': 'tukey', **meta['tukey_window']}
#         f.create_dataset(f'asds_{ifo}', data=asds[:num_psds_max])
#         gps_times[ifo] = meta['start_times'][:num_psds_max]
#
#     f.attrs['metadata'] = str(settings)
#     f.close()
#
#     asd_dataset = ASDDataset(join(data_dir, f'asds_{run}.hdf5'))
#     asd_samples = asd_dataset.sample_random_asds()
#
#     window_factor = get_window_factor({'window_type': 'tukey',
#                                        **meta['tukey_window']})
#
#     noise_std = np.sqrt(window_factor) / \
#                 np.sqrt(4*asd_dataset.metadata['domain_dict']['kwargs']['delta_f'])
#
#     domain = build_domain(asd_dataset.metadata['domain_dict'])
#
#     print(window_factor)
#     print(noise_std**2)
#     print(noise_std)
#
#     # print(domain.window_factor)
#     # print(domain.noise_std**2)
#
#     print('done')
