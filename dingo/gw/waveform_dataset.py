import ast
from typing import Dict, Union

import h5py
import numpy as np
import pandas as pd
import scipy
from sklearn.utils.extmath import randomized_svd
from torchvision.transforms import Compose
from torch.utils.data import Dataset, DataLoader
from os.path import join
import time

from dingo.gw.domains import build_domain


class SVDBasis:
    def __init__(self):
        self.V = None
        self.Vh = None
        self.n = None

    def generate_basis(self, training_data: np.ndarray, n: int,
                       method: str = 'random'):
        """Generate the SVD basis from training data and store it.

        The SVD decomposition takes

        training_data = U @ diag(s) @ Vh

        where U and Vh are unitary.

        Parameters
        ----------
        training_data: np.ndarray
            Array of waveform data on the physical domain

        n: int
            Number of basis elements to keep.
            n=0 keeps all basis elements.
        method: str
            Select SVD method, 'random' or 'scipy'
        """
        if method == 'random':
            if n == 0:
                n = min(training_data.shape)

            U, s, Vh = randomized_svd(training_data, n, random_state=0)

            self.Vh = Vh.astype(np.complex64)
            self.V = self.Vh.T.conj()
            self.n = n
            self.s = s
        elif method == 'scipy':
            # Code below uses scipy's svd tool. Likely slower.
            U, s, Vh = scipy.linalg.svd(training_data, full_matrices=False)
            V = Vh.T.conj()

            if (n == 0) or (n > len(V)):
                self.V = V
                self.Vh = Vh
            else:
                self.V = V[:, :n]
                self.Vh = Vh[:n, :]

            self.n = len(self.Vh)
            self.s = s
        else:
            raise ValueError(f'Unsupported SVD method: {method}.')

    def test_basis(self, test_data, n_values=(50, 100, 200, 300, 500, 600, 800),
                   outfile=None):
        """
        Test basis by computing mismatches of original waveforms in test_data
        with reconstructed waveforms.

        Parameters
        ----------
        test_data:
            Array with test_data
        n_values:
            Iterable with values for n used to test reduced basis
        outfile:
            Save test_stats to outfile if not None
        """
        test_stats = {'s': self.s, 'mismatches': {}, 'max_deviations': {}}
        for n in n_values:
            if n > self.n: continue
            matches = []
            max_deviations = []
            for h_test in test_data:
                h_RB = h_test @ self.V[:,:n]
                h_reconstructed = h_RB @ self.Vh[:n]
                norm1 = np.mean(np.abs(h_test) ** 2)
                norm2 = np.mean(np.abs(h_reconstructed) ** 2)
                inner = np.mean(h_test.conj() * h_reconstructed).real
                matches.append(inner / np.sqrt(norm1 * norm2))
                max_deviations.append(np.max(np.abs(h_test - h_reconstructed)))
            mismatches = 1 - np.array(matches)

            test_stats['mismatches'][n] = mismatches
            test_stats['max_deviations'][n] = max_deviations

            print(f'n = {n}')
            print('  Mean mismatch = {}'.format(np.mean(mismatches)))
            print('  Standard deviation = {}'.format(np.std(mismatches)))
            print('  Max mismatch = {}'.format(np.max(mismatches)))
            print('  Median mismatch = {}'.format(np.median(mismatches)))
            print('  Percentiles:')
            print('    99    -> {}'.format(np.percentile(mismatches, 99)))
            print('    99.9  -> {}'.format(np.percentile(mismatches, 99.9)))
            print('    99.99 -> {}'.format(np.percentile(mismatches, 99.99)))

        if outfile is not None:
            np.save(outfile, test_stats, allow_pickle=True)

    def basis_coefficients_to_fseries(self, coefficients: np.ndarray):
        """
        Convert from basis coefficients to frequency series.

        Parameters
        ----------
        coefficients:
            Array of basis coefficients
        """
        return coefficients @ self.Vh

    def fseries_to_basis_coefficients(self, fseries: np.ndarray):
        """
        Convert from frequency series to basis coefficients.

        Parameters
        ----------
        fseries:
            Array of frequency series
        """
        return fseries @ self.V

    def from_file(self, filename: str):
        """
        Load basis matrix V from a file.

        Parameters
        ----------
        filename:
            File in .npy format
        """
        self.V = np.load(filename)
        self.Vh = self.V.T.conj()
        self.n = self.V.shape[1]

    def to_file(self, filename: str):
        """
        Save basis matrix V to a file.

        Parameters
        ----------
        filename:
            File in .npy format
        """
        if self.V is not None:
            np.save(filename, self.V)


class WaveformDataset(Dataset):
    """This class loads a dataset of simulated waveforms (plus and cross
    polarizations, as well as associated parameter values.

    This class loads a stored set of waveforms from an HDF5 file.
    Waveform polarizations are generated by the scripts in
    gw.waveform_dataset_generation.
    Once a waveform data set is in memory, the waveform data are consumed through
    a __getitem__() call, optionally applying a chain of transformations, which
    are classes that implement the __call__() method.
    """

    def __init__(self, dataset_file: str, transform=None,
                 single_precision=True):
        """
        Parameters
        ----------
        dataset_file : str
            Load the waveform dataset from this HDF5 file.
        transform :
            Transformations to apply.
        """
        self.transform = transform
        self._Vh = None
        self.single_precision = single_precision
        self.load(dataset_file)


    def __len__(self):
        """The number of waveform samples."""
        return len(self._parameter_samples)


    def __getitem__(self, idx) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """
        Return a nested dictionary containing parameters and waveform polarizations
        for sample with index `idx`. If defined a chain of transformations are being
        applied to the waveform data.
        """
        parameters = self._parameter_samples.iloc[idx].to_dict()
        waveform_polarizations = {'h_cross': self._hc[idx],
                                  'h_plus': self._hp[idx]}
        data = {'parameters': parameters, 'waveform': waveform_polarizations}
        if self._Vh is not None:
            data['waveform']['h_plus'] = data['waveform']['h_plus'] @ self._Vh
            data['waveform']['h_cross'] = data['waveform']['h_cross'] @ self._Vh
        if self.transform:
            data = self.transform(data)
        return data


    def get_info(self):
        """
        Print information on the stored pandas DataFrames.
        This is before any transformations are done.
        """
        self._parameter_samples.info(memory_usage='deep')
        self._hc.info(memory_usage='deep')
        self._hp.info(memory_usage='deep')


    def load(self, filename: str = 'waveform_dataset.h5'):
        """
        Load waveform data set from HDF5 file.

        Parameters
        ----------
        filename : str
            The name of the HDF5 file containing the data set.
        """
        fp = h5py.File(filename, 'r')

        parameter_array = fp['parameters'][:]
        self._parameter_samples = pd.DataFrame(parameter_array)

        grp = fp['waveform_polarizations']
        assert list(grp.keys()) == ['h_cross', 'h_plus']
        self._hc = grp['h_cross'][:]
        self._hp = grp['h_plus'][:]

        if 'rb_matrix_V' in fp.keys():
            V = fp['rb_matrix_V'][:]
            self._Vh = V.T.conj()

        self.data_settings = ast.literal_eval(fp.attrs['settings'])
        self.domain = build_domain(self.data_settings['domain_settings'])
        self.is_truncated = False

        fp.close()

        # set requested datatype; if dtype is different for _hc/_hp and _Vh,
        # __getitem__() becomes super slow
        dtype = np.complex64 if self.single_precision else np.complex128
        self._hc = np.array(self._hc, dtype=dtype)
        self._hp = np.array(self._hp, dtype=dtype)
        self._Vh = np.array(self._Vh, dtype=dtype)



    def truncate_dataset_domain(self, new_range = None):
        """
        The waveform dataset provides waveforms polarizations in a particular
        range. In uniform Frequency domain for instance, this range is
        [0, domain._f_max]. In practice one may want to apply data conditioning
        different to that of the dataset by specifying a different range,
        and truncating this dataset accordingly. That corresponds to
        truncating the likelihood integral.

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
        if self._Vh is not None:
            assert self._Vh.shape[-1] == len_domain_original, \
                f'Compression matrix Vh with shape {self._Vh.shape} is not ' \
                f'compatible with the domain of length {len_domain_original}.'
            self._Vh = self.domain.truncate_data(
                self._Vh, allow_for_flexible_upper_bound=(new_range is not
                                                          None))
        else:
            raise NotImplementedError('Truncation of the dataset is currently '
                                      'only implemented for compressed '
                                      'polarization data.')

        self.is_truncated = True


def generate_and_save_reduced_basis(wfd,
                                    omitted_transforms,
                                    N_train = 50_000,
                                    N_test = 10_000,
                                    batch_size = 1000,
                                    num_workers = 0,
                                    n_rb = 200,
                                    out_dir = None,
                                    suffix = ''
                                    ):
    """
    Generate a reduced basis (rb) for network initialization. To that end,
    a set of non-noisy waveforms from wfd are sampled (important: noise
    addition and strain repackaging transforms need to be omitted!), and an SVD
    basis is build based on these.

    :param wfd: dingo.gw.waveform_dataset.WaveformDataset
        waveform dataset for rb generation
    :param omitted_transforms: transforms
        transforms to be omitted for rb generation
    :param N_train: int
        size of training set for rb
    :param N_test: int
        size of test set for rb
    :param batch_size: int
        batch size for dataloader
    :param num_workers: int
        number of workers for dataloader
    :param n_rb: int
        number of rb basis elements
    :param out_dir: str
        directory for saving rb matrices
    :param suffix: str
        suffix for names of saved rb matrices V
    :return: V_paths
        paths to saved rb matrices V
    """
    wfd_transform_original = wfd.transform
    transforms_rb = [tr for tr in wfd.transform.transforms if type(tr) not in
                     omitted_transforms]
    wfd.transform = Compose(transforms_rb)
    ifos = list(wfd[0][1].keys())
    M = len(wfd[0][1][ifos[0]])

    # get training data for reduced basis
    N = N_train + N_test
    print('Collecting data for reduced basis generation.', end=' ')
    time_start = time.time()
    rb_train_data = {ifo: np.empty((N, M), dtype=np.complex128) for ifo in ifos}
    rb_loader = DataLoader(wfd, batch_size=batch_size, num_workers=num_workers)
    for idx, data in enumerate(rb_loader):
        strain_data = data[1]
        lower = idx * batch_size
        n = min(batch_size, N - lower)
        for k in rb_train_data.keys():
            rb_train_data[k][lower:lower+n] = strain_data[k][:n]
        if lower + n == N:
            break
    print(f'Done. This took {time.time() - time_start:.0f} s.\n')

    # generate rb
    print('Generating SVD basis for ifo:')
    time_start = time.time()
    basis_dict = {}
    for ifo in rb_train_data.keys():
        print(ifo, end=' ')
        basis = SVDBasis()
        basis.generate_basis(rb_train_data[ifo][:N_train], n_rb)
        basis_dict[ifo] = basis
        print('done')
    print(f'This took {time.time() - time_start:.0f} s.\n')

    # save rb
    V_paths = []
    if out_dir is not None:
        print(f'Saving SVD basis matrices to {out_dir}', end=' ')
        for ifo in basis_dict:
            basis_dict[ifo].to_file(join(out_dir, f'V_{ifo}{suffix}.npy'))
            np.save(join(out_dir, f's_{ifo}{suffix}.npy'), basis_dict[ifo].s)
            V_paths.append(join(out_dir, f'V_{ifo}{suffix}.npy'))
    print('Done')

    # test rb
    if out_dir is not None:
        print(f'Testing SVD basis matrices, saving stats to {out_dir}')
        for ifo in basis_dict:
            basis_dict[ifo].test_basis(
                rb_train_data[ifo][N_train:],
                outfile=join(out_dir, f'V_{ifo}{suffix}_stats.npy'))
    print('Done')

    # set wfd.transform to original transform
    wfd.transform = wfd_transform_original

    return V_paths