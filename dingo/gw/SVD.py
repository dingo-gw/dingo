import time
from os.path import join

import numpy as np
import scipy
from sklearn.utils.extmath import randomized_svd
from torch.utils.data import DataLoader
from torchvision.transforms import Compose


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

    def test_basis(self, test_data, n_values=(50, 100, 128, 200, 300, 500, 600),
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


def build_svd():

    pass


def build_svd_old(num_chunks: int, outfile: str, rb_max: int = 0, train_fraction: float = 1.0):
    """
    Create and save SVD basis

    Parameters
    ----------
    num_chunks:
        Number of polarization data chunks the full parameter array was split into
    outfile:
        Output file for the SVD basis V matrix
    rb_max:
        Truncate the SVD at this size
    train_fraction:
        Fraction of data to be used for basis training, rest is for testing
    """
    logger.info('Load polarization data for all chunks ...')
    # load polarization data for h_plus and h_cross separately
    h_plus = np.vstack([load_polarizations_for_index(idx, compressed=False)[0]
                        for idx in tqdm(np.arange(num_chunks))])
    h_cross = np.vstack([load_polarizations_for_index(idx, compressed=False)[1]
                        for idx in tqdm(np.arange(num_chunks))])
    N_train = int(len(h_plus) * train_fraction)
    train_data = np.vstack((h_plus[:N_train], h_cross[:N_train]))
    test_data = np.vstack((h_plus[N_train:], h_cross[N_train:]))

    logger.info('Creating basis ...')
    basis = SVDBasis()
    basis.generate_basis(train_data, rb_max)
    basis.to_file(outfile)
    logger.info('Done.')
    logger.info('Testing basis ...')
    basis.test_basis(test_data,
                     outfile='.'.join(outfile.split('.')[:-1])+'_stats.npy')
    logger.info('Done.')
    return basis.n, basis.V.shape


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


class ApplySVD(object):

    def __init__(self, svd_basis: SVDBasis):
        self.svd_basis = svd_basis

    def __call__(self, uncompressed_waveform: dict):
        compressed_waveform = {k: self.svd_basis.fseries_to_basis_coefficients(v) for k, v in uncompressed_waveform.items()}
        return compressed_waveform


class UndoSVD(object):

    def __init__(self, svd_basis: SVDBasis):
        self.svd_basis = svd_basis

    def __call__(self, compressed_waveform: dict):
        uncompressed_waveform = {k: self.svd_basis.basis_coefficients_to_fseries(v) for k, v in compressed_waveform.items()}
        return uncompressed_waveform
