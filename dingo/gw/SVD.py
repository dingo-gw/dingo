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

    def generate_basis(self, training_data: np.ndarray, n: int, method: str = "random"):
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
        if method == "random":
            if n == 0:
                n = min(training_data.shape)

            U, s, Vh = randomized_svd(training_data, n, random_state=0)

            self.Vh = Vh.astype(np.complex128)  # TODO: fix types
            self.V = self.Vh.T.conj()
            self.n = n
            self.s = s
        elif method == "scipy":
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
            raise ValueError(f"Unsupported SVD method: {method}.")

    def test_basis(
        self, test_data, n_values=(50, 100, 128, 200, 300, 500, 600), outfile=None
    ):
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
        test_stats = {"s": self.s, "mismatches": {}, "max_deviations": {}}
        for n in n_values:
            if n > self.n:
                continue
            matches = []
            max_deviations = []
            for h_test in test_data:
                h_RB = h_test @ self.V[:, :n]
                h_reconstructed = h_RB @ self.Vh[:n]
                norm1 = np.mean(np.abs(h_test) ** 2)
                norm2 = np.mean(np.abs(h_reconstructed) ** 2)
                inner = np.mean(h_test.conj() * h_reconstructed).real
                matches.append(inner / np.sqrt(norm1 * norm2))
                max_deviations.append(np.max(np.abs(h_test - h_reconstructed)))
            mismatches = 1 - np.array(matches)

            test_stats["mismatches"][n] = mismatches
            test_stats["max_deviations"][n] = max_deviations

            print(f"n = {n}")
            print("  Mean mismatch = {}".format(np.mean(mismatches)))
            print("  Standard deviation = {}".format(np.std(mismatches)))
            print("  Max mismatch = {}".format(np.max(mismatches)))
            print("  Median mismatch = {}".format(np.median(mismatches)))
            print("  Percentiles:")
            print("    99    -> {}".format(np.percentile(mismatches, 99)))
            print("    99.9  -> {}".format(np.percentile(mismatches, 99.9)))
            print("    99.99 -> {}".format(np.percentile(mismatches, 99.99)))

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

    def from_V(self, V):
        self.V = V
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


class ApplySVD(object):
    def __init__(self, svd_basis: SVDBasis):
        self.svd_basis = svd_basis

    def __call__(self, uncompressed_waveform: dict):
        compressed_waveform = {
            k: self.svd_basis.fseries_to_basis_coefficients(v)
            for k, v in uncompressed_waveform.items()
        }
        return compressed_waveform


class UndoSVD(object):
    def __init__(self, svd_basis: SVDBasis):
        self.svd_basis = svd_basis

    def __call__(self, compressed_waveform: dict):
        uncompressed_waveform = {
            k: self.svd_basis.basis_coefficients_to_fseries(v)
            for k, v in compressed_waveform.items()
        }
        return uncompressed_waveform
