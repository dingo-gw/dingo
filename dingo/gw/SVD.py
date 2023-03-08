import numpy as np
import pandas as pd
import scipy
from sklearn.utils.extmath import randomized_svd
from dingo.core.dataset import DingoDataset


class SVDBasis(DingoDataset):

    dataset_type = "svd_basis"

    def __init__(
        self,
        file_name=None,
        dictionary=None,
    ):
        self.V = None
        self.Vh = None
        self.s = None
        self.n = None
        self.mismatches = None
        super().__init__(
            file_name=file_name,
            dictionary=dictionary,
            data_keys=["V", "s", "mismatches"],
        )

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

    def compute_test_mismatches(
        self,
        data: np.ndarray,
        parameters: pd.DataFrame = None,
        increment: int = 50,
        verbose: bool = False,
    ):
        """
        Test SVD basis by computing mismatches of compressed / decompressed data
        against original data. Results are saved as a DataFrame.

        Parameters
        ----------
        data : np.ndarray
            Array of data sets to validate against.
        parameters : pd.DataFrame
            Optional labels for the data sets. This is useful for checking performance on
            particular regions of the parameter space.
        increment : int
            Specifies SVD truncations for computing mismatches. E.g., increment = 50
            means that the SVD will be truncated at size [50, 100, 150, ..., len(data)].
        verbose : bool
            Whether to print summary statistics.
        """
        if len(data) != len(parameters):
            raise ValueError(
                f"Incompatible data: len(data) == {len(data)} and len("
                f"parameters) == {len(parameters)} do not match."
            )
        if parameters is not None:
            self.mismatches = parameters.copy()
        else:
            self.mismatches = pd.DataFrame()

        for n in np.append(np.arange(increment, self.n, increment), self.n):
            mismatches = np.empty(len(data))
            for i, d in enumerate(data):
                compressed = d @ self.V[:, :n]
                reconstructed = compressed @ self.Vh[:n]
                norm1 = np.sqrt(np.sum(np.abs(d) ** 2))
                norm2 = np.sqrt(np.sum(np.abs(reconstructed) ** 2))
                inner = np.sum(d.conj() * reconstructed).real
                mismatches[i] = 1 - inner / (norm1 * norm2)
            self.mismatches[f"mismatch n={n}"] = mismatches

        if verbose:
            self.print_validation_summary()

    def print_validation_summary(self):
        """
        Print a summary of the validation mismatches.
        """
        if self.mismatches is not None:
            for col in self.mismatches:
                if "mismatch" in col:
                    n = int(col.split(sep="=")[-1])
                    mismatches = self.mismatches[col]
                    print(f"n = {n}")
                    print("  Mean mismatch = {}".format(np.mean(mismatches)))
                    print("  Standard deviation = {}".format(np.std(mismatches)))
                    print("  Max mismatch = {}".format(np.max(mismatches)))
                    print("  Median mismatch = {}".format(np.median(mismatches)))
                    print("  Percentiles:")
                    print("    99    -> {}".format(np.percentile(mismatches, 99)))
                    print("    99.9  -> {}".format(np.percentile(mismatches, 99.9)))
                    print("    99.99 -> {}".format(np.percentile(mismatches, 99.99)))

    def decompress(self, coefficients: np.ndarray):
        """
        Convert from basis coefficients back to raw data representation.

        Parameters
        ----------
        coefficients : np.ndarray
            Array of basis coefficients

        Returns
        -------
        array of decompressed data
        """
        return coefficients @ self.Vh

    def compress(self, data: np.ndarray):
        """
        Convert from data (e.g., frequency series) to compressed representation in
        terms of basis coefficients.

        Parameters
        ----------
        data : np.ndarray

        Returns
        -------
        array of basis coefficients
        """
        return data @ self.V

    def from_file(self, filename):
        """
        Load the SVD basis from a HDF5 file.

        Parameters
        ----------
        filename : str
        """
        super().from_file(filename)
        if self.V is None:
            raise KeyError("File does not contain SVD V matrix. No SVD basis to load.")
        self.Vh = self.V.T.conj()
        self.n = self.V.shape[1]

    def from_dictionary(self, dictionary: dict):
        """
        Load the SVD basis from a dictionary.

        Parameters
        ----------
        dictionary : dict
            The dictionary should contain at least a 'V' key, and optionally an 's' key.
        """
        super().from_dictionary(dictionary)
        if self.V is None:
            raise KeyError("dict does not contain SVD V matrix. No SVD basis to load.")
        self.Vh = self.V.T.conj()
        self.n = self.V.shape[1]

    # def truncate(self, n: int):
    #     """
    #     Truncate size of SVD.
    #
    #     Parameters
    #     ----------
    #     n : int
    #         New SVD size. Should be less than current size.
    #     """
    #     if n > self.n or n < 0:
    #         print(f"Cannot truncate SVD from size n={self.n} to n={n}.")
    #     else:
    #         self.V = self.V[:, :n]
    #         self.Vh = self.Vh[:n, :]
    #         self.s = self.s[:n]
    #         self.n = n


class ApplySVD(object):
    """Transform operator for applying an SVD compression / decompression."""

    def __init__(self, svd_basis: SVDBasis, inverse: bool = False):
        """
        Parameters
        ----------
        svd_basis : SVDBasis
        inverse : bool
            Whether to apply for the forward (compression) or inverse (decompression)
            transform. Default: False.
        """
        self.svd_basis = svd_basis
        self.inverse = inverse

    def __call__(self, waveform: dict):
        """
        Parameters
        ----------
        waveform : dict
            Values should be arrays containing waveforms to be transformed.

        Returns
        -------
        dict of the same form as the input, but with transformed waveforms.
        """
        if not self.inverse:
            func = self.svd_basis.compress
        else:
            func = self.svd_basis.decompress
        return {k: func(v) for k, v in waveform.items()}
