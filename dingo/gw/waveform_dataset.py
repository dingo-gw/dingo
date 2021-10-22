import os
import ast
import pickle
from typing import Dict, Union, Tuple
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
import scipy
from sklearn.utils.extmath import randomized_svd
from torch.utils.data import Dataset
from tqdm import tqdm

from dingo.api import structured_array_from_dict_of_arrays
from dingo.gw.waveform_generator import WaveformGenerator
from dingo.gw.domains import build_domain
from bilby.gw.prior import BBHPriorDict


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

            U, s, Vh = randomized_svd(training_data, n)

            self.Vh = Vh.astype(np.complex64)
            self.V = self.Vh.T.conj()
            self.n = n
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
        else:
            raise ValueError(f'Unsupported SVD method: {method}.')

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
    """This class generates, saves, and loads a dataset of simulated waveforms
    (plus and cross polarizations, as well as associated parameter values.

    This class can generate waveform polarizations from scratch given a waveform
    generator and an intrinsic prior distribution. Alternatively it can load
    a stored set of waveforms from an HDF5 file.
    Once a waveform data set is in memory, the waveform data are consumed through
    a __getitem__() call, optionally applying a chain of transformations, which
    are classes that implement the __call__() method.
    """

    def __init__(self, dataset_file: str = None,
                 prior: BBHPriorDict = None,
                 waveform_generator: WaveformGenerator = None,
                 transform=None):
        """
        Parameters
        ----------
        dataset_file : str
            Load the waveform dataset from this HDF5 file.
        prior : BBHPriorDict
            The BBHPriorDict instance from which to draw parameter samples.
            It needs to contain intrinsic waveform parameters (and reference values as delta distributions)
            for generating waveform polarizations. Later we will also draw
            extrinsic parameters from a BBHExtrinsicPriorDict.
            If the prior is unspecified, sampling the prior is not supported,
            and read_parameter_samples() must be called before generate_dataset().
        waveform_generator : WaveformGenerator
            The waveform generator object to use to generate waveforms.
        transform :
            Transformations to apply.
        """
        self._prior = prior
        self._waveform_generator = waveform_generator
        if self._waveform_generator is not None:
            self.domain = self._waveform_generator.domain
        self._parameter_samples = None
        self._waveform_polarizations = None
        if dataset_file is not None:
            self.load(dataset_file)
        self.transform = transform

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
        waveform_polarizations = self._waveform_polarizations.iloc[idx].to_dict()
        data = {'parameters': parameters, 'waveform': waveform_polarizations}
        if '_Vh' in self.__dict__:
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
        self._waveform_polarizations.info(memory_usage='deep')
        # Possibly capture and save the output
        # import io
        # buffer = io.StringIO()
        # df.info(buf=buffer)
        # s = buffer.getvalue()
        # with open("df_info.txt", "w", encoding="utf-8") as f:
        #     f.write(s)

    def sample_prior(self, size: int = 0, add_reference_values: bool = True):
        """
        Draw (intrinsic) prior samples.

        Notes:
          * Since we're using bilby's prior classes we are automatically
            using numpy random number generator
          * The order of returned parameters is random
          * The fixed t_geocent and d_L reference values must be included in the prior.
            Make sure not to confuse these fixed values with samples drawn from a typical
            extrinsic distribution.

        Parameters
        ----------
        size : int
            The number of samples to draw.
        """
        parameter_samples_dict = self._prior.sample(size=size)
        self._parameter_samples = pd.DataFrame(parameter_samples_dict)


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
        polarization_dict_2d = {k: v[:] for k, v in grp.items()}
        polarization_dict = {k: [x for x in polarization_dict_2d[k]] for k in ['h_plus', 'h_cross']}
        self._waveform_polarizations = pd.DataFrame(polarization_dict)

        if 'rb_matrix_V' in fp.keys():
            V = fp['rb_matrix_V'][:]
            self._Vh = V.T.conj()

        self.data_settings = ast.literal_eval(fp.attrs['settings'])
        self.domain = build_domain(self.data_settings['domain_settings'])

        fp.close()

    def _write_dataframe_to_hdf5(self, fp: h5py.File,
                                 group_name: str,
                                 df: pd.DataFrame):
        """
        Write a DataFrame containing a dict of 2D arrays to HDF5.

        This creates a group containing as many 1D datasets
        as there are keys in the DataFrame. I.e. the stored
        format is a number of 1D arrays for parameters and
        2D arrays for waveform polarizations.

        Parameters
        ----------
        fp : h5py.File
            A h5py file object for the output file.
        group_name : str
            The name of the HDF5 group the data will be written to.
        df : pd.DataFrame
            A pandas DataFrame containing the data.
        """
        keys = list(df.keys())
        data = df.to_numpy().T
        grp = fp.create_group(group_name)
        for k, v in zip(keys, data):
            if v.dtype == np.dtype('O'):
                v = np.vstack(v)  # convert object array into a proper 2D array
            grp.create_dataset(str(k), data=v)

    def dataframe_to_structured_array(self, df: pd.DataFrame):
        """
        Convert a pandas DataFrame of parameters to a structured numpy array.

        Parameters
        ----------
        df:
            A pandas DataFrame
        """
        d = {k: np.array(list(v.values())) for k, v in df.to_dict().items()}
        return structured_array_from_dict_of_arrays(d)

    def save(self, filename: str = 'waveform_dataset.h5',
             parameter_fmt: str = 'structured_array',
             compress_data: bool = True, n_rb: int = 0):
        """
        Save waveform data set to a HDF5 file.

        Parameters
        ----------
        filename : str
            The name of the output HDF5 file.

        parameter_fmt : str
            Selects the way the waveform parameters are stored in the HDF5 file.
            'group': saves each parameter as a dataset in a group 'parameters'
            'structured_array': saves parameters as a 2D structured array with field names

        compress_data : bool
            If True project waveform polarizations onto an SVD basis.
            Save the projection coefficients and the SVD basis to HDF5.

        n_rb : int
            The number of basis functions at which the SVD should be truncated.
            n_rb = 0: no compression
        """
        fp = h5py.File(filename, 'w')

        if parameter_fmt == 'group':
            # This will create a group 'parameters' with a dataset for each parameter
            self._write_dataframe_to_hdf5(fp, 'parameters', self._parameter_samples)
        elif parameter_fmt == 'structured_array':
            # Alternative which converts to a structured array
            parameters_struct_arr = self.dataframe_to_structured_array(self._parameter_samples)
            # d = {k: np.array(list(v.values())) for k, v in self._parameter_samples.to_dict().items()}
            # parameters_struct_arr = structured_array_from_dict_of_arrays(d)
            fp.create_dataset('parameters', data=parameters_struct_arr)

        if compress_data:
            pol_arrays = {k: np.vstack(v.to_numpy().T) for k, v in self._waveform_polarizations.items()}
            basis = SVDBasis()
            basis.generate_basis(np.concatenate((pol_arrays['h_plus'],
                                                 pol_arrays['h_cross'])), n_rb)
            grp = fp.create_group('waveform_polarizations')
            for k, v in pol_arrays.items():
                h_proj = basis.fseries_to_basis_coefficients(v)
                grp.create_dataset(str(k), data=h_proj)
            fp.create_dataset('rb_matrix_V', data=basis.V)
        else:
            self._write_dataframe_to_hdf5(fp, 'waveform_polarizations', self._waveform_polarizations)

        fp.attrs['settings'] = str({'domain_settings': self.domain.domain_dict})

        fp.close()


if __name__ == "__main__":
    """Explore chaining transforms together and applying these to a waveform dataset."""
    from dingo.gw.domains import UniformFrequencyDomain
    from dingo.gw.detector_network import DetectorNetwork, RandomProjectToDetectors
    from dingo.gw.noise import AddNoiseAndWhiten
    from dingo.gw.prior_split import default_intrinsic_dict
    from transforms import StandardizeParameters, ToNetworkInput, Compose

    domain_kwargs = {'f_min': 20.0, 'f_max': 4096.0, 'delta_f': 1.0 / 4.0, 'window_factor': 1.0}
    domain = UniformFrequencyDomain(**domain_kwargs)
    priors = BBHPriorDict(default_intrinsic_dict)
    approximant = 'IMRPhenomXPHM'
    waveform_generator = WaveformGenerator(approximant, domain)

    # Generate a dataset using the first waveform dataset object
    wd = WaveformDataset(prior=priors, waveform_generator=waveform_generator)
    n_waveforms = 17
    wd.generate_dataset(size=n_waveforms)

    # 1. RandomProjectToDetectors  (single waveform)

    det_network = DetectorNetwork(["H1", "L1"], domain, start_time=priors.reference_geocentric_time)
    rp_det = RandomProjectToDetectors(det_network, priors)
    data = wd[1]
    # {'parameters': parameters, 'waveform': waveform_polarizations}
    strain_dict = rp_det(data)  # needs to take a dict instead
    print(strain_dict)  # {'H1':..., 'L1':...}

    # Example of applying projection of detectors as a transformation
    wd = WaveformDataset(prior=priors, waveform_generator=waveform_generator, transform=rp_det)
    n_waveforms = 17
    wd.generate_dataset(size=n_waveforms)

    # 2. AddNoiseAndWhiten (also standardizes data)  (single waveform)

    nw = AddNoiseAndWhiten(det_network)
    nw_strain_dict = nw(strain_dict)
    # plt.loglog(domain(), np.abs(strain_dict['waveform']['H1']), label='d')
    # plt.loglog(domain(), np.abs(nw_strain_dict['waveform']['H1']), label='d_w + n')
    # plt.show()


    # 3. StandardizeParameters (only standardizes parameters and leaves waveforms alone?) (single waveform)
    # Example: Define fake means and stdevs
    # TODO: Replace this with the correct means and variances for a particular prior distribution in each parameter
    #  - look at _compute_parameter_statistics(); this could be achieved by subclassing the bilby priors
    #  - in practise we do not need to get the means and variances used for standardization exactly right,
    #    as long as we make sure to store them, so that the transformation can be undone
    # For reference parameter values could just put the fixed value as a mean and stdev = 1.
    mu_dict = {'phi_jl': 1.0, 'tilt_1': 1.0, 'theta_jn': 2.0, 'tilt_2': 1.0, 'mass_1': 54.0, 'phi_12': 0.5,
               'chirp_mass': 40.0, 'phase': np.pi, 'a_2': 0.5, 'mass_2': 39.0, 'mass_ratio': 0.5,
               'a_1': 0.5, 'f_ref': 20.0, 'luminosity_distance': 1000.0, 'geocent_time': 1126259642.413,
               'ra': 2.5, 'dec': 1.0, 'psi': np.pi}
    std_dict = mu_dict

    # Chaining transforms
    transform = Compose([
        RandomProjectToDetectors(det_network, priors),
        AddNoiseAndWhiten(det_network),
        StandardizeParameters(mu=mu_dict, std=std_dict),
        ToNetworkInput(domain)
    ])

    wd = WaveformDataset(prior=priors, waveform_generator=waveform_generator, transform=transform)
    n_waveforms = 17
    wd.generate_dataset(size=n_waveforms)
    print(wd[9])
    # Raises an AttributeError as RandomProjectToDetectors does not have an inverse
    #err = {k: np.abs(transform.inverse(wd[9])['parameters'][k] - wd[9]['parameters'][k]) for k in mu_dict.keys()}

   # 4. ToTensor / ToNetworkInput (which formats the data appropriately for input to the network) (single waveform)
   # Output the dimensions so that network pars can be set
    tni = ToNetworkInput(domain)
    x_shape, y_shape = tni.get_output_dimensions(nw_strain_dict)
    print(x_shape, y_shape)
    xy = tni(nw_strain_dict)
    #print(xy)

