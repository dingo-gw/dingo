import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from torch.utils.data import Dataset

from dingo.gw.parameters import GWPriorDict
from dingo.gw.waveform_generator import WaveformGenerator


class WaveformDataset(Dataset):
    """This class generates, saves, and loads a dataset of simulated waveforms (plus and cross
    polarizations, as well as associated parameter values.

    There are two ways of calling this -- should this be split into two classes?
    1. call with wf generator and extrinsic parameters
    2. call with dataset file and a transform (or composition of transforms -- there is a pytorch way of doing this)
        - example of which transforms should be performed?

    Data will be consumed through __getitem__ through a chain of transforms
    TODO:
      * add compression
    """

    def __init__(self, dataset_file: str = None,
                 priors: GWPriorDict = None,
                 waveform_generator: WaveformGenerator = None,
                 transform=None):
        """
        Parameters
        ----------
        dataset_file : str
            Load the waveform dataset from this HDF5 file.
        priors : GWPriorDict
            The GWPriorDict instance from which to draw parameter samples.
            It needs to contain intrinsic waveform parameters (and reference values)
            for generating waveform polarizations. Later we will also draw
            extrinsic parameters from it.
        waveform_generator : WaveformGenerator
            The waveform generator object to use to generate waveforms.
        transform :
            Transformations to apply.
        """
        self._priors = priors
        self._waveform_generator = waveform_generator
        self._parameter_samples = None
        self._waveform_polarizations = None
        if dataset_file is not None:
            self.load(dataset_file)
        self.transform = transform

    def __len__(self):
        """The number of waveform samples."""
        return len(self._parameter_samples)

    def __getitem__(self, idx):
        """Return dictionary containing parameters and waveform polarizations
        for sample with index `idx`."""
        parameters = self._parameter_samples.iloc[idx].to_dict()
        waveform_polarizations = self._waveform_polarizations.iloc[idx].to_dict()
        data = {'parameters': parameters, 'waveform': waveform_polarizations}
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

    def generate_dataset(self, size: int = 0):
        """Generate a waveform dataset.

        Parameters
        ----------
        size : int
            The number of samples to draw and waveforms to generate.
        """
        # Draw intrinsic prior samples
        # Note: Since we're using bilby's prior classes we are automatically using numpy rng
        parameter_samples_dict = self._priors.sample_intrinsic(size=size)
        self._parameter_samples = pd.DataFrame(parameter_samples_dict)
        # The fixed f_ref and d_L reference values are added automatically in the call to sample_intrinsic.
        # In the case of d_L make sure not to confuse them with samples drawn from a typical extrinsic distribution.

        print('Generating waveform polarizations ...')  # Switch to logging
        # TODO: Currently, simple in memory generation of wfs on a single core; extend to multiprocessing or MPI
        #  For large datasets may not be able to store it in memory and need to write to disk while generating
        wf_list_of_dicts = [self._waveform_generator.generate_hplus_hcross(p.to_dict())
                            for _, p in tqdm(self._parameter_samples.iterrows())]
        polarization_dict = {k: [wf[k] for wf in wf_list_of_dicts] for k in ['h_plus', 'h_cross']}
        self._waveform_polarizations = pd.DataFrame(polarization_dict)

        # If safe downcast to float: converted_float = gl_float.apply(pd.to_numeric, downcast='float')
        # Look at WaveformDataset.generate_dataset()

    def load(self, filename: str = 'waveform_dataset.h5'):
        """
        Load waveform data set from HDF5 file.

        Parameters
        ----------
        filename : str
            The name of the HDF5 file containing the data set.
        """
        fp = h5py.File(filename, 'r')

        grp = fp['parameters']
        parameter_samples_dict = {k: v[:] for k, v in grp.items()}
        self._parameter_samples = pd.DataFrame(parameter_samples_dict)

        grp = fp['waveform_polarizations']
        polarization_dict_2D = {k: v[:] for k, v in grp.items()}
        polarization_dict = {k: [x for x in polarization_dict_2D[k]] for k in ['h_plus', 'h_cross']}
        self._waveform_polarizations = pd.DataFrame(polarization_dict)

        fp.close()

    def _write_datafrane_to_hdf5(self, fp: h5py.File,
                                 group_name: str,
                                 df: pd.DataFrame):
        """
        Write a DataFrame containing a dict of 2D arrays to HDF5.

        Parameters
        ----------
        fp : h5py.File
            A h5py file object for the output file.
        group_name : str
            The name of the HDF5 group the data will be written to.
        df : pd.DataFrame
            A pandas DataFrame containing thw data.
        """
        keys = list(df.keys())
        data = df.to_numpy().T
        grp = fp.create_group(group_name)
        for k, v in zip(keys, data):
            if v.dtype == np.dtype('O'):
                v = np.vstack(v)  # convert object array into a proper 2D array
            grp.create_dataset(str(k), data=v)

    def save(self, filename: str = 'waveform_dataset.h5'):
        """
        Save waveform data set to a HDF5 file.

        Parameters
        ----------
        filename : str
            The name of the output HDF5 file.
        """
        # TODO: use SVD to compress more than just HDF5 compression?
        # Using h5py
        fp = h5py.File(filename, 'w')
        self._write_datafrane_to_hdf5(fp, 'parameters', self._parameter_samples)
        self._write_datafrane_to_hdf5(fp, 'waveform_polarizations', self._waveform_polarizations)
        fp.close()

        # Using pandas and PyTables
        # Warning: PyTables will pickle object types that it cannot map directly to c-types
        # hdf = pd.HDFStore(filename, mode='a', complevel=complevel)
        # # Using the 'table' format we could hdf.append() data. If we don't do that, use 'fixed'.
        # hdf.put('parameters', self._parameter_samples, format='fixed', data_columns=None)
        # hdf.put('waveform polarizations', self._waveform_polarizations, format='fixed', data_columns=None)
        # hdf.close()

    def split_into_train_test(self, train_fraction):
        # of type WaveformDataset
        # TODO: implement
        pass



if __name__ == "__main__":
    """Explore chaining transforms together and applying these to a waveform dataset."""
    from dingo.gw.domains import UniformFrequencyDomain
    from dingo.gw.detector_network import DetectorNetwork, RandomProjectToDetectors
    from dingo.gw.noise import AddNoiseAndWhiten
    from transforms import StandardizeParameters, ToNetworkInput, Compose
    import matplotlib.pyplot as plt

    domain_kwargs = {'f_min': 20.0, 'f_max': 4096.0, 'delta_f': 1.0 / 4.0, 'window_factor': 1.0}
    domain = UniformFrequencyDomain(**domain_kwargs)
    priors = GWPriorDict(geocent_time_ref=1126259642.413, luminosity_distance_ref=500.0,
                         reference_frequency=20.0)
    approximant = 'IMRPhenomXPHM'
    waveform_generator = WaveformGenerator(approximant, domain)


    # Generate a dataset using the first waveform dataset object
    wd = WaveformDataset(priors=priors, waveform_generator=waveform_generator)
    n_waveforms = 17
    wd.generate_dataset(size=n_waveforms)

    # 1. RandomProjectToDetectors  (single waveform)
    #    in:  waveform_polarizations: Dict[str, np.ndarray], waveform_parameters: Dict[str, float]
    #    out: Dict[str, np.ndarray]      {'H1':h1_strain, ...}
    #    changes extrinsic pars

    det_network = DetectorNetwork(["H1", "L1"], domain, start_time=priors.reference_geocentric_time)
    rp_det = RandomProjectToDetectors(det_network, priors)
    data = wd[1]
    #{'parameters': parameters, 'waveform': waveform_polarizations}
    strain_dict = rp_det(data)  # needs to take a dict instead
    print(strain_dict)  # {'H1':..., 'L1':...}

    # Example of applying projection of detectors as a transformation
    wd = WaveformDataset(priors=priors, waveform_generator=waveform_generator, transform=rp_det)
    n_waveforms = 17
    wd.generate_dataset(size=n_waveforms)

    # 2. AddNoiseAndWhiten (also standardizes data)  (single waveform)
    #    in:  strain_dict: Dict[str, np.ndarray], waveform_parameters: Dict[str, float]
    #    out: Dict[str, np.ndarray]      {'H1':h1_strain_plus_noise_whitened, ...}
    #    provide inverse ASD (fixed - at least needs to be the same shape, or a draw from a PSD dataset)

    nw = AddNoiseAndWhiten(det_network)
    nw_strain_dict = nw(strain_dict)
    # plt.loglog(domain(), np.abs(strain_dict['waveform']['H1']), label='d')
    # plt.loglog(domain(), np.abs(nw_strain_dict['waveform']['H1']), label='d_w + n')
    # plt.show()


    # 3. StandardizeParameters (only standardizes parameters and leaves waveforms alone?) (single waveform)
    #    in: Dict[str, np.ndarray] : {'parameters': ..., 'waveform': ...}
    #    out:
    # TODO: Move to parameters
    # Example: Define fake means and stdevs
    # TODO: Replace this with the correct ones given a particular prior choice
    #  -- look at _compute_parameter_statistics()
    # This could be achieved by subclassing the bilby priors
    # For reference parameter values could just put the fixed value as a mean and stdev = 1.
    mu_dict = {'phi_jl': 1.0, 'tilt_1': 1.0, 'theta_jn': 2.0, 'tilt_2': 1.0, 'mass_1': 54.0, 'phi_12': 0.5,
               'chirp_mass': 40.0, 'phase': np.pi, 'a_2': 0.5, 'mass_2': 39.0, 'mass_ratio': 0.5,
               'a_1': 0.5, 'f_ref': 20.0, 'luminosity_distance': 1000.0, 'geocent_time': 1126259642.413,
               'ra': 2.5, 'dec': 1.0, 'psi': np.pi}
    std_dict = mu_dict


    print('Chaining transforms')
    # Note: This is difficult to test for extrinsic parameters, since they are generate in step 1
    # and standardized in step 2, so that the transform chain combined with its inverse is not the
    # identity
    transform = Compose([
        RandomProjectToDetectors(det_network, priors),
        AddNoiseAndWhiten(det_network),
        StandardizeParameters(mu=mu_dict, std=std_dict),
        ToNetworkInput(domain)
    ])

    wd = WaveformDataset(priors=priors, waveform_generator=waveform_generator, transform=transform)
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

