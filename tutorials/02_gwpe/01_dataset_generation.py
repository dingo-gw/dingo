"""
Expected output of this is a waveform_dataset.hdf5 file with a dataset of
N waveforms. It contains the following files.

'parameters':
    (N,15)-dimensional array with parameter sampled from the prior.
    Extrinsic parameters are set to reference values.
'rb_matrix_V':
    SVD matrix V that is used to compress/decompress the raw waveform
    polarizations.
'hc':
    (N,n_rb)-dimensional array with h_cross polarizations for the
    corresponding parameters. Saved in a compressed form using the SVD matrix V.
'hp':
    (N,n_rb)-dimensional array with h_plus polarizations for the
    corresponding parameters. Saved in a compressed form using the SVD matrix V.
"""

from os.path import join
from dingo.gw.waveform_generator import WaveformGenerator
from dingo.gw.waveform_dataset import WaveformDataset
from dingo.api import build_prior, build_domain, structured_array_from_dict_of_arrays
import yaml


def generate_waveform_dataset(waveforms_directory: str, settings_file: str,
                              dataset_file: str, n_wfs: int, n_rb: int):
    """
    Parse settings file, set up priors, domain, waveform generator
    and generated a data set of waveforms which is saved in
    compressed form in a HDF5 file.

    Parameters
    ----------
    waveforms_directory: str
        Directory containing settings file.
        The generated dataset will be saved there as well.

    settings_file:
        yaml file which contains options for the parameter prior,
        and waveform domain and model settings.

    dataset_file:
        Filename for the HDF5 file to which the generated
        waveform polarizations and parameters are writted.

    n_wfs:
        Number of waveforms to generate

    n_rb:
        Number basis functions to include in SVD basis
    """

    # Load settings
    with open(join(waveforms_directory, settings_file), 'r') as stream:
        settings = yaml.safe_load(stream)

    # Build prior distribution
    prior_settings = settings['prior_settings']
    prior = build_prior(prior_settings['intrinsic_parameters'],
                        prior_settings['extrinsic_parameters_reference_values'],
                        add_extrinsic_priors=True)

    # Define physical domain and set up waveform generator
    # MP: settings has f_ref, but we added 'reference_frequency' at the prior generation step
    #  A domain doesn't know about f_ref and is not tied to a single f_ref value
    domain = build_domain(settings['domain_settings'])
    waveform_generator = WaveformGenerator(
        settings['waveform_generator_settings']['approximant'], domain)

    wd = WaveformDataset(priors=prior, waveform_generator=waveform_generator, transform=None)

    # TODO: implement and use wd.read_parameter_samples(filename) for regression test against dev code
    #  to substitute loaded parameters from file
    # Explicitly sample parameters
    wd.sample_intrinsic(size=n_wfs, add_reference_values=True)

    # Generate polarizations for parameters in the dataset
    wd.generate_dataset(size=n_wfs)

    # TODO:
    #   * generate n_rb full polarizations first to generate the SVD matrix V,
    #   * then generate all N polarizations and save them to the hdf5 file in the rb
    #     compression.

    # MP: At the moment, wd.generate_dataset generates all waveforms and
    # then the truncated SVD basis is computed from the full dataset.
    wd.save(join(waveforms_directory, dataset_file), compress_data=True, n_rb=n_rb)


if __name__ == "__main__":
    # directory for the waveform dataset, contains settings.yaml file with settings
    # for the prior, approximant and data conditioning
    waveforms_directory = './datasets/waveforms/'
    # number of waveforms in the dataset
    # n_wfs = 100_000
    n_wfs = 200  # debugging setting
    # number of waveforms used to generate the reduced basis for compression
    # n_rb = 30_000
    n_rb = 10  # debugging setting

    settings_file = 'settings.yaml'
    dataset_file = 'dataset_test.hdf5'

    generate_waveform_dataset(waveforms_directory, settings_file, dataset_file, n_wfs, n_rb)

    # Test that the parameters and field names are written correctly to HDF5
    import h5py
    fp = h5py.File(join(waveforms_directory, dataset_file), 'r')
    pars = fp['parameters'][:]
    print(pars)
    print(pars.dtype)
    print(pars.shape)  # looks like 1D
    print(len(pars.dtype))  # but have this many fields = columns
    fp.close()

