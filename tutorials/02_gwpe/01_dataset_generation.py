'''
Exepected output of this is a waveform_dataset.hdf5 file with a dataset of
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
'''

from os.path import join
from dingo.gw.waveform_generator import WaveformGenerator
import yaml

# directory for the waveform dataset, contains settings.yaml file with settings
# for the prior, approximant and data conditioning
waveforms_directory = './datasets/waveforms/'
# number of waveforms in the dataset
N = 100_000
# number of waveforms used to generate the reduced basis for compression
n_rb = 30_000

# load settings
with open(join(waveforms_directory, 'settings.yaml'), 'r') as stream:
    settings = yaml.safe_load(stream)

# build the prior
prior_settings = settings['prior_settings']
'''This function should build the prior from the human-readable settings.'''
prior = build_prior(**prior_settings)

# sample parameters and save them to the hdf5 file
'''This should work already.'''
parameters = prior.sample_intrinsic(size=N, add_reference_values=True)
'''We save the parameters as an array of shape (N, 15), where 15 is the 
number of parameters. The api function below takes the parameter dicts and 
transforms them to the array.'''
parameters = dingo.api.parameter_dict_to_array(
    parameters, prior_settings['parameter_indices']
)

# define domain and waveform generator
domain = build_domain(settings['domain_settings'])
waveform_generator = WaveformGenerator(
    settings['waveform_generator_settings']['approximant'], domain)

# generate polarizations for parameters in the dataset
# generate n_rb full polarizations first to generate the SVD matrix V,
# then generate all N polarizations and save them to the hdf5 file in the rb
# compression.









'''Put these functions into dingo.api'''

def build_prior(**kwargs):
    pass

def build_domain(domain_settings):
    if domain_settings['name'].lower() == 'uniformfrequencydomain':
        return UniformFrequencyDomain(**domain_settings['kwargs'])
    else:
        raise ValueError(f'Domain {domain_settings["name"]} not impoemented.')

def get_params_dict_from_array(params_array, params_inds, f_ref=None):
    '''
    Transforms an array with shape (num_parameters) to a dict. This is
    necessary for the waveform generator interface.

    :param params_array: Array with parameters
    :param params_inds: Indices for the individual parameter keys.
    :param f_ref: Reference frequency of approximant
    :return: params: Dictionary with the parameters
    '''
    if len(params_array.shape) > 1:
        raise ValueError('This function only transforms a single set of '
                         'parameters to a dict at a time.')
    params = {}
    for k, v in params_inds.items():
        params[k] = params_array[v]
    if f_ref is not None:
        params['f_ref'] = f_ref
    return params