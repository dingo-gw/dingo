from dingo.gw.parameters import GWPriorDict, generate_parameter_prior_dictionary, generate_default_prior_dictionary
from typing import Dict, List, Union
import pandas as pd
import numpy as np


def build_prior(p_intrinsic: Dict[str, Union[List, str]],
                p_extrinsic: Dict[str, float], add_extrinsic_priors: bool = True):
    """
    Generate Dictionary of prior instances for intrinsic parameters.

    Parameters
    ----------
    p_intrinsic: Dict
        A dictionary containing prior options for intrinsic parameters
        Allowed values for each parameter are:
            * 'default' to use a default prior
            * a list [prior_class_name, minimum, maximum] for a custom prior

    p_extrinsic: Dict
        A dictionary containing reference values for extrinsic parameters

    add_extrinsic_priors: bool
        Add default priors for ra, dec, psi, d_L if they have not been specified.

    Note that default priors are added for certain missing parameters, but not
    for others. Check the warning messages that are being generated in these cases.
      * If mass priors are missing, default mass priors are added (including constraints)
      * If priors for ra, dec, psi are missing, default priors are added.
        If add_extrinsic_priors == True, then no warning messages will be shown.
      * If the luminosity distance prior is missing a default prior is added.
        If add_extrinsic_priors == True, then no warning messages will be shown.
      * If spin priors are incomplete generating waveforms will fail.

    Depending on the particular prior choices the dimensionality of a
    parameter sample obtained from the returned GWPriorDict will vary.
    """
    default_prior = generate_default_prior_dictionary()
    parameter_dict = {
        k: {'class_name': v[0], 'minimum': v[1], 'maximum': v[2]}
        for k, v in p_intrinsic.items() if isinstance(v, List)}
    parameter_prior_dict = generate_parameter_prior_dictionary(parameter_dict)
    parameter_prior_dict_default = {k: default_prior[k] for k, v in p_intrinsic.items() if v == 'default'}
    parameter_prior_dict.update(parameter_prior_dict_default)

    if add_extrinsic_priors:
        # Avoid warning messages for extrinsic parameters: d_L, ra, dec, psi
        parameter_prior_dict = GWPriorDict.add_ra_dec_psi_dL(parameter_prior_dict)

    geocent_time_ref = 0  # dummy value
    kwargs = {'luminosity_distance_ref': p_extrinsic['luminosity_distance'],
              'reference_frequency': p_extrinsic['reference_frequency'],
              'geocent_time_ref': geocent_time_ref}

    return GWPriorDict(parameter_prior_dict, **kwargs)


def structured_array_from_dict_of_arrays(d: Dict[str, np.ndarray], fmt: str = 'f8'):
    """
    Given a dictionary of 1-D numpy arrays, create a numpy structured array
    with field names equal to the dict keys and using the specified format.

    Parameters
    ----------
    d : Dict
        A dictionary of 1-dimensional numpy arrays
    fmt : str
        Format string for the array datatype.
    """
    arr_raw = np.array(list(d.values()))
    par_names = list(d.keys())
    dtype = np.dtype([x for x in zip(par_names, [fmt for _ in par_names])])
    arr_struct = np.array([tuple(x) for x in arr_raw.T], dtype=dtype)

    # Check: TODO: move to a unit test
    arr_rec = pd.DataFrame(d).to_records()
    assert np.all([np.allclose(arr_struct[k], arr_rec[k]) for k in par_names])

    return arr_struct


def get_params_dict_from_array(params_array, params_inds, f_ref=None):
    """
    Transforms an array with shape (num_parameters) to a dict. This is
    necessary for the waveform generator interface.

    Parameters
    ----------
    params_array:
        Array with parameters
    params_inds:
        Indices for the individual parameter keys.
    f_ref:
        Reference frequency of approximant
    params:
        Dictionary with the parameters
    """
    if len(params_array.shape) > 1:
        raise ValueError('This function only transforms a single set of '
                         'parameters to a dict at a time.')
    params = {}
    for k, v in params_inds.items():
        params[k] = params_array[v]
    if f_ref is not None:
        params['f_ref'] = f_ref
    return params
