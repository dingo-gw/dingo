import numpy as np
from scipy.signal import tukey
from dingo.gw.prior_split import default_extrinsic_dict
from dingo.gw.prior_split import BBHExtrinsicPriorDict

def find_axis(array, dim):
    """Looks for axis with dimension dim in array, and returns its index."""
    indices = np.where(np.array(array.shape) == dim)[0]
    if len(indices) > 1:
        raise ValueError(f'Automatic axis detection ambiguous. Array of shape '
                         f'{array.shape} has {len(indices)} axes of dimension '
                         f'{dim}.')
    if len(indices) < 1:
        raise ValueError(f'Automatic axis detection failed. Array of shape '
                         f'{array.shape} has no axis of dimension {dim}.')
    else:
        return indices[0]

def truncate_array(array, axis, lower, upper):
    """Truncate array to [lower:upper] along selected axis."""
    sl = [slice(None)] * array.ndim
    sl[axis] = slice(lower, upper)
    return array[tuple(sl)]

def get_window_factor(window_kwargs):
    """Compute window factor from window_kwargs."""
    window_type = window_kwargs['window_type']
    if window_type == 'tukey':
        roll_off, T, f_s = window_kwargs['roll_off'], window_kwargs['T'], \
                           window_kwargs['f_s']
        alpha = 2 * roll_off / T
        w = tukey(int(T * f_s), alpha)
        return np.sum(w ** 2) / (T * f_s)
    else:
        raise NotImplementedError(f'Unknown window type {window_type}.')

def get_extrinsic_prior_dict(extrinsic_prior):
    """Build dict for extrinsic prior by starting with
    default_extrinsic_dict, and overwriting every element for which
    extrinsic_prior is not default."""
    extrinsic_prior_dict = default_extrinsic_dict.copy()
    for k, v in extrinsic_prior.items():
        if v.lower() != 'default':
            extrinsic_prior_dict[k] = v
    return extrinsic_prior_dict

def get_standardization_dict(extrinsic_prior_dict, wfd, selected_parameters):
    # get mean and std for extrinsic prior
    ext_prior = BBHExtrinsicPriorDict(extrinsic_prior_dict)
    mean_extrinsic, std_extrinsic = ext_prior.mean_std(ext_prior.keys())
    # get mean and std for intrinsic prior
    mean_intrinsic = {k: np.mean(wfd._parameter_samples[k]) for k in
                      wfd._parameter_samples.keys()}
    std_intrinsic = {k: np.std(wfd._parameter_samples[k]) for k in
                     wfd._parameter_samples.keys()}
    # check that overlap between intrinsic and extrinsic parameters is only
    # due to fiducial values (-> std 0)
    for k in std_intrinsic.keys() & std_extrinsic.keys():
        assert std_intrinsic[k] == 0
    # merge dicts, overwriting fiducial values for parameters (e.g.,
    # luminosity_distance) in intrinsic parameters by the extrinsic ones
    mean = {**mean_intrinsic, **mean_extrinsic}
    std = {**std_intrinsic, **std_extrinsic}
    # return standardization dict
    standardization_dict = {'mean': {k: mean[k] for k in selected_parameters},
                            'std': {k: std[k] for k in selected_parameters}}
    return standardization_dict