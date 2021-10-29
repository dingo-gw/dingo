import numpy as np
from scipy.signal import tukey

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