import numpy as np

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