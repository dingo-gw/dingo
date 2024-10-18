from typing import Callable, Iterable, Dict

import numpy as np
import pandas as pd 


def get_version():
    try:
        from dingo import __version__

        return __version__
    except ImportError:
        return None


def recursive_check_dicts_are_equal(dict_a, dict_b):
    if dict_a.keys() != dict_b.keys():
        return False
    else:
        for k, v_a in dict_a.items():
            v_b = dict_b[k]
            if type(v_a) != type(v_b):
                return False
            if type(v_a) == dict:
                if not recursive_check_dicts_are_equal(v_a, v_b):
                    return False
            elif not np.all(v_a == v_b):
                return False
    return True

def call_func_strict_output_dim(
    func: Callable[[int], Iterable[Iterable]],
    num_request: int,
    buffer_fraction: float = 0.02, 
):
    """
    Repeatedly calls a function until the output shape is the 
    size of num_samples. This can be useful when a user requests 
    N samples from a function, but because of failures in that 
    function, only M < N sample are returned. 

    Parameters
    ----------
    func : Callable
        Function to repeatedly call. This should take an 
        in return an iterable of iterables. So for example 
        [np.ndarray, pd.DataFrame, dict{"key":np.array}].
        Each of these iterables should have the same length.
    num_request : int
        The output size of the function
    buffer_fraction : float, optional
        The fraction of extra samples to generate given 
        we know the fraction of failed waveforms.

    """
    num_total, true_num_request = 0, num_request
    final_output = None 
    while num_total < true_num_request:
        output = func(num_request)

        # for the first iteration we don't need to append
        # to previous outputs 
        if final_output is None:
            final_output = list(output)
            # estimating the failed fraction on the first iteration
            # is most accurate
            failed_fraction = 1 - (len(output[0]) / num_request)
            if 1 - failed_fraction < 1e-6:
                raise ValueError(
                    f"""{failed_fraction * 100:1f}% of the function failed to generate.
                    Please check the inputs to {func} are sensible an try again. """
                )
        else:
            # appending the output to the final output
            for i, iterable in enumerate(output):
                if isinstance(iterable, np.ndarray):
                    final_output[i] = np.vstack([final_output[i], iterable])

                if isinstance(iterable, pd.DataFrame):
                    final_output[i] = pd.concat([final_output[i], iterable])

                if isinstance(iterable, dict):
                    for key, value in iterable.items():
                        final_output[i][key] = np.vstack([final_output[i][key], value])

        # estimate the fraction of failures
        num_total += len(output[0])
        num_func_to_generate = num_request - num_total
        # generating extra calls given that we know the failing fraction 
        num_request = int((num_func_to_generate / failed_fraction) * (1 + buffer_fraction))

    # truncating the final output to the correct size
    for i, iterable in enumerate(final_output):
        if isinstance(iterable, np.ndarray):
            final_output[i] = iterable[:true_num_request]

        if isinstance(iterable, pd.DataFrame):
            final_output[i] = iterable.iloc[:true_num_request]

        if isinstance(iterable, dict):
            for key, value in iterable.items():
                final_output[i][key] = value[:true_num_request]

    return final_output
