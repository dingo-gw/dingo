from typing import Tuple


def get_batch_size_of_input_sample(input_sample: dict) -> Tuple[bool, int]:
    """
    Parameters
    ----------
    input_sample : dict
        The input sample to transform. Can contain keys of
        parameters, polarizations

    Returns
    -------
    Tuple[bool, int]
        A tuple containing a boolean indicating if the input sample is batched
        and an integer representing the batch size of the input sample

    """
    if "parameters" in input_sample:
        params = next(iter(input_sample["parameters"].values()))

        # if parameters is a float not a batched input
        if isinstance(params, float):
            return False, 1
        else:
            # if single value array, return batch size 1
            if params.ndim < 1:
                return True, 1
            else:
                return True, params.shape[0]
    else:
        raise NotImplementedError(
            """The parameter key is used to determine the batch size of the
            input sample. If you want to determine the batch size a different
            way, you can implement it here.""" )
