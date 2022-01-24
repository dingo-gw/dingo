def structured_array_from_dict_of_arrays(d: Dict[str, np.ndarray], fmt: str = "f8"):
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


def dataframe_to_structured_array(df: pd.DataFrame):
    """
    Convert a pandas DataFrame of parameters to a structured numpy array.

    Parameters
    ----------
    df:
        A pandas DataFrame
    """
    d = {k: np.array(list(v.values())) for k, v in df.to_dict().items()}
    return structured_array_from_dict_of_arrays(d)


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
        raise ValueError(
            "This function only transforms a single set of "
            "parameters to a dict at a time."
        )
    params = {}
    for k, v in params_inds.items():
        params[k] = params_array[v]
    if f_ref is not None:
        params["f_ref"] = f_ref
    return params