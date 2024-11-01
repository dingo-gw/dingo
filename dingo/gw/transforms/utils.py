def get_batch_size_of_input_sample(input_sample):
    if "parameters" in input_sample:
        params = next(iter(input_sample["parameters"].values()))
        if isinstance(params, float):
            return False, 1
        else:
            return params.ndim > 1, params.shape[0]