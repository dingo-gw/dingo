def get_batch_size_of_input_sample(input_sample):
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