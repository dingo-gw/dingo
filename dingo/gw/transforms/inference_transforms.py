class PostCorrectGeocentTime(object):
    """
    Post correction for geocent time: add GNPE proxy (only necessary if exact
    equivariance is enforced)
    """

    def __init__(self):
        pass

    def __call__(self, input_sample):
        sample = input_sample.copy()
        parameters = sample["parameters"].copy()
        extrinsic_parameters = sample["extrinsic_parameters"].copy()
        parameters["geocent_time"] -= extrinsic_parameters.pop("geocent_time")
        sample["parameters"] = parameters
        sample["extrinsic_parameters"] = extrinsic_parameters
        return sample

class CopyToExtrinsicParameters(object):
    """
    Post correction for geocent time: add GNPE proxy (only necessary if exact
    equivariance is enforced)
    """

    def __init__(self, *parameter_list):
        self.parameter_list = parameter_list

    def __call__(self, input_sample):
        sample = input_sample.copy()
        extrinsic_parameters = sample["extrinsic_parameters"].copy()
        for par in self.parameter_list:
            extrinsic_parameters[par] = sample["parameters"][par]
        sample["extrinsic_parameters"] = extrinsic_parameters
        return sample

