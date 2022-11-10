import numpy as np
import torch


class PostCorrectGeocentTime(object):
    """
    Post correction for geocent time: add GNPE proxy (only necessary if exact
    equivariance is enforced)
    """

    def __init__(self, inverse=False):
        self.inverse = inverse

    def __call__(self, input_sample):
        sign = (1, -1)[self.inverse]
        sample = input_sample.copy()
        parameters = sample["parameters"].copy()
        extrinsic_parameters = sample["extrinsic_parameters"].copy()
        parameters["geocent_time"] -= extrinsic_parameters.pop("geocent_time") * sign
        sample["parameters"] = parameters
        sample["extrinsic_parameters"] = extrinsic_parameters
        return sample


class CopyToExtrinsicParameters(object):
    """
    Copy parameters specified in self.parameter_list from sample["parameters"] to
    sample["extrinsic_parameters"].
    """

    def __init__(self, *parameter_list):
        self.parameter_list = parameter_list

    def __call__(self, input_sample):
        sample = input_sample.copy()
        extrinsic_parameters = sample["extrinsic_parameters"].copy()
        for par in self.parameter_list:
            if par in sample['parameters']:
                extrinsic_parameters[par] = sample["parameters"][par]
        sample["extrinsic_parameters"] = extrinsic_parameters
        return sample


class ExpandStrain(object):
    """
    Expand the waveform of sample by adding a batch axis and copying the waveform
    num_samples times along this new axis. This is useful for generating num_samples
    samples at inference time.
    """

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __call__(self, input_sample):
        sample = input_sample.copy()
        waveform = input_sample["waveform"]
        sample["waveform"] = waveform.expand(self.num_samples, *waveform.shape)
        return sample


class ToTorch(object):
    """
    Convert all numpy arrays sample to torch tensors and push them to the specified
    device. All items of sample that are not numpy arrays (e.g., dicts of arrays)
    remain unchanged.
    """

    def __init__(self, device="cpu"):
        self.device = device

    def __call__(self, input_sample):
        sample = input_sample.copy()
        for k, v in sample.items():
            if type(v) == np.ndarray:
                sample[k] = torch.from_numpy(v).to(self.device, non_blocking=True)
        return sample


class ResetSample(object):
    """
    Resets sample:
        * waveform was potentially modified by gnpe transforms, so reset to waveform_
        * optionally remove all non-required extrinsic parameters
    """

    def __init__(self, extrinsic_parameters_keys=None):
        self.extrinsic_parameters_keys = extrinsic_parameters_keys

    def __call__(self, input_sample):
        sample = input_sample.copy()
        # reset the waveform
        sample["waveform"] = sample["waveform_"].clone()
        # optionally remove all non-required extrinsic parameters
        if self.extrinsic_parameters_keys is not None:
            sample["extrinsic_parameters"] = {
                k: sample["extrinsic_parameters"][k]
                for k in self.extrinsic_parameters_keys
            }
        return sample
