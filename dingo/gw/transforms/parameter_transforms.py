import numpy as np
import torch
import pandas as pd
from dingo.gw.prior import BBHExtrinsicPriorDict
from .utils import get_batch_size_of_input_sample


class SampleExtrinsicParameters(object):
    """
    Sample extrinsic parameters and add them to sample in a separate dictionary.
    """

    def __init__(self, extrinsic_prior_dict):
        self.extrinsic_prior_dict = extrinsic_prior_dict
        self.prior = BBHExtrinsicPriorDict(extrinsic_prior_dict)

    def __call__(self, input_sample):
        sample = input_sample.copy()
        batched, batch_size = get_batch_size_of_input_sample(input_sample)
        extrinsic_parameters = self.prior.sample(batch_size if batched else None)
        extrinsic_parameters = {
            k: v.astype(np.float32) if batched else float(v)
            for k, v in extrinsic_parameters.items()
        }
        sample["extrinsic_parameters"] = extrinsic_parameters
        return sample

    @property
    def reproduction_dict(self):
        return {"extrinsic_prior_dict": self.extrinsic_prior_dict}


class SelectStandardizeRepackageParameters(object):
    """
    This transformation selects the parameters in standardization_dict,
    normalizes them by setting p = (p - mean) / std, and repackages the
    selected parameters to a numpy array.

    as_type: str = None
        only applies, if self.inverse == True
        * if None, data type is kept
        * if 'dict', dict with
        * if 'pandas', use pandas.DataFrame
    """

    def __init__(
        self,
        parameters_dict,
        standardization_dict,
        inverse=False,
        as_type=None,
        device="cpu",
    ):
        self.parameters_dict = parameters_dict
        self.mean = standardization_dict["mean"]
        self.std = standardization_dict["std"]
        self.N = len(self.mean.keys())
        if self.mean.keys() != self.std.keys():
            raise ValueError("Keys of means and stds do not match.")
        self.inverse = inverse
        self.as_type = as_type
        self.device = device

    def __call__(self, input_sample, as_type=None):
        """
        * if self.inverse == False:
            Normalize parameters (specified in self.parameters_dict),
            repackage to numpy array.
        * if self.inverse == True:
            Applies only to sample['inference_parameters'].
            Undo normalization and return as type self.as_type.
            Also transform input_sample['log_prob'], if present, according to the
            change-of-variables rule.

        Parameters
        ----------
        input_sample: dict
            input sample

        Returns
        -------
        sample: dict
            transformed sample
        """

        if not self.inverse:
            # Look for parameters in either the parameters dict, or the
            # extrinsic_parameters dict. extrinsic_parameters supersedes.
            if "extrinsic_parameters" in input_sample:
                full_parameters = {
                    **input_sample["parameters"],
                    **input_sample["extrinsic_parameters"],
                }
            else:
                full_parameters = input_sample["parameters"]

            sample = input_sample.copy()
            for k, v in self.parameters_dict.items():
                if len(v) > 0:
                    if isinstance(full_parameters[v[0]], torch.Tensor):
                        standardized = torch.empty(
                            (*full_parameters[v[0]].shape, len(v)),
                            dtype=torch.float32,
                            device=self.device,
                        )
                    elif isinstance(full_parameters[v[0]], np.ndarray):
                        standardized = np.empty(
                            (*full_parameters[v[0]].shape, len(v)), dtype=np.float32
                        )
                    else:
                        standardized = np.empty(len(v), dtype=np.float32)
                    for idx, par in enumerate(v):
                        if self.std[par] == 0:
                            raise ValueError(
                                f"Parameter {par} with standard deviation zero is included in inference parameters. "
                                f"This is not allowed. Please remove it from inference_parameters or create a new "
                                f"dataset where std({par}) is not zero."
                            )
                        standardized[..., idx] = (
                            full_parameters[par] - self.mean[par]
                        ) / self.std[par]
                    sample[k] = standardized

        else:
            sample = input_sample.copy()
            inference_parameters = self.parameters_dict["inference_parameters"]

            parameters = input_sample["parameters"][:]
            assert parameters.shape[-1] == len(inference_parameters), (
                f"Expected {len(inference_parameters)} parameters "
                f"({inference_parameters}), but got {parameters.shape[-1]}."
            )

            # de-normalize parameters
            for idx, par in enumerate(inference_parameters):
                parameters[..., idx] = (
                    parameters[..., idx] * self.std[par] + self.mean[par]
                )

            # TODO: Can we remove the as_type option? Do we ever want anything other
            #  than a dict?
            # return normalized parameters as desired type
            if self.as_type is None:
                sample["parameters"] = parameters

            elif self.as_type == "dict":
                sample["parameters"] = {}
                for idx, par in enumerate(inference_parameters):
                    sample["parameters"][par] = parameters[..., idx]

            elif self.as_type == "pandas":
                sample["parameters"] = pd.DataFrame(
                    np.array(parameters), columns=inference_parameters
                )

            else:
                raise NotImplementedError(
                    f"Unexpected type {self.as_type}, "
                    f"expected one of [None, pandas, dict]."
                )

            # TODO: Implement this for the forward map, if needed.
            if "log_prob" in sample:
                log_std = np.sum(np.log([self.std[p] for p in inference_parameters]))
                sample["log_prob"] -= log_std

        return sample


class StandardizeParameters:
    """
    Standardize parameters according to the transform (x - mu) / std.
    """

    def __init__(self, mu, std):
        """
        Initialize the standardization transform with means
        and standard deviations for each parameter

        Parameters
        ----------
        mu : Dict[str, float]
            The (estimated) means
        std : Dict[str, float]
            The (estimated) standard deviations
        """
        self.mu = mu
        self.std = std
        if not set(mu.keys()) == set(std.keys()):
            raise ValueError(
                "The keys in mu and std disagree:" f"mu: {mu.keys()}, std: {std.keys()}"
            )

    def __call__(self, samples):
        """Standardize the parameter array according to the
        specified means and standard deviations.

        Parameters
        ----------
        samples: Dict[Dict, Dict]
            A nested dictionary with keys 'parameters', 'waveform',
            'noise_summary'.

        Only parameters included in mu, std get transformed.
        """
        x = samples["parameters"]
        y = {k: (x[k] - self.mu[k]) / self.std[k] for k in self.mu.keys()}
        samples_out = samples.copy()
        samples_out["parameters"] = y
        return samples_out

    def inverse(self, samples):
        """De-standardize the parameter array according to the
        specified means and standard deviations.

        Parameters
        ----------
        samples: Dict[Dict, Dict]
            A nested dictionary with keys 'parameters', 'waveform',
            'noise_summary'.

        Only parameters included in mu, std get transformed.
        """
        y = samples["parameters"]
        x = {k: self.mu[k] + y[k] * self.std[k] for k in self.mu.keys()}
        samples_out = samples.copy()
        samples_out["parameters"] = x
        return samples_out
