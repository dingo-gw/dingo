import numpy as np
import torch
import pandas as pd
from bilby.core.prior import PriorDict
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


class DistancePriorConditioning(object):
    """
    Prior conditioning on the upper bound of the luminosity-distance prior. For each
    sample, draw luminosity_distance_max from a hyperprior, then redraw
    luminosity_distance from the base (extrinsic) prior truncated to
    [minimum, luminosity_distance_max], by inverse transform sampling. The network
    conditions on log_luminosity_distance_max, so that an event can be analyzed with
    the base prior truncated at any bound in the range of the hyperprior, while the
    hyperprior controls how much training goes to small distances.

    With probability log_uniform_fraction, luminosity_distance is instead drawn
    log-uniformly over the same range. This places more training samples at distances
    small compared to the bound; the network then learns the posterior under the
    corresponding mixture prior, which importance sampling against the base prior
    corrects.
    """

    context_parameters = ["log_luminosity_distance_max"]

    def __init__(self, base_prior, hyperprior, log_uniform_fraction: float = 0.0):
        """
        Parameters
        ----------
        base_prior : str or bilby.core.prior.Prior
            The luminosity_distance prior, e.g. UniformComovingVolume.
        hyperprior : str or bilby.core.prior.Prior
            Prior for luminosity_distance_max, e.g. LogUniform. Its maximum must equal
            that of the base prior, and its minimum must exceed that of the base prior.
        log_uniform_fraction : float
            Fraction of samples with log-uniformly drawn luminosity_distance.
        """
        priors = PriorDict(
            {"luminosity_distance": base_prior, "luminosity_distance_max": hyperprior}
        )
        self.base_prior = priors["luminosity_distance"]
        self.hyperprior = priors["luminosity_distance_max"]
        self.log_uniform_fraction = log_uniform_fraction
        if not np.isclose(self.hyperprior.maximum, self.base_prior.maximum):
            raise ValueError(
                f"The luminosity_distance_max hyperprior maximum "
                f"({self.hyperprior.maximum}) must equal the luminosity_distance prior "
                f"maximum ({self.base_prior.maximum})."
            )
        if not self.hyperprior.minimum > self.base_prior.minimum:
            raise ValueError(
                f"The luminosity_distance_max hyperprior minimum "
                f"({self.hyperprior.minimum}) must exceed the luminosity_distance prior "
                f"minimum ({self.base_prior.minimum})."
            )
        if not 0 <= log_uniform_fraction <= 1:
            raise ValueError(
                f"log_uniform_fraction must be in [0, 1], got {log_uniform_fraction}."
            )

    def __call__(self, input_sample):
        sample = input_sample.copy()
        extrinsic_parameters = sample["extrinsic_parameters"].copy()
        batched = isinstance(extrinsic_parameters["luminosity_distance"], np.ndarray)
        size = len(extrinsic_parameters["luminosity_distance"]) if batched else None
        d_max = self.hyperprior.sample(size)
        d_min = self.base_prior.minimum
        u = np.random.uniform(size=size)
        d = np.where(
            np.random.uniform(size=size) < self.log_uniform_fraction,
            d_min * (d_max / d_min) ** u,
            self.base_prior.rescale(u * self.base_prior.cdf(d_max)),
        )
        for k, v in [
            ("luminosity_distance", d),
            ("log_luminosity_distance_max", np.log(d_max)),
        ]:
            extrinsic_parameters[k] = v.astype(np.float32) if batched else float(v)
        sample["extrinsic_parameters"] = extrinsic_parameters
        return sample


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
