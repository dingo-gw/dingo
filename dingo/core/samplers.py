import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp
from torchvision.transforms import Compose

from dingo.core.models import PosteriorModel
from dingo.core.samples_dataset import SamplesDataset

#
# Sampler classes are based on Bilby Samplers.
#


class Sampler(object):
    """
    Sampler class that wraps a PosteriorModel. Allows for conditional and unconditional
    models.

    Draws samples from the model based on (optional) context data, and outputs in various
    formats.
    """

    def __init__(self, model: PosteriorModel):
        """
        Parameters
        ----------
        model : PosteriorModel
        """
        self.model = model
        self.metadata = self.model.metadata.copy()

        # For unconditional models, the context will be stored with the model,
        # so we copy it here. This is necessary for calculating the likelihood for
        # importance sampling.
        self.context = self.model.context

        if "base" in self.metadata:
            self.base_model_metadata = self.metadata["base"]
        else:
            self.base_model_metadata = self.metadata

        self.transforms_pre = Compose([])
        self.transforms_post = Compose([])
        self._search_parameter_keys = []
        self._constraint_parameter_keys = []
        self._fixed_parameter_keys = []
        self.inference_parameters = []
        self._build_prior()
        self._build_domain()
        self._reset_result()

        self._pesummary_package = "core"

    def _reset_result(self):
        """Clear out all data produced by self.run_sampler(), to prepare for the next
        sampler run."""
        self.samples = None
        self.log_evidence = None
        self.effective_sample_size = None

    def _run_sampler(self, num_samples: int, context: Optional[dict] = None) -> dict:

        if context is not None:
            x = context.copy()
            x["parameters"] = {}
            x["extrinsic_parameters"] = {}

            # transforms_pre are expected to transform the data in the same way for each
            # requested sample. We therefore expand it across the batch *after*
            # pre-processing.
            x = self.transforms_pre(context)
            x = x.expand(num_samples, *x.shape)

            # For a normalizing flow, we get the log_prob for "free" when sampling,
            # so we always include this. For other architectures, it may make sense to
            # have a flag for whether to calculate the log_prob.
            self.model.model.eval()
            with torch.no_grad():
                y, log_prob = self.model.model.sample_and_log_prob(x)

        else:
            self.model.model.eval()
            with torch.no_grad():
                y, log_prob = self.model.model.sample_and_log_prob(
                    num_samples=num_samples
                )

        samples = self.transforms_post({"parameters": y, "log_prob": log_prob})
        result = samples["parameters"]
        result["log_prob"] = samples["log_prob"]

        return result

    def run_sampler(
        self,
        num_samples: int,
        context: Optional[dict] = None,
        batch_size: Optional[int] = None,
        event_metadata: Optional[dict] = None,
    ):
        """
        Generates samples and stores them along with metadata in self.

        Allows for batched sampling, e.g., if limited by GPU memory.

        Actual sampling is performed by self._run_sampler().

        Parameters
        ----------
        num_samples : int
            Number of samples requested.
        context : dict, optional
            Data on which to condition the sampler.
            For injections, there should be a 'parameters' key with truth values.
        batch_size : int, optional
            Batch size for sampler.
        event_metadata : dict, optional
            Metadata for data analyzed. Stored along with sample metadata, and can in
            principle influence any post-sampling parameter transformations (e.g.,
            sky position correction).
        """
        # Reset sampling results and store all metadata associated with data.
        self._reset_result()
        if context is not None:
            self.metadata["injection_parameters"] = context.pop("parameters", None)
            self.context = context
        self.metadata["event"] = event_metadata
        self._check_context(context)

        # Carry out batched sampling by calling _run_sample() on each batch and
        # consolidating the results.
        if batch_size is None:
            batch_size = num_samples
        full_batches, remainder = divmod(num_samples, batch_size)
        samples = [self._run_sampler(batch_size, context) for _ in range(full_batches)]
        if remainder > 0:
            samples.append(self._run_sampler(remainder, context))
        samples = {p: torch.cat([s[p] for s in samples]) for p in samples[0].keys()}

        # Apply any post-sampling corrections to sampled parameters, and place on CPU.
        self._post_correct(samples)
        samples = {k: v.cpu().numpy() for k, v in samples.items()}

        self.samples = pd.DataFrame(samples)

    def _check_context(self, context: Optional[dict] = None):
        # TODO: Add some checks that the context is appropriate.
        pass

    def _post_correct(self, samples: dict):
        pass

    def _build_prior(self):
        self.prior = None

    def _build_domain(self):
        self.domain = None

    def _build_likelihood(self, **likelihood_kwargs):
        self.likelihood = None

    def importance_sample(self, num_processes: int = 1, **likelihood_kwargs):

        if self.samples is None:
            raise KeyError(
                "Initial samples are required for importance sampling. "
                "Please execute run_sampler()."
            )
        if "log_prob" not in self.samples:
            raise KeyError(
                "Stored samples do not contain log probability, which is "
                "needed for importance sampling."
            )

        self._build_likelihood(**likelihood_kwargs)

        # Proposal samples and associated log probability have already been calculated
        # using the stored model. These form a normalized probability distribution.

        log_prob_proposal = self.samples["log_prob"].to_numpy()
        theta = self.samples.drop(columns="log_prob")

        # TODO: Expand theta_all to include all parameters that make up the prior,
        #  including constraints. These might not be included within the inference
        #  parameters theta.
        theta_all = theta

        # Calculate the (un-normalized) target density as prior times likelihood,
        # evaluated at the same sample points. The expensive part is the likelihood,
        # so we allow for multiprocessing.

        log_prior = self.prior.ln_prob(theta_all, axis=0)

        # The prior may evaluate to -inf for certain samples. For these, we do not want
        # to evaluate the likelihood, in particular because it may not even be possible
        # to generate data outside the prior (e.g., for BH spins > 1). We represent the
        # log likelihood of such samples as NaN.

        within_prior = log_prior != -np.inf
        print(f"Calculating {np.sum(within_prior)} likelihoods.")
        t0 = time.time()
        log_likelihood = np.empty_like(log_prior)
        log_likelihood[:] = np.nan
        log_likelihood[within_prior] = self.likelihood.log_likelihood_multi(
            theta.loc[within_prior], num_processes=num_processes
        )
        print(f"Done. This took {time.time() - t0:.2f} seconds.")

        # Calculate weights, careful to handle samples where the likelihood was not
        # evaluated.
        log_weights = log_prior
        log_weights[within_prior] += (
            log_likelihood[within_prior] - log_prob_proposal[within_prior]
        )
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= np.mean(weights)

        self.samples["weights"] = weights
        self.samples["log_likelihood"] = log_likelihood
        self.samples["log_prior"] = log_prior
        # Note that self.samples['log_prob'] is the *proposal* log_prob, not the
        # importance_weighted log_prob.

        self.log_evidence = logsumexp(log_weights) - np.log(len(self.samples))
        self.effective_sample_size = np.sum(weights) ** 2 / np.sum(weights ** 2)
        breakpoint()

    def write_pesummary(self, filename):
        from pesummary.io import write
        from pesummary.utils.samples_dict import SamplesDict

        samples_dict = SamplesDict(self.samples)
        write(
            samples_dict,
            package=self._pesummary_package,
            file_format="hdf5",
            filename=filename,
        )
        # TODO: Save much more information.

    def to_samples_dataset(self) -> SamplesDataset:
        data_dict = {
            "settings": self.metadata,
            "samples": self.samples,
            "context": self.context,
            "log_evidence": self.log_evidence,
            "effective_sample_size": self.effective_sample_size,
        }
        return SamplesDataset(dictionary=data_dict)

    def to_hdf5(self, label="", outdir="."):
        dataset = self.to_samples_dataset()
        file_name = "dingo_samples_" + label + ".hdf5"
        dataset.to_file(file_name=Path(outdir, file_name))

    def print_summary(self):
        print("Number of samples:", len(self.samples))
        if self.log_evidence is not None:
            print("Log(evidence):", self.log_evidence)
            print(
                f"Effective sample size: {self.effective_sample_size:.1f} "
                f"({100 * self.effective_sample_size / len(self.samples):.2f}%)"
            )


class GNPESampler(Sampler):
    """
    Base class for GNPE sampler. It wraps a PosteriorModel, and must contain also an NPE
    sampler, which is used to generate initial samples.
    """

    def __init__(
        self,
        model: PosteriorModel,
        init_sampler: Sampler,
        num_iterations: int,
    ):
        """
        Parameters
        ----------
        model : PosteriorModel
        init_sampler : Sampler
            Used for generating initial samples
        num_iterations : int
            Number of GNPE iterations to be performed by sampler.
        """
        super().__init__(model)
        self.init_sampler = init_sampler
        self.num_iterations = num_iterations
        self.gnpe_parameters = None

    @property
    def init_sampler(self):
        return self._init_sampler

    @init_sampler.setter
    def init_sampler(self, value):
        self._init_sampler = value
        self.metadata["init_model"] = self._init_sampler.model.metadata

    @property
    def num_iterations(self):
        return self._num_iterations

    @num_iterations.setter
    def num_iterations(self, value):
        self._num_iterations = value
        self.metadata["num_iterations"] = self._num_iterations

    def _run_sampler(self, num_samples: int, context: Optional[dict] = None) -> dict:

        assert context is not None
        data_ = self.init_sampler.transforms_pre(context)

        x = {
            "extrinsic_parameters": self.init_sampler._run_sampler(
                num_samples, context
            ),
            "parameters": {},
        }
        for i in range(self.num_iterations):
            print(i)
            x["extrinsic_parameters"] = {
                k: x["extrinsic_parameters"][k] for k in self.gnpe_parameters
            }
            d = data_.clone()
            x["data"] = d.expand(num_samples, *d.shape)

            x = self.transforms_pre(x)
            x["parameters"] = self.model.sample(x["data"], x["context_parameters"])
            x = self.transforms_post(x)

        samples = x["parameters"]

        return samples
