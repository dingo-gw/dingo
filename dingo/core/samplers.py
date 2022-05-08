import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
from bilby.core.prior import Prior, Constraint, DeltaFunction
from bilby.core.result import Result
from scipy.special import logsumexp
from torchvision.transforms import Compose

from dingo.core.likelihood import Likelihood
from dingo.core.models import PosteriorModel
from dingo.gw.gwutils import get_extrinsic_prior_dict
from dingo.gw.prior import build_prior_with_defaults

#
# Sampler classes are based on Bilby Samplers.
#

class ConditionalSampler(object):
    """
    Conditional sampler class that wraps a PosteriorModel.

    Draws samples from the model based on context data, and outputs in various formats.
    """

    def __init__(self, model: PosteriorModel, likelihood: Optional[Likelihood] = None):
        """
        Parameters
        ----------
        model : PosteriorModel
        """
        self.model = model
        self.transforms_pre = Compose([])
        self.transforms_post = Compose([])
        self._search_parameter_keys = []
        self._constraint_parameter_keys = []
        self._fixed_parameter_keys = []
        self._build_prior()
        self.likelihood = likelihood
        self._reset_sampler()
        self._pesummary_package = 'core'

    def _run_sampler(self, num_samples: int, context: dict) -> dict:
        x = context.copy()
        x["parameters"] = {}
        x["extrinsic_parameters"] = {}

        # transforms_pre are expected to transform the data in the same way for each
        # requested sample. We therefore expand it across the batch *after*
        # pre-processing.
        x = self.transforms_pre(context)
        x = x.expand(num_samples, *x.shape)
        y = self.model.sample(x)
        samples = self.transforms_post({"parameters": y})["parameters"]

        return samples

    def run_sampler(
        self,
        num_samples: int,
        context: dict,
        batch_size: Optional[int] = None,
        label: Optional[str] = None,
        event_metadata: Optional[dict] = None,
        # as_type: str = "result",
    ):
        """
        Generates samples and returns them in requested format. Samples (and metadata)
        are also saved within the ConditionalSampler instance.

        Allows for batched sampling, e.g., if limited by GPU memory.

        Actual sampling is performed by self._run_sampler().

        Parameters
        ----------
        num_samples : int
            Number of samples requested.
        context : dict
            Data on which to condition the sampler.
            For injections, there should be a 'parameters' key with truth values.
        batch_size : int
            (Optional) Batch size for sampler.
        label : str
            (Optional) Label which is forwarded to the Results instance.
        event_metadata : dict
            (Optional) Metadata for data analyzed. Stored along with sample metadata,
            and can in principle influence any post-sampling parameter transformations
            (e.g., sky position correction).
        as_type : str
            Format of output ('results', 'pandas', or 'dict').

        Returns
        -------
        Samples in format specified by as_type.
        """
        # Reset sampler and store all metadata associated with data.
        self._reset_sampler()
        self.injection_parameters = context.pop("parameters", None)
        self.label = label
        self._store_metadata(event_metadata=event_metadata)

        # Carry out batched sampling by calling _run_sample() on each batch and
        # consolidating the results.
        if batch_size is None:
            batch_size = num_samples
        full_batches, remainder = divmod(num_samples, batch_size)
        batch_sizes = [batch_size] * full_batches
        if remainder > 0:
            batch_sizes += [remainder]
        sample_list = []
        for i, n in enumerate(batch_sizes):
            # TODO: Make sure we don't need to copy the context.
            print(f"Sampling batch {i+1} of {len(batch_sizes)}, size {n}.")
            sample_list.append(self._run_sampler(n, context))
        samples = {
            p: torch.cat([s[p] for s in sample_list]) for p in sample_list[0].keys()
        }

        # Apply any post-sampling corrections to sampled parameters, and place on CPU.
        self._post_correct(samples)
        samples = {k: v.cpu().numpy() for k, v in samples.items()}

        self.samples = pd.DataFrame(samples)

        # Prepare output
        # self._generate_result()

        # if as_type == "result":
        #     return self.result
        # elif as_type == "pandas":
        #     samples = pd.DataFrame(self.samples)
        #     samples.attrs = self.metadata
        #     return samples
        # elif as_type == "dict":
        #     return self.samples

    def _post_correct(self, samples: dict):
        pass

    def _build_prior(self):
        """Build the prior based on model metadata."""
        intrinsic_prior = self.model.metadata["dataset_settings"]["intrinsic_prior"]
        extrinsic_prior = get_extrinsic_prior_dict(
            self.model.metadata["train_settings"]["data"]["extrinsic_prior"]
        )
        self.prior = build_prior_with_defaults({**intrinsic_prior, **extrinsic_prior})

        # Initialize lists of parameters (from Bilby)
        for key in self.prior:
            if isinstance(self.prior[key], Prior) and self.prior[key].is_fixed is False:
                self._search_parameter_keys.append(key)
            elif isinstance(self.prior[key], Constraint):
                self._constraint_parameter_keys.append(key)
            elif isinstance(self.prior[key], DeltaFunction):
                # self.likelihood.parameters[key] = self.prior[key].sample()
                self._fixed_parameter_keys.append(key)

    # def _generate_result(self):
    #     result_kwargs = dict(
    #         label=self.label,
    #         # outdir=self.outdir,
    #         sampler=self.__class__.__name__.lower(),
    #         search_parameter_keys=self._search_parameter_keys,
    #         fixed_parameter_keys=self._fixed_parameter_keys,
    #         constraint_parameter_keys=self._constraint_parameter_keys,
    #         priors=self.prior,
    #         meta_data=self.metadata,
    #         injection_parameters=self.injection_parameters,
    #         sampler_kwargs=None,
    #         use_ratio=False,
    #     )
    #     self.result = Result(**result_kwargs)
    #
    #     if 'weights' in self.samples:
    #         self.result.nested_samples = pd.DataFrame(self.samples)
    #         # TODO: Calculate unweighted samples and store them in self.result.samples.
    #     else:
    #         self.result.samples = pd.DataFrame(self.samples)
    #
    #     self.result.log_evidence = self.log_evidence
    #
    #     # TODO: decide whether to run this, and whether to use it to generate
    #     #  additional parameters. This may depend on how pesummary processes the
    #     #  Results file.
    #     # self.result.samples_to_posterior()

    def _store_metadata(self, event_metadata: Optional[dict] = None):
        self.metadata = dict(
            model=self.model.metadata,
            event=event_metadata,
        )

    def _reset_sampler(self):
        """Clear out all data produced by self.run_sampler(), to prepare for the next
        sampler run."""
        self.result = None
        self.samples = None
        self.injection_parameters = None
        self.label = None
        self.log_evidence = None

    def importance_sample(self, num_processes: int = 1):

        if self.likelihood is None:
            raise KeyError("A likelihood is required to calculate importance weights.")
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

        # Proposal samples and associated log probability have already been calculated
        # using the stored model. These form a normalized probability distribution.

        log_prob_proposal = self.samples["log_prob"]
        theta = {
            k: v
            for k, v in self.samples.items()
            if k in self._search_parameter_keys or k in self._constraint_parameter_keys
        }

        # Calculate the (un-normalized) target density as prior times likelihood,
        # evaluated at the same sample points. The expensive part is the likelihood,
        # so we allow for multiprocessing.

        num_samples = len(log_prob_proposal)

        log_prior = self.prior.ln_prob(theta, axis=0)
        print(f"Calculating {num_samples} likelihoods.")
        t0 = time.time()
        log_likelihood = self.likelihood.log_likelihood_multi(
            theta, num_processes=num_processes
        )
        print(f"Done. This took {time.time() - t0:.2f} seconds.")

        # Calculate weights.

        log_weights = log_prior + log_likelihood - log_prob_proposal
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= np.mean(weights)

        self.samples['weights'] = weights
        self.samples['log_likelihood'] = log_likelihood
        self.samples['log_prior'] = log_prior
        # Note that self.samples['log_prob'] is the proposal log_prob, not the
        # importance_weighted log_prob.

        self.log_evidence = logsumexp(log_weights) - float(np.log(num_samples))

        # self._generate_result()

    def write_pesummary(self, filename):
        from pesummary.io import write
        from pesummary.utils.samples_dict import SamplesDict

        samples_dict = SamplesDict(self.samples)
        write(samples_dict, package=self._pesummary_package, file_format="hdf5",
              filename=filename)
        # TODO: Save much more information.

    def summary(self):
        print("Number of samples:", len(self.samples))
        print("log_evidence:", self.log_evidence)


class GNPESampler(ConditionalSampler):
    """
    Base class for GNPE sampler. It wraps a PosteriorModel, and must contain also an NPE
    sampler, which is used to generate initial samples.
    """

    def __init__(
        self,
        model: PosteriorModel,
        init_sampler: ConditionalSampler,
        num_iterations: int,
    ):
        """
        Parameters
        ----------
        model : PosteriorModel
        init_sampler : ConditionalSampler
            Used for generating initial samples
        num_iterations : int
            Number of GNPE iterations to be performed by sampler.
        """
        super().__init__(model)
        self.init_sampler = init_sampler
        self.num_iterations = num_iterations
        self.gnpe_parameters = None

    def _run_sampler(self, num_samples: int, context: dict):
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

    def _store_metadata(self, **kwargs):
        super()._store_metadata(**kwargs)
        self.metadata["init_model"] = self.init_sampler.model.metadata

        # TODO: Could also go in sampler_kwargs, which we don't use now.
        self.metadata["num_iterations"] = self.num_iterations
