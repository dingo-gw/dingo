import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp
from torchvision.transforms import Compose

from dingo.core.likelihood import Likelihood
from dingo.core.models import PosteriorModel

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

    def __init__(self, model: PosteriorModel, likelihood: Optional[Likelihood] = None):
        """
        Parameters
        ----------
        model : PosteriorModel
        """
        self.model = model
        if 'base' in self.model.metadata:
            self.base_model_metadata = self.model.metadata['base']['model']
        else:
            self.base_model_metadata = self.model.metadata

        self.transforms_pre = Compose([])
        self.transforms_post = Compose([])
        self._search_parameter_keys = []
        self._constraint_parameter_keys = []
        self._fixed_parameter_keys = []
        self.inference_parameters = []
        self._build_prior()
        self.likelihood = likelihood
        self._reset_sampler()
        self._pesummary_package = "core"

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
        label: Optional[str] = None,
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
        label : str, optional
            Label for the event.
        event_metadata : dict, optional
            Metadata for data analyzed. Stored along with sample metadata, and can in
            principle influence any post-sampling parameter transformations (e.g.,
            sky position correction).
        """
        # Reset sampler and store all metadata associated with data.
        self._reset_sampler()
        if context is not None:
            self.injection_parameters = context.pop("parameters", None)
        self.label = label
        self._store_metadata(event_metadata=event_metadata)
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

        samples = pd.DataFrame(samples)
        samples.attrs = self.metadata
        self.samples = samples

    def _check_context(self, context: Optional[dict] = None):
        # TODO: Add some checks that the context is appropriate.
        pass

    def _post_correct(self, samples: dict):
        pass

    def _build_prior(self):
        pass

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

        log_prob_proposal = self.samples["log_prob"].to_numpy()
        theta = self.samples.drop(columns='log_prob')

        # TODO: Expand theta_all to include all parameters that make up the prior,
        #  including constraints. These might not be included within the inference
        #  parameters theta.
        theta_all = theta

        # Calculate the (un-normalized) target density as prior times likelihood,
        # evaluated at the same sample points. The expensive part is the likelihood,
        # so we allow for multiprocessing.

        breakpoint()
        num_samples = len(log_prob_proposal)

        log_prior = self.prior.ln_prob(theta_all, axis=0)
        print(f"Calculating {num_samples} likelihoods.")
        t0 = time.time()
        log_likelihood = self.likelihood.log_likelihood_multi(
            theta, num_processes=num_processes
        )
        print(f"Done. This took {time.time() - t0:.2f} seconds.")

        # Calculate weights.
        breakpoint()
        log_weights = log_prior + log_likelihood - log_prob_proposal
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= np.mean(weights)

        self.samples["weights"] = weights
        self.samples["log_likelihood"] = log_likelihood
        self.samples["log_prior"] = log_prior
        # Note that self.samples['log_prob'] is the proposal log_prob, not the
        # importance_weighted log_prob.

        self.log_evidence = logsumexp(log_weights) - float(np.log(num_samples))

        # self._generate_result()

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

    def summary(self):
        print("Number of samples:", len(self.samples))
        print("log_evidence:", self.log_evidence)


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

    def _store_metadata(self, **kwargs):
        super()._store_metadata(**kwargs)
        self.metadata["init_model"] = self.init_sampler.model.metadata

        # TODO: Could also go in sampler_kwargs, which we don't use now.
        self.metadata["num_iterations"] = self.num_iterations
