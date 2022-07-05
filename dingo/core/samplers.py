import time
from pathlib import Path
from typing import Optional, Union, Dict
import sys

import numpy
import numpy as np
import pandas as pd
import math
import torch
from scipy.special import logsumexp
from torchvision.transforms import Compose
import tempfile
from bilby.core.prior import Constraint

from dingo.core.models import PosteriorModel
from dingo.core.samples_dataset import SamplesDataset
from dingo.core.density import train_unconditional_density_estimator
from dingo.core.utils import torch_detach_to_cpu

# FIXME: transform below should be in core
from dingo.gw.transforms import SelectStandardizeRepackageParameters

#
# Sampler classes are motivated by the approach of Bilby.
#


class Sampler(object):
    """
    Sampler class that wraps a PosteriorModel. Allows for conditional and unconditional
    models.

    Draws samples from the model based on (optional) context data. Also implements
    importance sampling.

    Methods
    -------
        run_sampler
        log_prob
        importance_sample
        to_samples_dataset
        to_hdf5
        print_summary
    """

    def __init__(
        self,
        model: PosteriorModel = None,
        samples_dataset: SamplesDataset = None,
    ):
        """
        Parameters
        ----------
        model : PosteriorModel
        """
        if (model is None) + (samples_dataset is None) != 1:
            raise ValueError(
                "Sampler should be initialized with either a model, or samples_dataset."
            )

        self.model = model
        self.samples_dataset = samples_dataset

        if self.model is not None:
            self.metadata = self.model.metadata.copy()

            if "parameter_samples" in self.metadata["train_settings"]["data"]:
                self.unconditional_model = True
                # For unconditional models, the context will be stored with the model. It
                # is needed for calculating the likelihood for importance sampling.
                # However, it will not be used when sampling from the model, since it is
                # unconditional.
                self.context = self.model.context
            else:
                self.unconditional_model = False
                self.context = None

            if "base" in self.metadata:
                self.base_model_metadata = self.metadata["base"]
            else:
                self.base_model_metadata = self.metadata

            self.inference_parameters = self.metadata["train_settings"]["data"][
                "inference_parameters"
            ]

        elif self.samples_dataset is not None:
            self.metadata = None
            self.unconditional_model = True
            self.context = self.samples_dataset.context
            self.base_model_metadata = self.samples_dataset.settings
            self.samples = self.samples_dataset.samples
            data_settings = self.base_model_metadata["train_settings"]["data"]
            self.inference_parameters = data_settings["inference_parameters"]

        self.transform_pre = Compose([])
        self.transform_post = Compose([])
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

    @property
    def context(self):
        """Data on which to condition the sampler. For injections, there should be a
        'parameters' key with truth values."""
        return self._context

    @context.setter
    def context(self, value):
        if value is not None:
            self._check_context(value)
            if "parameters" in value:
                self.metadata["injection_parameters"] = value.pop("parameters")
        self._context = value

    def _check_context(self, context: Optional[dict] = None):
        # TODO: Add some checks that the context is appropriate.
        pass

    @property
    def event_metadata(self):
        """Metadata for data analyzed. Can in principle influence any post-sampling
        parameter transformations (e.g., sky position correction), as well as the
        likelihood detector positions."""
        return self.base_model_metadata.get("event")

    @event_metadata.setter
    def event_metadata(self, value):
        self.base_model_metadata["event"] = value

    def prepare_log_prob(self, *args, **kwargs):
        pass

    def _run_sampler(
        self,
        num_samples: int,
        context: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        if not self.unconditional_model:
            if context is None:
                raise ValueError("Context required to run sampler.")
            x = context.copy()
            x["parameters"] = {}
            x["extrinsic_parameters"] = {}

            # transforms_pre are expected to transform the data in the same way for each
            # requested sample. We therefore expand it across the batch *after*
            # pre-processing.
            x = self.transform_pre(context)
            x = x.expand(num_samples, *x.shape)
            x = [x]
            # The number of samples is expressed via the first dimension of x,
            # so we must pass num_samples = 1 to sample_and_log_prob().
            num_samples = 1
        else:
            if context is not None:
                raise ValueError(
                    "Context should not be passed to an unconditional sampler."
                )
            x = []

        # For a normalizing flow, we get the log_prob for "free" when sampling,
        # so we always include this. For other architectures, it may make sense to
        # have a flag for whether to calculate the log_prob.
        self.model.model.eval()
        with torch.no_grad():
            y, log_prob = self.model.model.sample_and_log_prob(
                *x, num_samples=num_samples
            )

        samples = self.transform_post({"parameters": y, "log_prob": log_prob})
        result = samples["parameters"]
        result["log_prob"] = samples["log_prob"]
        return result

    def run_sampler(
        self,
        num_samples: int,
        batch_size: Optional[int] = None,
        **kwargs,
    ):
        """
        Generates samples and stores them as class attribute.

        Allows for batched sampling, e.g., if limited by GPU memory. Actual sampling is
        performed by _run_sampler().

        Parameters
        ----------
        num_samples : int
            Number of samples requested.
        batch_size : int, optional
            Batch size for sampler.
        get_log_prob  :  bool = False
            If set, compute log_prob for each sample
        """
        self._reset_result()

        if self.samples_dataset is not None:
            # if self.samples_dataset is set, running the
            samples = self.samples_dataset.samples.sample(
                num_samples, ignore_index=True
            )
            samples = {k: np.array(samples[k]) for k in samples.columns}
            self._post_process(samples)
            self.samples = pd.DataFrame(samples)
            return

        print(f"Running sampler to generate {num_samples} samples.")
        t0 = time.time()
        if not self.unconditional_model:
            if self.context is None:
                raise ValueError("Context must be set in order to run sampler.")
            context = self.context
        else:
            context = None

        # Carry out batched sampling by calling _run_sample() on each batch and
        # consolidating the results.
        if batch_size is None:
            batch_size = num_samples
        full_batches, remainder = divmod(num_samples, batch_size)
        samples = [
            self._run_sampler(batch_size, context, **kwargs)
            for _ in range(full_batches)
        ]
        if remainder > 0:
            samples.append(self._run_sampler(remainder, context))
        samples = {p: torch.cat([s[p] for s in samples]) for p in samples[0].keys()}
        # get_log_prob = True
        # if get_log_prob and "log_prob" not in samples.keys():
        #     log_prob = self.log_prob_mc(
        #         samples, n_mc=1000, batch_size=batch_size,
        #     )
        # # samples = {p: torch.cat([s[p] for s in samples]) for p in k_parameters}
        samples = {k: v.cpu().numpy() for k, v in samples.items()}

        # Apply any post-sampling transformation to sampled parameters (e.g.,
        # correction for t_ref or sampling of synthetic phase), and place on CPU.
        self._post_process(samples)
        self.samples = pd.DataFrame(samples)
        print(f"Done. This took {time.time() - t0:.1f} s.")
        sys.stdout.flush()

    def log_prob(self, samples: pd.DataFrame) -> np.ndarray:
        """
        Calculate the model log probability at specific sample points.

        Parameters
        ----------
        samples : pd.DataFrame
            Sample points at which to calculate the log probability.

        Returns
        -------
        np.array of log probabilities.
        """
        if self.context is None and not self.unconditional_model:
            raise ValueError("Context must be set in order to calculate log_prob.")

        # This undoes any post-correction that would have been done to the samples,
        # before evaluating the log_prob. E.g., the t_ref / sky position correction.
        samples = samples.copy()
        self._post_process(samples, inverse=True)

        # Standardize the sample parameters and place on device.
        y = samples[self.inference_parameters].to_numpy()
        standardization = self.metadata["train_settings"]["data"]["standardization"]
        mean = np.array([standardization["mean"][p] for p in self.inference_parameters])
        std = np.array([standardization["std"][p] for p in self.inference_parameters])
        y = (y - mean) / std
        y = torch.from_numpy(y).to(device=self.model.device, dtype=torch.float32)

        if not self.unconditional_model:
            x = self.context.copy()
            x["parameters"] = {}
            x["extrinsic_parameters"] = {}

            # Context is the same for each sample. Expand across batch dimension after
            # pre-processing.
            x = self.transform_pre(self.context)
            x = x.expand(len(samples), *x.shape)
            x = [x]
        else:
            x = []

        self.model.model.eval()
        with torch.no_grad():
            log_prob = self.model.model.log_prob(y, *x)

        log_prob = log_prob.cpu().numpy()
        log_prob -= np.sum(np.log(std))

        # Pre-processing step may have included a log_prob with the samples.
        if "log_prob" in samples:
            log_prob += samples["log_prob"].to_numpy()

        return log_prob

    def log_prob_mc(self, *args, **kwargs):
        pass

    def _post_process(self, samples: Union[dict, pd.DataFrame], inverse: bool = False):
        pass

    def _build_prior(self):
        self.prior = None

    def _build_domain(self):
        self.domain = None

    def _build_likelihood(self, **likelihood_kwargs):
        self.likelihood = None

    def importance_sample(self, num_processes: int = 1, **likelihood_kwargs):
        """
        Calculate importance weights for samples.

        Importance sampling starts with samples have been generated from a proposal
        distribution q(theta), in this case a neural network model. Certain networks
        (i.e., non-GNPE) also provide the log probability of each sample,
        which is required for importance sampling.

        Given the proposal, we re-weight samples according to the (un-normalized)
        target distribution, which we take to be the likelihood L(theta) times the
        prior pi(theta). This gives sample weights

            w(theta) ~ pi(theta) L(theta) / q(theta),

        where the overall normalization does not matter (and we take to have mean 1).
        Since q(theta) enters this expression, importance sampling is only possible
        when we know the log probability of each sample.

        As byproducts, this method also estimates the evidence and effective sample
        size of the importance sampled points.

        This method modifies the samples pd.DataFrame in-place, adding new columns for
        log_likelihood, log_prior, and weights. It also stores log_evidence and
        effective_sample_size attributes.

        Parameters
        ----------
        num_processes : int
            Number of parallel processes to use when calculating likelihoods. (This is
            the most expensive task.)
        likelihood_kwargs : dict
            kwargs that are forwarded to the likelihood constructor. E.g., options for
            marginalization.
        """

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

        delta_log_prob_target = 0
        if "delta_log_prob_target" in self.samples.columns:
            delta_log_prob_target = self.samples["delta_log_prob_target"].to_numpy()

        # select parameters in self.samples (required as log_prob and potentially gnpe
        # proxies are also stored in self.samples, but are not needed for the likelihood.
        param_keys = [k for k, v in self.prior.items() if not isinstance(v, Constraint)]
        theta = self.samples[param_keys]

        # Calculate the (un-normalized) target density as prior times likelihood,
        # evaluated at the same sample points.
        log_prior = self.prior.ln_prob(theta, axis=0)

        # Check whether any constraints are violated that involve parameters not
        # already present in theta.
        constraints = self.prior.evaluate_constraints(theta)
        np.putmask(log_prior, constraints == 0, -np.inf)

        # The prior may evaluate to -inf for certain samples. For these, we do not want
        # to evaluate the likelihood, in particular because it may not even be possible
        # to generate data outside the prior (e.g., for BH spins > 1). Since there is
        # no point in keeping these samples, we simply drop them; this means we do not
        # have to make special exceptions for outside-prior samples elsewhere in the
        # code. They do not contribute directly to the evidence or the effective sample
        # size, so we are not losing anything useful.

        within_prior = (log_prior + delta_log_prob_target) != -np.inf
        num_samples = len(self.samples)
        if num_samples != np.sum(within_prior):
            print(
                f"Of {num_samples} samples, "
                f"{num_samples - np.sum(within_prior)} lie outside the prior. "
                f"Dropping these."
            )
            theta = theta.iloc[within_prior].reset_index(drop=True)
            log_prob_proposal = log_prob_proposal[within_prior]
            log_prior = log_prior[within_prior]
            delta_log_prob_target = delta_log_prob_target[within_prior]

        print(f"Calculating {len(theta)} likelihoods.")
        t0 = time.time()
        log_likelihood = self.likelihood.log_likelihood_multi(
            theta, num_processes=num_processes
        )
        print(f"Done. This took {time.time() - t0:.2f} seconds.")

        # Calculate weights and normalize them to have mean 1.
        log_weights = (
            log_prior + log_likelihood + delta_log_prob_target - log_prob_proposal
        )
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= np.mean(weights)

        self.samples = theta
        self.samples["log_prob"] = log_prob_proposal  # Proposal log_prob, not target!
        self.samples["weights"] = weights
        self.samples["log_likelihood"] = log_likelihood
        self.samples["log_prior"] = log_prior
        # self.samples["delta_log_prob_target"] = delta_log_prob_target

        # The evidence
        #           Z = \int d\theta \pi(\theta) L(\theta),
        #
        #                   where   \pi = prior,
        #                           L = likelihood.
        #
        # For importance sampling, we estimate this using Monte Carlo integration using
        # the proposal distribution q(\theta),
        #
        #           Z = \int d\theta q(\theta) \pi(\theta) L(\theta) / q(\theta)
        #             ~ (1/N) \sum_i \pi(\theta_i) L(\theta_i) / q(\theta_i)
        #
        #                   where we are summing over samples \theta_i ~ q(\theta).
        #
        # The integrand is just the importance weight (prior to any normalization). It
        # is more numerically stable to evaluate log(Z),
        #
        #           log Z ~ \log \sum_i \exp( log \pi_i + log L_i - log q_i ) - log N
        #                 = logsumexp ( log_weights ) - log N
        #
        # Notes
        # -----
        #   * We use the logsumexp functions, which is more numerically stable.
        #   * N = num_samples is the *original* number of samples (including the
        #     zero-weight ones that we dropped).
        #   * q, \pi, L must be distributions in the same parameter space (the same
        #     coordinates). We have undone any standardizations so this is the case.

        self.log_evidence = logsumexp(log_weights) - np.log(num_samples)
        self.effective_sample_size = np.sum(weights) ** 2 / np.sum(weights ** 2)

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


class UnconditionalSampler(Sampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.unconditional_model = True
        self._initialize_transforms()

    def _initialize_transforms(self):
        # Postprocessing transform only:
        #   * De-standardize data and extract inference parameters. Be careful to use
        #     the standardization of the correct model, not the base model.
        self.transform_post = SelectStandardizeRepackageParameters(
            {"inference_parameters": self.inference_parameters},
            self.metadata["train_settings"]["data"]["standardization"],
            inverse=True,
            as_type="dict",
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
        self.gnpe_parameters = None  # Should be set in subclass.
        self.gnpe_proxy_sampler = None
        self.log_prob_correction = None  # log_prob correction, accounting for std
        self.gnpe_proxy_sampler = None

    @property
    def init_sampler(self):
        return self._init_sampler

    @init_sampler.setter
    def init_sampler(self, value):
        self._init_sampler = value
        self.metadata["init_model"] = self._init_sampler.model.metadata

    @property
    def num_iterations(self):
        """The number of GNPE iterations to perform when sampling."""
        return self._num_iterations

    @num_iterations.setter
    def num_iterations(self, value):
        self._num_iterations = value
        self.metadata["num_iterations"] = self._num_iterations

    def _kernel_log_prob(self, samples):
        raise NotImplementedError("To be implemented in subclass.")

    def prepare_log_prob(
        self,
        num_samples: int,
        nde_settings: dict,
        batch_size: Optional[int] = None,
        threshold_std: Optional[float] = np.inf,
    ):
        """
        Prepare gnpe sampling with log_prob. This is required, since in its vanilla
        form gnpe does not provide the density for its samples.

        Specifically, we train an unconditional neural density estimator (nde) for the
        gnpe proxies. This requires running the gnpe sampler till convergence, and
        extracting the gnpe proxies after the final gnpe iteration. The nde is trained
        to match the distribution over gnpe proxies, which provides a way of rapidly
        sampling (converged!) gnpe proxies *and* evaluating the log_prob.

        After this preparation step, self.run_sampler can leverage
        self.gnpe_proxy_sampler (which is based on the aforementioned trained nde) to
        sample gnpe proxies, such that one gnpe iteration is sufficient. The log_prob of
        the samples in the *joint* space (inference parameters + gnpe proxies) is then
        simply given by the sum of the corresponding log_probs (from self.model and
        self.gnpe_proxy_sampler.model).

        Parameters
        ----------
        num_samples: int
            number of samples for training of nde
        batch_size: int = None
            batch size for sampler
        threshold_std: float = np.inf
            gnpe proxies deviating by more then threshold_std standard deviations from
            the proxy mean (along any axis) are discarded.
        """
        self.run_sampler(num_samples, batch_size)
        gnpe_proxy_keys = [k for k in self.samples.keys() if k.startswith("GNPE:")]
        gnpe_proxy_pd = self.samples[gnpe_proxy_keys].rename(columns=lambda x: x[5:])
        gnpe_proxy_dataset = SamplesDataset(dictionary={"samples": gnpe_proxy_pd})
        # filter outliers, as they decrease the performance of the density estimator
        mean = np.mean(gnpe_proxy_dataset.samples, axis=0)
        std = np.std(gnpe_proxy_dataset.samples, axis=0)
        lower, upper = mean - threshold_std * std, mean + threshold_std * std
        inds = np.where(
            np.all((lower <= gnpe_proxy_dataset.samples), axis=1)
            * np.all((gnpe_proxy_dataset.samples <= upper), axis=1)
        )[0]
        if len(inds) / len(gnpe_proxy_dataset.samples) < 0.95:
            raise ValueError("Too many proxy samples outside of specified range.")
        gnpe_proxy_dataset.samples = gnpe_proxy_dataset.samples.iloc[inds]
        nde_settings["data"] = {"inference_parameters": [k[5:] for k in gnpe_proxy_keys]}
        nde_settings["training"]["device"] = str(self.model.device)
        with tempfile.TemporaryDirectory() as tmpdirname:
            gnpe_proxy_model = train_unconditional_density_estimator(
                gnpe_proxy_dataset,
                nde_settings,
                tmpdirname,
            )
        self.gnpe_proxy_sampler = UnconditionalSampler(model=gnpe_proxy_model)

        # Get correction for log_prob, which arises due to the standardization
        # underlying the density estimators self.model and self.gnpe_proxy_sampler.model.
        self.log_prob_correction = 0
        for sampler in [self, self.gnpe_proxy_sampler]:
            std = sampler.metadata["train_settings"]["data"]["standardization"]["std"]
            std = np.array([std[p] for p in sampler.inference_parameters])
            self.log_prob_correction += np.sum(np.log(std))

    def _run_sampler(
        self,
        num_samples: int,
        context: Optional[dict] = None,
        use_gnpe_proxy_sampler=False,
    ) -> dict:
        if context is None:
            raise ValueError("self.context must be set to run sampler.")
        use_gnpe_proxy_sampler = self.gnpe_proxy_sampler is not None

        data_ = self.init_sampler.transform_pre(context)

        if not use_gnpe_proxy_sampler:
            # Run gnpe iterations to jointly infer gnpe proxies and inference parameters.
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

                x = self.transform_pre(x)
                x["parameters"] = self.model.sample(x["data"], x["context_parameters"])
                x = self.transform_post(x)
                samples = x["parameters"]
        else:
            # Infer gnpe proxies based on trained model in gnpe_proxy sampler.
            # With this, we can infer the inference parameters with a single pass
            # through the gnpe network.
            d = data_.clone()

            gnpe_proxies_in = self.gnpe_proxy_sampler._run_sampler(
                num_samples=num_samples, batch_size=num_samples
            )
            log_prob_gnpe_proxies = gnpe_proxies_in.pop("log_prob")

            x = {
                "extrinsic_parameters": {},
                "parameters": {},
                "gnpe_proxies_in": gnpe_proxies_in,
                "data": d.expand(num_samples, *d.shape),
            }
            x = self.transform_pre(x)
            with torch.no_grad():
                x["parameters"], log_prob = self.model.model.sample_and_log_prob(
                    x["data"],
                    x["context_parameters"],
                )
            x = self.transform_post(x)

            samples = x["parameters"]
            samples["log_prob"] = (
                log_prob + log_prob_gnpe_proxies + self.log_prob_correction
            )

            # The log_prob returned by gnpe is not just the log_prob over parmeters
            # theta, but instead the log_prob in the *joint* space q(theta,theta^|x),
            # including the proxies theta^. For importance sampling this means,
            # that the target density is
            #
            #       p(theta,theta^|x) = p(theta^|theta) * p(theta|x).
            #
            # We compute log[p(theta^|theta)] below and store it as
            # samples["delta_log_prob_target"], such that for importance sampling we
            # only need to evaluate log[p(theta|x)] and add this correction.
            all_params = {**x["extrinsic_parameters"], **samples, **x["gnpe_proxies"]}
            all_params = {k: torch_detach_to_cpu(v) for k, v in all_params.items()}
            kernel_log_prob = self._kernel_log_prob(all_params)
            samples["delta_log_prob_target"] = torch.Tensor(kernel_log_prob)

        # add gnpe parameters:
        for k, v in x["gnpe_proxies"].items():
            assert k.endswith("_proxy")
            samples["GNPE:" + k] = x["gnpe_proxies"][k]

        return samples
