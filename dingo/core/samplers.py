import time
from pathlib import Path
from typing import Optional, Union, Dict

import numpy
import numpy as np
import pandas as pd
import math
import torch
from scipy.special import logsumexp
from torchvision.transforms import Compose
import tempfile

from dingo.core.models import PosteriorModel
from dingo.core.samples_dataset import SamplesDataset
from dingo.core.density import (
    train_unconditional_density_estimator,
    get_default_nde_settings_3d,
)

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

    def __init__(self, model: PosteriorModel):
        """
        Parameters
        ----------
        model : PosteriorModel
        """
        self.model = model
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

        self.transform_pre = Compose([])
        self.transform_post = Compose([])
        self.inference_parameters = self.metadata["train_settings"]["data"][
            "inference_parameters"
        ]
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
        # get_log_prob: bool = False,
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
        theta = self.samples.drop(columns="log_prob")

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

        within_prior = log_prior != -np.inf
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

        print(f"Calculating {len(theta)} likelihoods.")
        t0 = time.time()
        log_likelihood = self.likelihood.log_likelihood_multi(
            theta, num_processes=num_processes
        )
        print(f"Done. This took {time.time() - t0:.2f} seconds.")

        # Calculate weights and normalize them to have mean 1.
        log_weights = log_prior + log_likelihood - log_prob_proposal
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= np.mean(weights)

        self.samples = theta
        self.samples["log_prob"] = log_prob_proposal  # Proposal log_prob, not target!
        self.samples["weights"] = weights
        self.samples["log_likelihood"] = log_likelihood
        self.samples["log_prior"] = log_prior

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

    def prepare_log_prob(
        self,
        num_samples: int,
        batch_size: Optional[int] = None,
    ):
        self.run_sampler(num_samples, batch_size)
        gnpe_proxy_keys = [k for k in self.samples.keys() if k.startswith("GNPE:")]
        gnpe_proxy_pd = self.samples[gnpe_proxy_keys].rename(columns=lambda x: x[5:])
        gnpe_proxy_dataset = SamplesDataset(dictionary={"samples": gnpe_proxy_pd})
        # samples_dict = self.samples[
        #     [k for k in self.samples.keys() if k.startswith("GNPE:")]
        # ].to_dict()
        # samples_dataset = SamplesDataset(dictionary=samples_dict)
        with tempfile.TemporaryDirectory() as tmpdirname:
            settings = get_default_nde_settings_3d(
                device=str(self.model.device),
                inference_parameters=[k[5:] for k in gnpe_proxy_keys],
            )
            gnpe_proxy_model = train_unconditional_density_estimator(
                gnpe_proxy_dataset,
                settings,
                tmpdirname,
            )
            # from dingo.gw.inference.gw_samplers import GWSamplerUnconditional
            self.gnpe_proxy_sampler = UnconditionalSampler(model=gnpe_proxy_model)

        sampler = self.gnpe_proxy_sampler
        std_gnpe_proxy_sampler = np.array(
            [
                sampler.metadata["train_settings"]["data"]["standardization"]["std"][p]
                for p in sampler.inference_parameters
            ]
        )
        sampler = self
        std_sampler = np.array(
            [
                sampler.metadata["train_settings"]["data"]["standardization"]["std"][p]
                for p in sampler.inference_parameters
            ]
        )
        self.log_prob_correction = -np.sum(np.log(std_sampler)) - np.sum(
            np.log(std_gnpe_proxy_sampler)
        )

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
                log_prob + log_prob_gnpe_proxies - self.log_prob_correction
            )

        # add gnpe parameters:
        for k, v in x["gnpe_proxies"].items():
            assert k.endswith("_proxy")
            samples["GNPE:" + k] = x["gnpe_proxies"][k]

        #
        # get_log_prob = True
        # n_log_prob = 10
        # if get_log_prob:
        #     extrinsic_parameters = x["extrinsic_parameters"]
        #     for i_log_prob in range(1, n_log_prob+1):
        #         # Shift everything by i along the batch dimension.
        #         # This has the effect of evaluating the log_prob for random samples of
        #         # the proxies, which corresponds to a Monte Carlo integration.
        #         extrinsic_parameters_shifted = torch.roll(
        #             extrinsic_parameters[k], shifts=i, dims=0
        #         )
        #         x["extrinsic_parameters"] = {
        #             k: extrinsic_parameters_shifted for k in self.gnpe_parameters
        #         }
        #         # remove context parameters for safety, we want these to be sampled
        #         # from the *shifted* extrinsic parameters, so we don't want to allow
        #         # any further information to sneak in
        #         x.pop("context_parameters", None)
        #         d = data_.clone()
        #         x["data"] = d.expand(num_samples, *d.shape)
        #
        #         x = self.transform_pre(x)
        #         x["parameters"] =
        #         x["parameters"] = self.model.sample(x["data"], x["context_parameters"])
        #         x = self.transform_post(x)
        #         # theta, log_prob = self.model.model.sample_and_log_prob(
        #         #     x["data"], x["context_parameters"]
        #         # )
        #         # print(torch.abs(log_prob - log_prob_ref) / torch.abs(log_prob))
        #         x["extrinsic_parameters"] = {
        #             k: x["extrinsic_parameters"][k] for k in self.gnpe_parameters
        #         }
        #         d = data_.clone()
        #         x["data"] = d.expand(num_samples, *d.shape)
        #
        #         x = self.transform_pre(x)
        #         self.model.model.log_prob(
        #             x["parameters"]
        #         )

        return samples

    def log_prob_mc(
        self,
        samples: Dict[str, torch.Tensor],
        n_mc: int = 100,
        batch_size: int = 100,
    ) -> np.ndarray:
        parameters, gnpe_proxies, gnpe_parameters = {}, {}, {}
        for k, v in samples.items():
            if not k.startswith("GNPE:"):
                parameters[k] = v
            elif k.endswith("_proxy"):
                gnpe_proxies[k[len("GNPE:") :]] = v
            else:
                gnpe_parameters[k[len("GNPE:") :]] = v
        num_samples = len(samples[k])
        # gnpe_keys = [k[:-len("_proxy")] for k in gnpe_proxies.keys()]
        gnpe_proxies = torch.cat(
            [gnpe_proxies[k + "_proxy"][:, None] for k in self.gnpe_parameters],
            dim=1,
        )
        gnpe_parameters = torch.cat(
            [gnpe_parameters[k][:, None] for k in self.gnpe_parameters],
            dim=1,
        )

        import pandas as pd

        params = pd.DataFrame(samples)
        n_train = int(len(params) * 0.9)
        x = params[["geocent_time", "GNPE:H1_time_proxy", "GNPE:L1_time_proxy"]]
        x_train, x_test = x[:n_train], x[n_train:]

        import yaml
        from os.path import join
        from dingo.core.density import train_unconditional_density_estimator
        from types import SimpleNamespace

        xs = SimpleNamespace()
        xs.samples = x
        xs.context = None
        xs.settins = None
        nde_dir = "/Users/maxdax/Documents/Projects/GW-Inference/dingo/utils/tmp_dir/"
        with open(join(nde_dir, "nde_settings.yaml")) as f:
            nde_settings = yaml.safe_load(f)
        nde = train_unconditional_density_estimator(xs, nde_settings, nde_dir)

        from sklearn.neighbors import KernelDensity

        x = np.array(gnpe_proxies)
        n_train = int(len(x) * 0.9)
        for bw in [1e-4, 1e-3, 1e-2, 1e-1]:
            x_train, x_test = x[:n_train], x[n_train:]
            kde = KernelDensity(bandwidth=bw, kernel="gaussian")
            kde.fit(x_train)
            train_score = kde.score_samples(x_train)
            test_score = kde.score_samples(x_test)
            print(
                f"Bandwith: {bw}\t"
                f"train score: {np.mean(train_score):.2f} ({np.min(train_score):.2f})\t"
                f"test score: {np.mean(test_score):.2f} ({np.min(test_score):.2f})"
            )

        # for bw in [1e-4, 1e-3, 1e-2, 1e-1]:
        #     kde = KernelDensity(bandwidth=bw, kernel='gaussian')
        #     kde.fit(x_train)
        #     train_score = kde.score_samples(x_train)
        #     test_score = kde.score_samples(x_test)
        #     print(
        #         f"Bandwith: {bw}\t"
        #         f"train score: {np.mean(train_score):.2f} ({np.min(train_score):.2f})\t"
        #         f"test score: {np.mean(test_score):.2f} ({np.min(test_score):.2f})"
        #     )
        kde = KernelDensity(bandwidth=1e-3, kernel="gaussian")
        kde.fit(x_train)
        from chainconsumer import ChainConsumer
        import scipy

        c = ChainConsumer()
        c.add_chain(x_test, color="blue", name="x_test")
        c.add_chain(kde.sample(len(x_test)), color="orange", name="kde")
        N = 2
        c.configure(
            linestyles=["-"] * N,
            linewidths=[1.5] * N,
            sigmas=[np.sqrt(2) * scipy.special.erfinv(x) for x in [0.5, 0.9]],
            shade=[False] + [True] * (N - 1),
            shade_alpha=0.3,
            bar_shade=False,
            label_font_size=10,
            tick_font_size=10,
            usetex=False,
            legend_kwargs={"fontsize": 30},
            kde=0.7,
        )
        c.plotter.plot(
            filename="/Users/maxdax/Documents/Projects/GW-Inference/dingo/utils/tmp.pdf"
        )

        log_probs = []

        n_per_batch = batch_size // n_mc  # number of likelihood approxmations per batch
        num_samples = 30
        for idx in range(math.ceil(num_samples / n_per_batch)):
            theta_batch = {
                k: v[idx * n_per_batch : (idx + 1) * n_per_batch]
                for k, v in parameters.items()
            }

            # select only gnpe_proxies for which q(theta | x, proxies) != 0
            gnpe_parameters_batch = gnpe_parameters[
                idx * n_per_batch : (idx + 1) * n_per_batch
            ]
            diff_gnpe_parameters_proxies = torch.abs(
                # expand is a view, but the diff will occupy memory
                gnpe_proxies.expand(n_per_batch, *gnpe_proxies.shape)
                - gnpe_parameters_batch[:, None, :].expand(-1, len(gnpe_proxies), -1)
            )
            max_diff = torch.Tensor([0.001, 0.001])
            gnpe_mask = (
                torch.sum(
                    (diff_gnpe_parameters_proxies < max_diff[None, None, :]),
                    dim=2,
                )
                == len(max_diff)
            )
            mc_indices = []
            hot_fraction = []
            for idx_theta in range(len(gnpe_mask)):
                hot_indices = torch.where(gnpe_mask[idx_theta])[0]
                hot_fraction.append(len(hot_indices) / gnpe_mask.shape[-1])
                mc_indices.append(hot_indices[torch.randperm(len(hot_indices))[:n_mc]])

            hot_fraction = torch.Tensor(hot_fraction)
            mc_indices = torch.cat(mc_indices)
            gnpe_proxies_in = gnpe_proxies[mc_indices]

            data_ = self.init_sampler.transform_pre(self.context)
            x = {
                "extrinsic_parameters": {},
                "parameters": {},
            }
            # x["extrinsic_parameters"] = {
            #     k: x["extrinsic_parameters"][k] for k in self.gnpe_parameters
            # }
            d = data_.clone()
            x["data"] = d.expand(len(gnpe_proxies_in), *d.shape)
            x["gnpe_proxies_in"] = {
                k + "_proxy": gnpe_proxies_in[:, idx]
                for idx, k in enumerate(self.gnpe_parameters)
            }
            x = self.transform_pre(x)
            # The above provides us with the context information that will be passed to
            # the NDE, x["data"], x["context_parameters"].
            # Now we need to prepare where to evaluate the flow, which requires
            # standardizing the parameters and optionally applying the inverse GNPE
            # postcorrection transformations.

            theta_batch_n = {
                k: v.repeat_interleave(n_mc) for k, v in theta_batch.items()
            }
            y = self.transform_post_inverse(
                {
                    # repeat each tensor in theta_batch n_mc times
                    "parameters": theta_batch_n,
                    "extrinsic_parameters": x["extrinsic_parameters"],
                }
            )
            y = y["inference_parameters"]

            log_prob = self.model.model.log_prob(
                y, x["data"], x["context_parameters"]
            ).detach()
            prob = torch.exp(log_prob)
            # Need to average the probability. Multiply it with hot_fraction, because
            # only this fraction has non-zero probs
            prob_avg = torch.mean(prob.reshape(-1, n_mc), dim=1) * hot_fraction

            log_probs.append(torch.log(prob_avg).cpu().numpy())

        return numpy.concatenate(log_probs)

        # Step 1: filter valid proxies in self.proxies, save ratio n_valid / n_all
        # Step 2: sample num_samples proxies for each sample
        # Step 3: Prepare data with the proxies: subtract H1 proxy, standardize
        # Step 4: Evaluate log_prob

    # with GNPE the log_prob becomes intractable, so we drop the log_prob method
    log_prob = None
