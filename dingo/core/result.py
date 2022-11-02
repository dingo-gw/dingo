import tempfile
import time

import numpy as np
from typing import Optional
from scipy.special import logsumexp
from bilby.core.prior import Constraint

from dingo.core.dataset import DingoDataset
from dingo.core.density import train_unconditional_density_estimator


DATA_KEYS = [
    "samples",
    "context",
    "event_metadata",
    "log_evidence",
    "log_evidence_std",
]


class Result(DingoDataset):
    """
    A dataset class to hold a collection of samples, implementing I/O.

    Attributes
    ----------
    samples : pd.Dataframe
        Contains parameter samples, as well as (possibly) log_prob, log_likelihood,
        weights, log_prior.
    context : dict
        Context data on which the samples were produced (e.g., strain data, ASDs).
    log_evidence : float
    effective_sample_size : float
    """

    def __init__(self, file_name=None, dictionary=None):
        super().__init__(
            file_name=file_name,
            dictionary=dictionary,
            data_keys=DATA_KEYS,
        )

        # TODO: Do we need to copy this? Or set as a property.
        # TODO: Check that we really want to run all these lines.
        self.metadata = self.settings.copy()
        data_settings = self.metadata["train_settings"]["data"]
        self.inference_parameters = data_settings["inference_parameters"]

        self._build_prior()
        self._build_domain()

    @property
    def base_metadata(self):
        if self.metadata["train_settings"]["data"].get("unconditional", False):
            return self.metadata["base"]
        else:
            return self.metadata

    def _build_domain(self):
        self.domain = None

    def _build_prior(self):
        self.prior = None

    def _build_likelihood(self, **likelihood_kwargs):
        self.likelihood = None

    @property
    def effective_sample_size(self):
        if 'weights' in self.samples:
            weights = self.samples['weights']
            return np.sum(weights) ** 2 / np.sum(weights ** 2)

    @property
    def n_eff(self):
        return self.effective_sample_size

    @property
    def sample_efficiency(self):
        if 'weights' in self.samples:
            return self.effective_sample_size / len(self.samples)

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
        log_likelihood, log_prior, and weights. It also stores log_evidence,
        effective_sample_size and n_eff attributes.

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
            raise KeyError("Proposal samples are required for importance sampling.")
        if "log_prob" not in self.samples:
            raise KeyError(
                "Stored samples do not include log probability, which is "
                "needed for importance sampling. To obtain the log probability, "
                "it is necessary to train an unconditional flow based on the existing "
                "samples. This can then be sampled with log probability."
            )

        self._build_likelihood(**likelihood_kwargs)

        # Proposal samples and associated log probability have already been calculated
        # using the stored model. These form a normalized probability distribution.
        log_prob_proposal = self.samples["log_prob"].to_numpy()

        delta_log_prob_target = np.zeros(len(self.samples))
        if "delta_log_prob_target" in self.samples.columns:
            delta_log_prob_target = self.samples["delta_log_prob_target"].to_numpy()

        # select parameters in self.samples (required as log_prob and potentially gnpe
        # proxies are also stored in self.samples, but are not needed for the likelihood.
        # TODO: replace by self.metadata["train_settings"]["data"]["inference_parameters"]
        param_keys = [k for k, v in self.prior.items() if not isinstance(v, Constraint)]
        aux_keys = list(set(self.samples.keys()).difference(param_keys))
        theta = self.samples[param_keys]
        aux_params = self.samples[aux_keys]

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
        # size, so we are not losing anything useful. However, it is important to count
        # them in num_samples when computing the evidence, since they contribute to the
        # normalization of the proposal distribution.

        within_prior = (log_prior + delta_log_prob_target) != -np.inf
        num_samples = len(self.samples)
        if num_samples != np.sum(within_prior):
            print(
                f"Of {num_samples} samples, "
                f"{num_samples - np.sum(within_prior)} lie outside the prior. "
                f"Dropping these."
            )
            theta = theta.iloc[within_prior].reset_index(drop=True)
            aux_params = aux_params.iloc[within_prior].reset_index(drop=True)
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
        for k in aux_keys:
            self.samples[k] = aux_params[k]
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

        # self.n_eff = np.sum(weights) ** 2 / np.sum(weights ** 2)
        # # ESS computed with len(weights) in denominator instead of num_samples,
        # # since we are interested in ESS per *likelihood evaluation*, not per
        # # Dingo sample.
        # self.effective_sample_size = self.n_eff / len(weights)

        self.log_evidence = logsumexp(log_weights) - np.log(num_samples)
        log_weights_all = np.pad(
            log_weights - self.log_evidence,
            (num_samples - len(log_weights), 0),
            constant_values=-np.inf,
        )
        assert np.allclose(np.mean(np.exp(log_weights_all)), 1)
        # log_evidence_std = 1/sqrt(n) (evidence_std / evidence)

        # With the weights saved, the property self.n_eff is defined. The uncertainty
        # in the log evidence also depends on the original num_samples, so we have to
        # preserve this.
        self.log_evidence_std = np.sqrt(
            (num_samples - self.n_eff) / (num_samples * self.n_eff)
        )

    def subset(self, parameters):
        """
        Return a new object of the same time, with only a subset of parameters.

        Parameters
        ----------
        parameters : list
            List of parameters to keep.

        Returns
        -------
        Result
        """
        result_dict = self.to_dictionary()
        result_dict["samples"] = self.samples[parameters]  # Drop log_probs, weights, etc.
        return type(self)(dictionary=result_dict)

    def train_unconditional_flow(
        self,
        parameters,
        nde_settings: dict,
        train_dir: Optional[str] = None,
        threshold_std: Optional[float] = np.inf,
    ):
        sub_result = self.subset(parameters)

        # Filter outliers, as they decrease the performance of the density estimator.
        mean = np.mean(sub_result.samples, axis=0)
        std = np.std(sub_result.samples, axis=0)
        lower, upper = mean - threshold_std * std, mean + threshold_std * std
        inds = np.where(
            np.all((lower <= sub_result.samples), axis=1)
            * np.all((sub_result.samples <= upper), axis=1)
        )[0]
        if len(inds) / len(sub_result.samples) < 0.95:
            raise ValueError("Too many proxy samples outside of specified range.")
        sub_result.samples = sub_result.samples.iloc[inds]
        nde_settings["data"] = {"inference_parameters": parameters}

        # TODO: Combine these into single call. I had trouble getting the temporary
        #  directory to work without the context manager.
        if train_dir is not None:
            unconditional_model = train_unconditional_density_estimator(
                sub_result,
                nde_settings,
                train_dir,
            )
        else:
            with tempfile.TemporaryDirectory() as tmpdirname:
                unconditional_model = train_unconditional_density_estimator(
                    sub_result,
                    nde_settings,
                    tmpdirname,
                )
        # unconditional_model.save_model("temp_model.pt")
        return unconditional_model

        # Note: self.gnpe_proxy_sampler.transform_post, and self.transform_post *must*
        # contain the SelectStandardizeRepackageParameters transformation, such that
        # the log_prob is correctly de-standardized!

    def print_summary(self):
        print("Number of samples:", len(self.samples))
        if self.log_evidence is not None:
            print(
                f"Log(evidence): {self.log_evidence:.3f} +- {self.log_evidence_std:.3f}"
            )
            print(
                f"Effective samples {self.n_eff:.1f}: "
                f"(Sample efficiency = {100 * self.sample_efficiency:.2f}%)"
            )
