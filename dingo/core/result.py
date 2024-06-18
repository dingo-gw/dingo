import copy
import math
import tempfile
import time

import numpy as np
from typing import Optional

import pandas as pd
from matplotlib import pyplot as plt
from scipy.constants import golden
from scipy.special import logsumexp
from bilby.core.prior import Constraint, DeltaFunction

from dingo.core.dataset import DingoDataset
from dingo.core.density import train_unconditional_density_estimator
from dingo.core.utils.misc import recursive_check_dicts_are_equal
from dingo.core.utils.plotting import plot_corner_multi

DATA_KEYS = [
    "samples",
    "context",
    "event_metadata",
    "importance_sampling_metadata",
    "log_evidence",
    "log_noise_evidence",
]


class Result(DingoDataset):
    """
    A dataset class to hold a collection of samples, implementing I/O, importance
    sampling, and unconditional flow training.

    Attributes:
        samples : pd.Dataframe
            Contains parameter samples, as well as (possibly) log_prob, log_likelihood,
            weights, log_prior, delta_log_prob_target.
        domain : Domain
            Should be implemented in a subclass.
        prior : PriorDict
            Should be implemented in a subclass.
        likelihood : Likelihood
            Should be implemented in a subclass.
        context : dict
            Context data from which the samples were produced (e.g., strain data, ASDs).
        metadata : dict
        event_metadata : dict
        log_evidence : float
        log_evidence_std : float (property)
        effective_sample_size, n_eff : float (property)
        sample_efficiency : float (property)
    """

    dataset_type = "core_result"

    def __init__(self, file_name=None, dictionary=None):
        self.event_metadata = None
        self.context = None
        self.samples = None
        self.log_noise_evidence = None
        super().__init__(
            file_name=file_name,
            dictionary=dictionary,
            data_keys=DATA_KEYS,
        )

        # Initialize as empty dict, so we can fill it up later.
        if self.importance_sampling_metadata is None:
            self.importance_sampling_metadata = {}

        self._build_prior()
        self._build_domain()
        if self.importance_sampling_metadata.get("updates"):
            self._rebuild_domain()

    @property
    def metadata(self):
        return self.settings

    @property
    def base_metadata(self):
        if self.metadata["train_settings"]["data"].get("unconditional", False):
            return self.metadata["base"]
        else:
            return self.metadata

    @property
    def injection_parameters(self):
        if self.context:
            return self.context.get("parameters")
        else:
            return None

    @property
    def constraint_parameter_keys(self):
        return [k for k, v in self.prior.items() if isinstance(v, Constraint)]

    @property
    def search_parameter_keys(self):
        return [
            k
            for k, v in self.prior.items()
            if (not isinstance(v, Constraint) and not isinstance(v, DeltaFunction))
        ]

    @property
    def fixed_parameter_keys(self):
        return [k for k, v in self.prior.items() if isinstance(v, DeltaFunction)]

    def _build_domain(self):
        self.domain = None

    def _build_prior(self):
        self.prior = None

    def _build_likelihood(self, **likelihood_kwargs):
        self.likelihood = None

    def reset_event(self, event_dataset):
        """
        Set the Result context and event_metadata based on an EventDataset.

        If these attributes already exist, perform a comparison to check for changes.
        Update relevant objects appropriately. Note that setting context and
        event_metadata attributes directly would not perform these additional checks and
        updates.

        Parameters
        ----------
        event_dataset: EventDataset
            New event to be used for importance sampling.
        """
        context = event_dataset.data
        event_metadata = event_dataset.settings

        if self.context is not None and not check_equal_dict_of_arrays(
            self.context, context
        ):
            # This is really just for notification. Actions are only taken if the
            # event metadata differ.
            print("\nNew event data differ from existing.")
        self.context = context

        if self.event_metadata is not None and self.event_metadata != event_metadata:
            print("Changes")
            print("=======")
            old_minus_new = dict(freeze(self.event_metadata) - freeze(event_metadata))
            print("Old event metadata:")
            for k in sorted(old_minus_new):
                print(f"  {k}:  {self.event_metadata[k]}")

            new_minus_old = dict(freeze(event_metadata) - freeze(self.event_metadata))
            print("New event metadata:")
            if self.importance_sampling_metadata.get("updates") is None:
                self.importance_sampling_metadata["updates"] = {}
            for k in sorted(new_minus_old):
                print(f"  {k}:  {event_metadata[k]}")
                self.importance_sampling_metadata["updates"][k] = event_metadata[k]

            self._rebuild_domain(verbose=True)
        self.event_metadata = event_metadata

    def _rebuild_domain(self, verbose=False):
        pass

    @property
    def num_samples(self):
        if self.samples is not None:
            return len(self.samples)
        else:
            return 0

    @property
    def effective_sample_size(self):
        if "weights" in self.samples:
            weights = self.samples["weights"]
            return np.sum(weights) ** 2 / np.sum(weights**2)
        else:
            return None

    @property
    def n_eff(self):
        return self.effective_sample_size

    @property
    def sample_efficiency(self):
        if "weights" in self.samples:
            return self.effective_sample_size / len(self.samples)
        else:
            return None

    @property
    def log_evidence_std(self):
        if "weights" in self.samples and self.log_evidence:
            return np.sqrt(
                (self.num_samples - self.n_eff) / (self.num_samples * self.n_eff)
            )
        else:
            return None

    @property
    def log_bayes_factor(self):
        if self.log_evidence and self.log_noise_evidence:
            return self.log_evidence - self.log_noise_evidence
        else:
            return None

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
        log_likelihood, log_prior, and weights. It also stores the log_evidence as an
        attribute.

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

        if "delta_log_prob_target" in self.samples.columns:
            delta_log_prob_target = self.samples["delta_log_prob_target"].to_numpy()
        else:
            delta_log_prob_target = 0.0

        # Calculate the (un-normalized) target density as prior times likelihood,
        # evaluated at the same sample points. The prior must be evaluated only for the
        # non-fixed (delta) parameters.
        param_keys_non_fixed = [
            k
            for k, v in self.prior.items()
            if not isinstance(v, (Constraint, DeltaFunction))
        ]
        theta_non_fixed = self.samples[param_keys_non_fixed]
        log_prior = self.prior.ln_prob(theta_non_fixed, axis=0)

        # select parameters in self.samples (required as log_prob and potentially gnpe
        # proxies are also stored in self.samples, but are not needed for the likelihood.
        # For evaluating the likelihood, we want to keep the fixed parameters.
        # TODO: replace by self.metadata["train_settings"]["data"]["inference_parameters"]
        param_keys = [k for k, v in self.prior.items() if not isinstance(v, Constraint)]
        theta = self.samples[param_keys]

        # The prior or delta_log_prob_target may be -inf for certain samples.
        # For these, we do not want to evaluate the likelihood, in particular because
        # it may not even be possible to generate signals outside the prior (e.g.,
        # for BH spins > 1).
        valid_samples = (log_prior + delta_log_prob_target) != -np.inf
        theta = theta.iloc[valid_samples]

        print(f"Calculating {len(theta)} likelihoods.")
        t0 = time.time()
        log_likelihood = self.likelihood.log_likelihood_multi(
            theta, num_processes=num_processes
        )
        print(f"Done. This took {time.time() - t0:.2f} seconds.")

        self.log_noise_evidence = self.likelihood.log_Zn
        self.samples["log_prior"] = log_prior
        self.samples.loc[valid_samples, "log_likelihood"] = log_likelihood
        self._calculate_evidence()

    def _calculate_evidence(self):
        """Calculate the Bayesian log evidence and sample weights.

        This is called at the end of importance sampling, when changing the prior,
        and when combining Results.

        The evidence

            Z = \\int d\\theta \\pi(\\theta) L(\\theta),

        where \\pi = prior, L = likelihood.

        For importance sampling, we estimate this using Monte Carlo integration using
        the proposal distribution q(\\theta),

            Z = \\int d\\theta q(\\theta) \\pi(\\theta) L(\\theta) / q(\\theta)
            \\sim (1/N) \\sum_i \\pi(\\theta_i) L(\\theta_i) / q(\\theta_i)

        where we are summing over samples \\theta_i \\sim q(\\theta).

        The integrand is just the importance weight (prior to any normalization). It
        is more numerically stable to evaluate \\log(Z),

            \\log Z \\sim \\log \\sum_i \\exp( \\log \\pi_i + \\log L_i - \\log q_i ) -
            \\log N
            = logsumexp ( log_weights ) - log N

        Notes
        -----
        * We use the logsumexp function, which is more numerically stable.
        * N = num_samples is the total number of samples (including the
            zero-weight samples).
        * q, \\pi, L must be distributions in the same parameter space (the same
            coordinates). We have undone any standardizations so this is the case.
        """
        if (
            "log_prob" in self.samples
            and "log_likelihood" in self.samples
            and "log_prior" in self.samples
        ):
            log_prob_proposal = self.samples["log_prob"]
            log_prior = self.samples["log_prior"]
            log_likelihood = self.samples["log_likelihood"]
            if "delta_log_prob_target" in self.samples:
                delta_log_prob_target = self.samples["delta_log_prob_target"]
            else:
                delta_log_prob_target = 0.0

            # *Un-normalized* log weights are needed to calculate evidence.
            log_weights = (
                log_prior
                + np.nan_to_num(log_likelihood)  # NaN = no log_likelihood evaluation
                + delta_log_prob_target
                - np.nan_to_num(
                    log_prob_proposal
                )  # NaN = outside prior so no synthetic
                # phase
            )
            self.log_evidence = logsumexp(log_weights) - np.log(self.num_samples)

            # Save the *normalized* weights.
            weights = np.exp(log_weights - np.max(log_weights))
            weights /= np.mean(weights)
            self.samples["weights"] = weights

    def sampling_importance_resampling(self, num_samples=None, random_state=None):
        """
        Generate unweighted posterior samples from weighted ones. New
        samples are sampled with probability proportional to the sample weight.
        Resampling is done with replacement, until the desired number of
        unweighted samples is obtained.

        Parameters
        ----------
        num_samples : int
            Number of samples to resample.
        random_state : int or None
            Sampling seed.

        Returns
        -------
        pd.Dataframe
            Unweighted samples
        """
        if num_samples is None:
            num_samples = len(self.samples)

        if num_samples > len(self.samples):
            raise ValueError("Cannot sample more points than in the weighted posterior")

        unweighted_samples = self.samples.sample(
            n=num_samples,
            weights=self.samples["weights"],
            replace=True,
            ignore_index=True,
            random_state=random_state,
        )
        return unweighted_samples.drop(["weights"], axis=1)

    def parameter_subset(self, parameters):
        """
        Return a new object of the same type, with only a subset of parameters. Drops
        all other columns in samples DataFrame as well (e.g., log_prob, weights).

        Parameters
        ----------
        parameters : list
            List of parameters to keep.

        Returns
        -------
        Result
        """
        result_dict = self.to_dictionary()
        result_dict["samples"] = self.samples[
            parameters
        ]  # Drop log_probs, weights, etc.
        return type(self)(dictionary=result_dict)

    def train_unconditional_flow(
        self,
        parameters,
        nde_settings: dict,
        train_dir: Optional[str] = None,
        threshold_std: Optional[float] = np.inf,
    ):
        """
        Train an unconditional flow to represent the distribution of self.samples.

        Parameters
        ----------
        parameters : list
            List of parameters over which to train the flow. Can be a subset of the
            existing parameters.
        nde_settings : dict
            Configuration settings for the neural density estimator.
        train_dir : Optional[str]
            Where to save the output of network training, e.g., logs, checkpoints. If
            not provide, a temporary directory is used.
        threshold_std : Optional[float]
            Drop samples more than threshold_std standard deviations away from the mean
            (in any parameter) before training the flow. This is meant to remove outlier
            samples.

        Returns
        -------
        PosteriorModel
        """
        sub_result = self.parameter_subset(parameters)

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

        temporary_directory = None
        if train_dir is None:
            temporary_directory = tempfile.TemporaryDirectory()
            train_dir = temporary_directory.name

        unconditional_model = train_unconditional_density_estimator(
            sub_result,
            nde_settings,
            train_dir,
        )

        if temporary_directory is not None:
            temporary_directory.cleanup()

        # unconditional_model.save_model("temp_model.pt")
        return unconditional_model

        # Note: self.gnpe_proxy_sampler.transform_post, and self.transform_post *must*
        # contain the SelectStandardizeRepackageParameters transformation, such that
        # the log_prob is correctly de-standardized!

    def print_summary(self):
        """
        Display the number of samples, and (if importance sampling is complete) the log
        evidence and number of effective samples.
        """
        print("Number of samples:", len(self.samples))
        if self.log_evidence is not None:
            print(
                f"Log(evidence): {self.log_evidence:.3f} +- {self.log_evidence_std:.3f}"
            )
            print(
                f"Effective samples {self.n_eff:.1f}: "
                f"(Sample efficiency = {100 * self.sample_efficiency:.2f}%)"
            )

    def split(self, num_parts):
        """
        Split the Result into a set of smaller results. The samples are evenly divided
        among the sub-results. Additional information (metadata, context, etc.) are
        copied into each.

        This is useful for splitting expensive tasks such as importance sampling across
        multiple jobs.

        Parameters
        ----------
        num_parts : int
            The number of parts to split the Result across.

        Returns
        -------
        list of sub-Results.
        """

        # Prepare a dictionary of all contents except the samples.
        dataset_dict_template = self.to_dictionary()
        del dataset_dict_template["samples"]

        part_size = self.num_samples // num_parts
        parts = []
        for i in range(num_parts):
            part_dict = copy.deepcopy(dataset_dict_template)
            if i < num_parts - 1:
                samples = self.samples.iloc[i * part_size : (i + 1) * part_size].copy()
            else:
                samples = self.samples.iloc[i * part_size :].copy()
            samples.reset_index(drop=True, inplace=True)
            part_dict["samples"] = samples
            part = type(self)(dictionary=part_dict)

            # Re-calculate evidence since it will differ for the new set of samples.
            part._calculate_evidence()
            parts.append(part)

        return parts

    @classmethod
    def merge(cls, parts):
        """
        Merge several Result instances into one. Check that they are compatible,
        in the sense of having the same metadata. Finally, calculate a new log evidence
        for the combined result.

        This is useful when recombining separate importance sampling jobs.

        Parameters
        ----------
        parts : list[Result]
            List of sub-Results to be combined.

        Returns
        -------
        Combined Result.
        """
        dataset_dict = parts[0].to_dictionary()
        del dataset_dict["log_evidence"]
        samples_parts = [dataset_dict.pop("samples")]

        for part in parts[1:]:
            part_dict = part.to_dictionary()
            del part_dict["log_evidence"]
            samples_parts.append(part_dict.pop("samples"))

            # Make sure we are not merging incompatible results. We deleted the
            # log_evidence since this can differ among the sub-results. Note that this
            # will also raise an error if files were created with different versions of
            # dingo.
            if not recursive_check_dicts_are_equal(part_dict, dataset_dict):
                raise ValueError("Results to be merged must have same metadata.")

        dataset_dict["samples"] = pd.concat(samples_parts, ignore_index=True)
        merged_result = cls(dictionary=dataset_dict)

        # Re-calculate the evidence based on the entire sample set.
        merged_result._calculate_evidence()
        return merged_result

    #
    # Plotting
    #

    def _cleaned_samples(self):
        """Return samples that exclude -inf and nan. This is used primarily for
        plotting."""

        # Do not plot any samples with -inf or nan. -inf can occur in
        # delta_log_prob_target or log_prior. nan occurs in log_likelihood when
        # log_likelihood not actually evaluated due to -inf in other columns (i.e.,
        # out of prior).

        return self.samples.replace(-np.inf, np.nan).dropna(axis=0)

    def plot_corner(
        self,
        parameters: list = None,
        filename: str = "corner.pdf",
        truths: dict = None,
        **kwargs,
    ):
        """
        Generate a corner plot of the samples.

        Parameters
        ----------
        parameters : list[str]
            List of parameters to include. If None, include all parameters.
            (Default: None)
        filename : str
            Where to save samples.
        truths : dict
            Dictionary of truth values to include.

        Other Parameters
        ----------------
        legend_font_size: int
            Font size of the legend.

        """
        theta = self._cleaned_samples()
        # delta_log_prob_target is not interesting so never plot it.
        theta = theta.drop(columns="delta_log_prob_target", errors="ignore")
        # corner cannot handle fixed parameters
        theta = theta.drop(columns=self.fixed_parameter_keys, errors="ignore")

        if "weights" in theta:
            weights = theta["weights"]
        else:
            weights = None
        # User option to plot specific parameters.
        if parameters:
            theta = theta[parameters]
        if truths is not None:
            kwargs["truths"] = [truths.get(k) for k in theta.columns]

        if weights is not None:
            plot_corner_multi(
                [theta, theta],
                weights=[None, weights.to_numpy()],
                labels=["Dingo", "Dingo-IS"],
                filename=filename,
                **kwargs,
            )
        else:
            plot_corner_multi(
                theta,
                labels=["Dingo"],
                filename=filename,
                **kwargs
            )

    def plot_log_probs(self, filename="log_probs.png"):
        """
        Make a scatter plot of the target versus proposal log probabilities. For the
        target, subtract off the log evidence.
        """
        theta = self._cleaned_samples()
        if "log_likelihood" in theta:
            log_prob_proposal = theta["log_prob"].to_numpy()
            if "delta_log_prob_target" in theta:
                log_prob_proposal -= theta["delta_log_prob_target"].to_numpy()

            log_prior = theta["log_prior"].to_numpy()
            log_likelihood = theta["log_likelihood"].to_numpy()

            x = log_prob_proposal
            y = log_prior + log_likelihood - self.log_evidence

            plt.figure(figsize=(6, 6))
            plt.xlabel("proposal log_prob")
            plt.ylabel("target log_prob - log_evidence")
            y_lower, y_upper = np.max(y) - 20, np.max(y)
            plt.ylim(y_lower, y_upper)
            n_below = len(np.where(y < y_lower)[0])
            plt.title(
                f"Target log probabilities\n({n_below} below {y_lower:.2f})\n"
                f"log(evidence) = {self.log_evidence:.3f} +- {self.log_evidence_std:.3f}"
            )
            plt.scatter(x, y, s=0.5)
            plt.plot([y_upper - 20, y_upper], [y_upper - 20, y_upper], color="black")
            plt.tight_layout()
            plt.savefig(filename)
        else:
            print("Results not importance sampled. Cannot produce log_prob plot.")

    def plot_weights(self, filename="weights.png"):
        """Make a scatter plot of samples weights vs log proposal."""
        theta = self._cleaned_samples()
        if "weights" in theta and "log_prob" in theta:
            x = theta["log_prob"].to_numpy()
            y = theta["weights"].to_numpy()
            y /= y.mean()

            plt.figure(figsize=(6 * golden, 6))
            plt.xlabel("proposal log_prob")
            plt.ylabel("weight (normalized)")
            y_lower = 1e-4
            y_upper = math.ceil(
                np.max(y) / 10 ** math.ceil(np.log10(np.max(y)) - 1)
            ) * 10 ** math.ceil(np.log10(np.max(y)) - 1)
            plt.ylim(y_lower, y_upper)
            n_below = len(np.where(y < y_lower)[0])
            plt.yscale("log")
            plt.title(
                f"Importance sampling weights\n({n_below} below {y_lower})\n"
                f"Effective samples: {self.n_eff:.0f} (Efficiency = "
                f"{100 * self.sample_efficiency:.2f}%)."
            )
            plt.scatter(x, y, s=0.5)
            plt.tight_layout()
            plt.savefig(filename)
        else:
            print("Results not importance sampled. Cannot plot weights.")


def check_equal_dict_of_arrays(a, b):
    if type(a) != type(b):
        return False

    if isinstance(a, dict):
        a_keys = set(a.keys())
        b_keys = set(b.keys())
        if a_keys != b_keys:
            return False

        for k in a_keys:
            if not check_equal_dict_of_arrays(a[k], b[k]):
                return False

        return True

    elif isinstance(a, np.ndarray):
        return np.array_equal(a, b)

    else:
        raise TypeError(f"Cannot compare items of type {type(a)}")


def freeze(d):
    if isinstance(d, dict):
        return frozenset((key, freeze(value)) for key, value in d.items())
    elif isinstance(d, list):
        return tuple(freeze(value) for value in d)
    return d
