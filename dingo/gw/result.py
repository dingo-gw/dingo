import copy
import time
from typing import Optional

import numpy as np
import yaml
from bilby.core.prior import Uniform, Constraint, PriorDict

from dingo.core.density import (
    interpolated_sample_and_log_prob_multi,
    interpolated_log_prob_multi,
)
from dingo.core.multiprocessing import apply_func_with_multiprocessing
from dingo.core.result import Result as CoreResult
from dingo.gw.conversion import change_spin_conversion_phase
from dingo.gw.domains import build_domain
from dingo.gw.gwutils import get_extrinsic_prior_dict, get_window_factor
from dingo.gw.likelihood import StationaryGaussianGWLikelihood
from dingo.gw.prior import build_prior_with_defaults


RANDOM_STATE = 150914


class Result(CoreResult):
    """
    A dataset class to hold a collection of gravitational-wave parameter samples and
    perform various operations on them.

    Compared to the base class, this class implements the domain, prior,
    and likelihood. It also includes a method for sampling the binary reference phase
    parameter based on the other parameters and the likelihood.

    Attributes:
        samples : pd.Dataframe
            Contains parameter samples, as well as (possibly) log_prob, log_likelihood,
            weights, log_prior, delta_log_prob_target.
        domain : Domain
            The domain of the data (e.g., FrequencyDomain), needed for calculating
            likelihoods.
        prior : PriorDict
            The prior distribution, used for importance sampling.
        likelihood : Likelihood
            The Likelihood object, needed for importance sampling.
        context : dict
            Context data from which the samples were produced (e.g., strain data, ASDs).
        metadata : dict
            Metadata inherited from the Sampler object. This describes the neural
            networks and sampling settings used.
        event_metadata : dict
            Metadata for the event analyzed, including time, data conditioning, channel,
            and detector information.
        log_evidence : float
            Calculated log(evidence) after importance sampling.
        log_evidence_std : float (property)
            Standard deviation of the log(evidence)
        effective_sample_size, n_eff : float (property)
            Number of effective samples, (\\sum_i w_i)^2 / \\sum_i w_i^2
        sample_efficiency : float (property)
            Number of effective samples / Number of samples
        synthetic_phase_kwargs : dict
            kwargs describing the synthetic phase sampling.
    """

    dataset_type = "gw_result"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def synthetic_phase_kwargs(self):
        return self.importance_sampling_metadata.get("synthetic_phase")

    @synthetic_phase_kwargs.setter
    def synthetic_phase_kwargs(self, value):
        self.importance_sampling_metadata["synthetic_phase"] = value

    @property
    def time_marginalization_kwargs(self):
        return self.importance_sampling_metadata.get("time_marginalization")

    @time_marginalization_kwargs.setter
    def time_marginalization_kwargs(self, value):
        self.importance_sampling_metadata["time_marginalization"] = value

    @property
    def phase_marginalization_kwargs(self):
        return self.importance_sampling_metadata.get("phase_marginalization")

    @phase_marginalization_kwargs.setter
    def phase_marginalization_kwargs(self, value):
        self.importance_sampling_metadata["phase_marginalization"] = value

    @property
    def calibration_marginalization_kwargs(self):
        return self.importance_sampling_metadata.get("calibration_marginalization")

    @calibration_marginalization_kwargs.setter
    def calibration_marginalization_kwargs(self, value):
        self.importance_sampling_metadata["calibration_marginalization"] = value

    @property
    def f_ref(self):
        return self.base_metadata["dataset_settings"]["waveform_generator"]["f_ref"]

    @property
    def approximant(self):
        return self.base_metadata["dataset_settings"]["waveform_generator"][
            "approximant"
        ]

    @property
    def interferometers(self):
        return list(self.context["waveform"].keys())

    @property
    def t_ref(self):
        # The detector reference positions during likelihood evaluation should be
        # based on the event time, since any post-correction to account for the training
        # reference time has already been applied to the samples.
        if self.event_metadata is not None and "time_event" in self.event_metadata:
            return self.event_metadata["time_event"]
        else:
            return self.base_metadata["train_settings"]["data"]["ref_time"]

    def _build_domain(self):
        """
        Construct the domain object based on model metadata. Includes the window factor
        needed for whitening data.

        Called by __init__() immediately after _build_prior().
        """
        self.domain = build_domain(self.base_metadata["dataset_settings"]["domain"])

        data_settings = self.base_metadata["train_settings"]["data"]
        if "domain_update" in data_settings:
            self.domain.update(data_settings["domain_update"])

        self.domain.window_factor = get_window_factor(data_settings["window"])

    def _rebuild_domain(self, verbose=False):
        """Rebuild the domain based on settings updated for importance sampling.

        These settings should be saved in self.importance_sampling_metadata["updates"],
        which is expected to be populated by reset_event()."""
        updates = self.importance_sampling_metadata["updates"].copy()

        # Assume that updates can contain T, f_s, roll_off, f_min, f_max, but no other
        # quantities that define a new domain (e.g., delta_f). Typical event metadata
        # will be constructed in this way.

        if "f_s" in updates or "T" in updates or "roll_off" in updates:
            window_settings = self.base_metadata["train_settings"]["data"][
                "window"
            ].copy()
            window_settings.update(
                (k, updates[k]) for k in set(window_settings).intersection(updates)
            )
            updates["window_factor"] = float(get_window_factor(window_settings))

        if "T" in updates:
            updates["delta_f"] = 1.0 / updates["T"]

        domain_dict = self.domain.domain_dict  # Existing settings
        domain_dict.update(
            (k, updates[k]) for k in set(domain_dict).intersection(updates)
        )

        if verbose:
            print("Rebuilding domain as follows:")
            print(
                yaml.dump(
                    domain_dict,
                    default_flow_style=False,
                    sort_keys=False,
                )
            )
        self.domain = build_domain(domain_dict)

    def _build_prior(self):
        """Build the prior based on model metadata. Called by __init__()."""
        intrinsic_prior = self.base_metadata["dataset_settings"]["intrinsic_prior"]
        extrinsic_prior = get_extrinsic_prior_dict(
            self.base_metadata["train_settings"]["data"]["extrinsic_prior"]
        )
        self.prior = build_prior_with_defaults({**intrinsic_prior, **extrinsic_prior})

        prior_update = self.importance_sampling_metadata.get("prior_update")
        if prior_update is not None:
            prior_update = PriorDict(prior_update.copy())
            self.prior.update(prior_update)

        # Split off prior over geocent_time if samples appear to be time-marginalized.
        # This needs to be saved to initialize the likelihood.
        if "geocent_time" in self.prior.keys() and "geocent_time" not in self.samples:
            self.geocent_time_prior = self.prior.pop("geocent_time")
        else:
            self.geocent_time_prior = None
        # Split off prior over phase if samples appear to be phase-marginalized.
        if "phase" in self.prior.keys() and "phase" not in self.samples:
            self.phase_prior = self.prior.pop("phase")
        else:
            self.phase_prior = None

    def update_prior(self, prior_update):
        """
        Update the prior based on a new dict of priors. Use the existing prior for
        parameters not included in the new dict.

        If class samples have not been importance sampled, then save new sample weights
        based on the new prior. If class samples have been importance sampled,
        then update the weights.

        Parameters
        ----------
        prior_update : dict
            Priors to update. This should be of the form {key : prior_str}, where str
            is a string that can instantiate a prior via PriorDict(prior_update). The
            prior_update is provided in this form so that it can be properly saved with
            the Result and later instantiated.
        """
        self.importance_sampling_metadata["prior_update"] = prior_update.copy()
        prior_update = PriorDict(prior_update)

        param_keys = [k for k, v in self.prior.items() if not isinstance(v, Constraint)]
        theta = self.samples[param_keys]

        if self.log_evidence is None:
            # Save old prior evaluations.
            log_prior_old = self.prior.ln_prob(theta, axis=0)

        # Update the prior itself, careful to split off geocent_time and phase priors
        # if necessary.
        if self.geocent_time_prior is not None and "geocent_time" in prior_update:
            self.geocent_time_prior = prior_update.pop("geocent_time")
        if self.phase_prior is not None and "phase" in prior_update:
            self.phase_prior = prior_update.pop("phase")
        self.prior.update(
            prior_update
        )  # TODO: Does this update cached constraint ratio?

        # Evaluate new prior.
        log_prior = self.prior.ln_prob(theta, axis=0)
        self.samples["log_prior"] = log_prior

        if self.log_evidence is None:
            # Save weights. Note that weights are 0 if outside the initial prior,
            # regardless of new prior. This makes sense since there is no way to assign
            # a sensible weight.
            log_weights = log_prior - log_prior_old
            weights = np.exp(log_weights - np.max(log_weights))
            weights /= np.mean(weights)
            self.samples["weights"] = weights

        else:
            # Recalculate the importance-sampling weights and log evidence.
            self._calculate_evidence()

    def _build_likelihood(
        self,
        time_marginalization_kwargs: Optional[dict] = None,
        phase_marginalization_kwargs: Optional[dict] = None,
        calibration_marginalization_kwargs: Optional[dict] = None,
        phase_grid: Optional[np.ndarray] = None,
    ):
        """
        Build the likelihood function based on model metadata. This is called at the
        beginning of importance_sample().

        Parameters
        ----------
        time_marginalization_kwargs: dict, optional
            kwargs for time marginalization. At this point the only kwarg is n_fft,
            which determines the number of FFTs used (higher n_fft means better
            accuracy, at the cost of longer computation time).
        phase_marginalization_kwargs: dict, optional
            kwargs for phase marginalization.
        calibration_marginalization_kwargs: dict
            Calibration marginalization parameters. If None, no calibration marginalization is used.
        """
        if time_marginalization_kwargs is not None:
            if self.geocent_time_prior is None:
                raise NotImplementedError(
                    "Time marginalization is not compatible with "
                    "non-marginalized network."
                )
            if type(self.geocent_time_prior) != Uniform:
                raise NotImplementedError(
                    "Only uniform time prior is supported for time marginalization."
                )
            time_marginalization_kwargs["t_lower"] = self.geocent_time_prior.minimum
            time_marginalization_kwargs["t_upper"] = self.geocent_time_prior.maximum

        if phase_marginalization_kwargs is not None:
            # check that phase prior is uniform [0, 2pi)
            if not (
                isinstance(self.phase_prior, Uniform)
                and (self.phase_prior._minimum, self.phase_prior._maximum)
                == (0, 2 * np.pi)
            ):
                raise ValueError(
                    f"Phase prior should be uniform [0, 2pi) for phase "
                    f"marginalization, but is {self.phase_prior}."
                )

        # This will save these settings when the Result instance is saved.
        self.time_marginalization_kwargs = time_marginalization_kwargs
        self.phase_marginalization_kwargs = phase_marginalization_kwargs
        self.calibration_marginalization_kwargs = calibration_marginalization_kwargs

        # Choose the WaveformGenerator domain. This is ultimately projected to the data domain, but generally we
        # allow it to be different, e.g., to start integrating from lower frequencies than the lower bound of the
        # likelihood integral. Usually we use the same domain as that used for the WaveformDataset. However,
        # if we make a change to certain settings during importance sampling, we need to ensure the projection is
        # still compatible:
        #
        # * delta_f (=1/T): cannot be changed in a domain projection, so update at the level of the
        #   WaveformGenerator.
        #
        # TODO: Add functionality to update other waveform settings, i.e., approximant, generation minimum and
        #  maximum frequencies, reference frequency, and starting frequency.

        wfg_domain_dict = self.base_metadata["dataset_settings"]["domain"].copy()
        if "updates" in self.importance_sampling_metadata:
            if "T" in self.importance_sampling_metadata["updates"]:
                delta_f_new = 1 / self.importance_sampling_metadata["updates"]["T"]
                print(
                    f'Updating waveform generation delta_f from {wfg_domain_dict["delta_f"]} to {delta_f_new}.'
                )
                wfg_domain_dict["delta_f"] = delta_f_new
        wfg_domain = build_domain(wfg_domain_dict)

        self.likelihood = StationaryGaussianGWLikelihood(
            wfg_kwargs=self.base_metadata["dataset_settings"]["waveform_generator"],
            wfg_domain=wfg_domain,
            data_domain=self.domain,
            event_data=self.context,
            t_ref=self.t_ref,
            time_marginalization_kwargs=time_marginalization_kwargs,
            phase_marginalization_kwargs=phase_marginalization_kwargs,
            calibration_marginalization_kwargs=calibration_marginalization_kwargs,
            phase_grid=phase_grid,
        )

    def sample_synthetic_phase(
        self,
        synthetic_phase_kwargs,
        inverse: bool = False,
    ):
        """
        Sample a synthetic phase for the waveform. This is a post-processing step
        applicable to samples theta in the full parameter space, except for the phase
        parameter (i.e., 14D samples). This step adds a phase parameter to the samples
        based on likelihood evaluations.

        A synthetic phase is sampled as follows.

            * Compute and cache the modes for the waveform mu(theta, phase=0) for
              phase 0, organize them such that each contribution m transforms as
              exp(-i * m * phase).
            * Compute the likelihood on a phase grid, by computing mu(theta, phase) from
              the cached modes. In principle this likelihood is exact, however, it can
              deviate slightly from the likelihood computed without cached modes for
              various technical reasons (e.g., slightly different windowing of cached
              modes compared to full waveform when transforming TD waveform to FD).
              These small deviations can be fully accounted for by importance sampling.
              *Note*: when approximation_22_mode=True, the entire waveform is assumed
              to transform as exp(2i*phase), in which case the likelihood is only exact
              if the waveform is fully dominated by the (2, 2) mode.
            * Build a synthetic conditional phase distribution based on this grid. We
              use an interpolated prior distribution bilby.core.prior.Interped,
              such that we can sample and also evaluate the log_prob. We add a constant
              background with weight self.synthetic_phase_kwargs to the kde to make
              sure that we keep a mass-covering property. With this, the importance
              sampling will yield exact results even when the synthetic phase conditional
              is just an approximation.

        Besides adding phase samples to self.samples['phase'], this method also modifies
        self.samples['log_prob'] by adding the log_prob of the synthetic phase
        conditional.

        This method modifies self.samples in place.

        Parameters
        ----------
        synthetic_phase_kwargs : dict
            This should consist of the kwargs
                approximation_22_mode (optional)
                num_processes (optional)
                n_grid
                uniform_weight (optional)
        inverse : bool, default False
            Whether to apply instead the inverse transformation. This is used prior to
            calculating the log_prob. In inverse mode, the posterior probability over
            phase is calculated for given samples. It is stored in self.samples[
            'log_prob'].
        """
        self.synthetic_phase_kwargs = synthetic_phase_kwargs

        approximation_22_mode = self.synthetic_phase_kwargs.get(
            "approximation_22_mode", True
        )

        if not (
            isinstance(self.phase_prior, Uniform)
            and (self.phase_prior._minimum, self.phase_prior._maximum) == (0, 2 * np.pi)
        ):
            raise ValueError(
                f"Phase prior should be uniform [0, 2pi) to work with synthetic phase."
                f" However, the prior is {self.phase_prior}."
            )

        # This builds on the Bilby approach to sampling the phase when using a
        # phase-marginalized likelihood.

        # Restrict to samples that are within the prior.
        param_keys = [k for k, v in self.prior.items() if not isinstance(v, Constraint)]
        theta = self.samples[param_keys]
        log_prior = self.prior.ln_prob(theta, axis=0)
        constraints = self.prior.evaluate_constraints(theta)
        np.putmask(log_prior, constraints == 0, -np.inf)
        within_prior = log_prior != -np.inf

        # Put a cap on the number of processes to avoid overhead:
        num_valid_samples = np.sum(within_prior)
        num_processes = min(
            self.synthetic_phase_kwargs.get("num_processes", 1), num_valid_samples // 10
        )

        print(f"Estimating synthetic phase for {num_valid_samples} samples.")
        t0 = time.time()

        if not inverse:
            # TODO: This can probably be removed.
            self._build_likelihood()

        if inverse:
            # We estimate the log_prob for given phases, so first save the evaluation
            # points.
            sample_phase = theta["phase"].to_numpy(copy=True)

        # For each sample, build the posterior over phase given the remaining parameters.

        phases = np.linspace(0, 2 * np.pi, self.synthetic_phase_kwargs["n_grid"])
        if approximation_22_mode:
            # For each sample, the un-normalized posterior depends only on (d | h(phase)):
            # The prior p(phase), and the inner products (h | h), and (d | d) only contribute
            # to the normalization. (We check above that p(phase) is constant.)
            theta["phase"] = 0.0
            d_inner_h_complex = self.likelihood.d_inner_h_complex_multi(
                theta.iloc[within_prior],
                num_processes,
            )

            # Evaluate the log posterior over the phase across the grid.
            phasor = np.exp(2j * phases)
            phase_log_posterior = np.outer(d_inner_h_complex, phasor).real
        else:
            self.likelihood.phase_grid = phases

            phase_log_posterior = apply_func_with_multiprocessing(
                self.likelihood.log_likelihood_phase_grid,
                theta.iloc[within_prior],
                num_processes=num_processes,
            )

        phase_posterior = np.exp(
            phase_log_posterior - np.amax(phase_log_posterior, axis=1, keepdims=True)
        )
        # Include a floor value to maintain mass coverage.
        phase_posterior += phase_posterior.mean(
            axis=-1, keepdims=True
        ) * self.synthetic_phase_kwargs.get("uniform_weight", 0.01)

        if not inverse:
            # Forward direction:
            #   (1) Sample a new phase according to the synthetic posterior.
            #   (2) Add the log_prob to the existing log_prob.

            new_phase, delta_log_prob = interpolated_sample_and_log_prob_multi(
                phases,
                phase_posterior,
                num_processes,
            )

            phase_array = np.full(len(theta), 0.0)
            phase_array[within_prior] = new_phase
            delta_log_prob_array = np.full(len(theta), -np.nan)
            delta_log_prob_array[within_prior] = delta_log_prob

            self.samples["phase"] = phase_array
            self.samples["log_prob"] += delta_log_prob_array

            # Insert the phase prior in the prior, since now the phase is present.
            self.prior["phase"] = self.phase_prior
            self.phase_prior = None

            # reset likelihood for safety
            # TODO: Can this be removed?
            self.likelihood = None

        else:
            # TODO: Possibly remove.
            # Inverse direction:
            #   (1) Evaluate the synthetic log prob for given phase points, and save it.

            log_prob = interpolated_log_prob_multi(
                phases,
                phase_posterior,
                sample_phase[within_prior],
                num_processes,
            )

            # Outside of prior, set log_prob to -np.nan.
            log_prob_array = np.full(len(theta), -np.nan)
            log_prob_array[within_prior] = log_prob
            self.samples["log_prob"] = log_prob_array
            del self.samples["phase"]

        print(f"Done. This took {time.time() - t0:.2f} s.")

    def get_samples_bilby_phase(self):
        """
        Convert the spin angles phi_jl and theta_jn to account for a difference in
        phase definition compared to Bilby.

        Returns
        -------
        pd.DataFrame
            Samples
        """
        spin_conversion_phase_old = self.base_metadata["dataset_settings"][
            "waveform_generator"
        ].get("spin_conversion_phase")

        # Redefine phase parameter to be consistent with Bilby.
        return change_spin_conversion_phase(
            self.samples, self.f_ref, spin_conversion_phase_old, None
        )

    @property
    def pesummary_samples(self):
        """Samples in a form suitable for PESummary.

        These samples are adjusted to undo certain conventions used internally by
        Dingo:
            * Times are corrected by the reference time t_ref.
            * Samples are unweighted, using a fixed random seed for sampling importance
            resampling.
            * The spin angles phi_jl and theta_jn are transformed to account for a
            difference in phase definition.
            * Some columns are dropped: delta_log_prob_target, log_prob
        """
        # Unweighted samples.
        samples = self.sampling_importance_resampling(random_state=RANDOM_STATE)

        # Remove unwanted columns.
        samples.drop(
            ["delta_log_prob_target", "log_prob"], axis=1, errors="ignore", inplace=True
        )
        for col in samples.columns:
            if col.endswith("_proxy"):
                samples.drop(col, axis=1, inplace=True)

        # Shift times. This requires double precision. There *should* be no non-numeric
        # values in the samples dataframe since resampling will have excluded
        # zero-weight samples (which could have nan likelihood).
        samples = samples.astype(float)
        for col in samples.columns:
            if "time" in col:
                samples.loc[:, col] += self.t_ref

        spin_conversion_phase_old = self.base_metadata["dataset_settings"][
            "waveform_generator"
        ].get("spin_conversion_phase")

        # Redefine phase parameter to be consistent with Bilby. COMMENTED BECAUSE SLOW
        samples = change_spin_conversion_phase(
            samples, self.f_ref, spin_conversion_phase_old, None
        )

        return samples

    @property
    def pesummary_prior(self):
        """The prior in a form suitable for PESummary.

        By convention, Dingo stores all times *relative* to a reference time, typically
        the trigger time for an event. The prior returned here corrects for that offset to
        be consistent with other codes.
        """
        prior = copy.deepcopy(self.prior)
        for p in prior:
            if "time" in p:
                try:
                    prior[p].maximum += self.t_ref
                    prior[p].minimum += self.t_ref
                except AttributeError:
                    continue
        return prior
