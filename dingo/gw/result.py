import copy
import time
from typing import Optional

import numpy as np
from bilby.core.prior import Uniform, Constraint, PriorDict, DeltaFunction
from bilby.gw.prior import CalibrationPriorDict
from bilby_pipe.utils import CALIBRATION_CORRECTION_TYPE_LOOKUP

from dingo.core.result import Result as CoreResult
from dingo.core.utils.backward_compatibility import check_minimum_version


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
            The domain of the data (e.g., UniformFrequencyDomain), needed for calculating
            likelihoods.
        prior : PriorDict
            The prior distribution, used for importance sampling.
        likelihood : Likelihood
            The Likelihood object, needed for importance sampling.
        context : dict
            Context data from which the samples were produced (e.g., strain data, ASDs).
        metadata : dict
            Metadata describing the neural networks and sampling settings used,
            including structured sampler provenance under `settings["sampler"]`.
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
    def calibration_sampling_kwargs(self):
        return self.importance_sampling_metadata.get("calibration_sampling")

    @calibration_sampling_kwargs.setter
    def calibration_sampling_kwargs(self, value):
        self.importance_sampling_metadata["calibration_sampling"] = value

    @property
    def use_base_domain(self) -> bool:
        return self.importance_sampling_metadata.get("use_base_domain", False)

    @use_base_domain.setter
    def use_base_domain(self, value: bool):
        if hasattr(self.domain, "base_domain"):
            self.importance_sampling_metadata["use_base_domain"] = value
            # The representation is context state: re-derive rather than mutate, so
            # the fresh context starts with an empty likelihood cache and a
            # previously built likelihood cannot leak across the change.
            if (
                self.sampler_context is not None
                and self.sampler_context.use_base_domain != value
            ):
                self.sampler_context = self.sampler_context.derive(
                    use_base_domain=value
                )

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

    @property
    def minimum_frequency(self) -> dict[str, float] | float:
        return self.event_metadata.get("minimum_frequency", self.domain.f_min)

    @minimum_frequency.setter
    def minimum_frequency(self, value: dict[str, float] | float):
        self.event_metadata["minimum_frequency"] = value

    @property
    def maximum_frequency(self) -> dict[str, float] | float:
        return self.event_metadata.get("maximum_frequency", self.domain.f_max)

    @maximum_frequency.setter
    def maximum_frequency(self, value: dict[str, float] | float):
        self.event_metadata["maximum_frequency"] = value

    def _build_domain(self):
        """Take the data domain from the sampler context -- its single owner. A
        context derived with importance-sampling updates already carries the
        rebuilt domain. Called by __init__() and after reset_event()."""
        check_minimum_version(self.version, raise_exception=False)
        if self.sampler_context is None:
            self.domain = None
            return
        self.domain = self.sampler_context.domain

    def _build_context(self):
        """Reconstruct the per-event sampler context from the serialized payload
        (settings + event data + event metadata), so that prior (and, later,
        likelihood) construction delegates to `GWSamplerContext` no matter how the
        Result was born -- live from a sampler or loaded from file."""
        # Only the metadata is required: the prior and domain views are defined
        # without event data (a result whose strain payload was stripped or left
        # on disk still has a working prior); the likelihood view checks for the
        # event data itself.
        if self.settings is None:
            return None
        from dingo.gw.inference.context import GWSamplerContext

        try:
            # base_metadata resolves the unconditional ("base") indirection, so
            # density-recovery results reconstruct from the analysis metadata.
            context = GWSamplerContext.from_model_metadata(
                self.base_metadata, self.context, self.event_metadata
            )
        except (KeyError, TypeError) as e:
            # Settings are not a full model metadata (e.g. minimal test payloads);
            # the result is then transport-only. Make the degradation visible
            # rather than silent.
            print(
                f"Could not reconstruct a sampler context from the result settings "
                f"({type(e).__name__}: {e}); prior, domain, and likelihood are "
                f"unavailable."
            )
            return None
        # Importance-sampling settings updates change the data representation, which
        # lives on a derived context (same event, different representation).
        metadata = self.importance_sampling_metadata or {}
        updates = metadata.get("updates")
        use_base_domain = metadata.get("use_base_domain", False)
        if updates or use_base_domain:
            context = context.derive(updates=updates, use_base_domain=use_base_domain)
        return context

    def _build_prior(self):
        """Take the static prior from the sampler context (its single owner), then
        apply the evolving analysis state: any importance-sampling prior update,
        and the split-off of time / phase priors for marginalized networks. Called
        by __init__(). Without a reconstructable context (a payload without full
        model metadata) the result is transport-only and the prior is `None`."""
        if self.sampler_context is None:
            self.prior = None
            self.geocent_time_prior = None
            self.phase_prior = None
            return
        # Deepcopy because the marginalization split-off below mutates it.
        self.prior = copy.deepcopy(self.sampler_context.prior)

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
        # Marginalization is validated here against the *evolved* prior (any
        # importance-sampling prior update, the time / phase split-offs), which
        # the sample-free context cannot see; the explicit bounds set below
        # therefore take precedence over the context's static-prior fill.
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

        if self.sampler_context is None:
            raise ValueError(
                "Building the likelihood requires a sampler context; this result "
                "does not carry full model metadata."
            )

        # Construction is owned by the sampler context; its (possibly derived)
        # representation already encodes the importance-sampling settings updates.
        # Validated marginalization bounds enter as arguments.
        #
        # TODO: Add functionality to update other waveform settings, i.e.,
        #  approximant, generation minimum and maximum frequencies, reference
        #  frequency, and starting frequency.
        self.likelihood = self.sampler_context.likelihood(
            time_marginalization_kwargs=time_marginalization_kwargs,
            phase_marginalization_kwargs=phase_marginalization_kwargs,
            calibration_marginalization_kwargs=calibration_marginalization_kwargs,
        )

    def sample_calibration_parameters(self, calibration_sampling_kwargs: dict):
        """
        Sample calibration parameters from the calibration prior and add them to the
        samples DataFrame. Also updates self.prior with the calibration priors and
        adjusts self.samples["log_prob"] accordingly.

        This should be called before importance_sample() when importance sampling
        over calibration uncertainty. The calibration prior log_prob is added to
        self.samples["log_prob"] so that it is properly accounted for in the
        importance sampling weights.

        After calling this method, each sample will have calibration parameters
        (e.g., recalib_H1_amplitude_0, recalib_H1_phase_0, etc.) that will be used
        by the likelihood to apply calibration corrections.

        Parameters
        ----------
        calibration_sampling_kwargs : dict
            Calibration sampling parameters. Keys:

            calibration_envelope : dict
                Dictionary of the form {"H1": filepath, "L1": filepath, ...} with
                locations of calibration envelope files (.txt).
            num_calibration_nodes : int
                Number of log-spaced frequency nodes for the calibration spline model.
            correction_type : str or dict or None, default "data"
                Whether envelopes are over eta ("data") or alpha ("template").
                Can be a string (applied to all detectors), a dict mapping ifo names
                to correction types, or None (uses defaults from CALIBRATION_CORRECTION_TYPE_LOOKUP).
        """
        self.calibration_sampling_kwargs = calibration_sampling_kwargs

        # Handle correction_type defaults
        correction_type = self.calibration_sampling_kwargs.get(
            "correction_type", "data"
        )
        if correction_type is None:
            correction_type_dict = {
                ifo: CALIBRATION_CORRECTION_TYPE_LOOKUP[ifo]
                for ifo in self.interferometers
            }
        elif correction_type == "data" or correction_type == "template":
            correction_type_dict = {
                ifo: correction_type for ifo in self.interferometers
            }
        elif isinstance(correction_type, dict):
            correction_type_dict = correction_type
        else:
            raise ValueError(f"{correction_type} not understood")

        # Build calibration priors for sampling
        calibration_priors = {}
        for ifo in self.interferometers:
            calibration_priors[ifo] = CalibrationPriorDict.from_envelope_file(
                self.calibration_sampling_kwargs["calibration_envelope"][ifo],
                self.domain.f_min,
                self.domain.f_max,
                self.calibration_sampling_kwargs["num_calibration_nodes"],
                ifo,
                correction_type=correction_type_dict[ifo],
            )

        # Removing the delta function priors on the frequency nodes, amplitude and phase.
        # Usually the frequency nodes are set to delta functions, but we also remove the
        # amplitude and phase delta functions if present.
        # This avoids large log probs and log priors, since the density of a delta function
        # at the sampled point is infinite. The delta functions do not affect the sampling,
        # since they just fix certain parameters to constant values.
        for ifo in calibration_priors:
            for param_name, prior_obj in list(calibration_priors[ifo].items()):
                if isinstance(prior_obj, DeltaFunction):
                    calibration_priors[ifo].pop(param_name)

        # Sample calibration parameters and calculate log_prob
        num_samples = len(self.samples)
        print(f"Sampling calibration parameters for {num_samples} samples.")

        delta_log_prob = np.zeros(num_samples)

        # Here we will sample the calibration parameters from the prior.
        # We treat the *prior as the proposal* distribution and
        # therefore add the log_prob of the sampled calibration parameters
        # to the existing log_prob. We also will update the prior
        # to include the calibration priors using the importance_sampling_metadata
        prior_update = self.importance_sampling_metadata.get("prior_update", {})
        for ifo, prior in calibration_priors.items():
            draws = prior.sample(num_samples)

            # Calculate log_prob of the calibration draws
            delta_log_prob += prior.ln_prob(draws, axis=0)

            # Add draws to samples
            for param_name, values in draws.items():
                self.samples[param_name] = np.array(values)

            # Update prior_update dict with calibration parameters. This is for
            # persistence when saving to hdf5
            for param_name, prior_obj in prior.items():
                # bilby's Prior.__repr__ isn't parseable for numpy scalars on numpy>2.0
                # Upstream fix: https://github.com/bilby-dev/bilby/pull/1108
                # Can be removed once dingo requires a bilby release that includes it.
                for attr, value in prior_obj.get_instantiation_dict().items():
                    if isinstance(value, np.generic):
                        setattr(prior_obj, attr, value.item())
                prior_update[param_name] = repr(prior_obj)

        # Store prior_update for persistence on save/reload
        self.importance_sampling_metadata["prior_update"] = prior_update

        # Update log_prob (add the log probability of sampled calibration parameters)
        self.samples["log_prob"] += delta_log_prob

        # Rebuild the prior (which will include calibration priors from prior_update)
        self._build_prior()

    def _sample_synthetic_phase_chain(
        self, theta, within_prior, approximation_22_mode, num_processes
    ):
        """Synthetic phase as a chain over the sampler context: the root emits the
        within-prior proposal samples with their stored log-prob, and
        `SyntheticPhaseFactor` draws `phase` -- the composer's ordinary log-prob fold
        returns the joint proposal density `log q(theta) + log q(phase | theta, d)`,
        which becomes the samples' log_prob. The factor builds the phase-full
        likelihood from the sampler context, whose (possibly derived) representation
        encodes the importance-sampling view (base domain, rebuilt domain, frequency
        updates). Out-of-prior rows get `phase = 0` and `log_prob = nan` (they
        receive zero weight in importance sampling regardless)."""
        from dingo.core.factors import ChainComposer, SampleTableFactor
        from dingo.gw.inference.steps import SyntheticPhaseFactor

        theta_within = theta.iloc[np.flatnonzero(within_prior)]
        table = SampleTableFactor(
            {k: theta_within[k].to_numpy() for k in theta_within.columns},
            log_prob=self.samples["log_prob"].to_numpy()[within_prior],
        )
        factor = SyntheticPhaseFactor(
            conditioning=list(theta_within.columns),
            n_grid=self.synthetic_phase_kwargs["n_grid"],
            approximation_22_mode=approximation_22_mode,
            uniform_weight=self.synthetic_phase_kwargs.get("uniform_weight", 0.01),
            num_processes=num_processes,
        )
        chain = ChainComposer([table, factor])
        out, log_prob = chain.sample_and_log_prob(
            len(theta_within), self.sampler_context
        )

        phase_array = np.full(len(theta), 0.0)
        phase_array[within_prior] = out["phase"].numpy()
        log_prob_array = np.full(len(theta), np.nan)
        log_prob_array[within_prior] = log_prob.numpy()
        self.samples["phase"] = phase_array
        self.samples["log_prob"] = log_prob_array

        # Insert the phase prior in the prior, since now the phase is present.
        self.prior["phase"] = self.phase_prior
        self.phase_prior = None
        # Any previously built likelihood does not describe the now-phase-full
        # samples; importance sampling rebuilds with its own marginalization
        # settings.
        self.likelihood = None

    def sample_synthetic_phase(self, synthetic_phase_kwargs):
        """
        Sample a synthetic phase for the waveform. This is a post-processing step
        applicable to samples theta in the full parameter space, except for the phase
        parameter (i.e., 14D samples). It adds a `phase` column to the samples and
        replaces `log_prob` with the joint proposal density
        `log q(theta) + log q(phase | theta, d)`.

        The phase distribution `q(phase | theta, d)` is constructed per sample by
        `SyntheticPhaseFactor` from the likelihood on a phase grid (with a uniform
        floor for mass coverage, so importance sampling remains exact even where the
        conditional is approximate); the step runs as a chain rooted in the
        within-prior proposal samples. Out-of-prior samples receive `phase = 0` and
        `log_prob = nan`, and carry zero weight in importance sampling.

        This method modifies self.samples in place. Afterwards the phase prior
        rejoins `self.prior`.

        Parameters
        ----------
        synthetic_phase_kwargs : dict
            Keys: `n_grid` (required), `approximation_22_mode` (optional; default
            True assumes a (2, 2)-dominated waveform, otherwise the exact mode sum
            is used, which requires the waveform generator's
            `spin_conversion_phase = 0`), `uniform_weight` (optional),
            `num_processes` (optional).
        """
        if self.sampler_context is None:
            raise ValueError(
                "Synthetic phase requires a sampler context; this result does not "
                "carry full model metadata."
            )

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

        # Restrict to samples that are within the prior.
        param_keys = [k for k, v in self.prior.items() if not isinstance(v, Constraint)]
        theta = self.samples[param_keys]

        # Compute log_prior only for non-DeltaFunction parameters.  DeltaFunction
        # priors return ln_prob = +inf at the peak, which causes check_ln_prob to
        # skip constraint evaluation and return +inf for every sample, so
        # np.isfinite(log_prior) would be False for all samples.  Additionally, RA
        # corrections (trigger_time vs model ref_time) can shift fixed parameters by
        # tiny amounts, making DeltaFunction ln_prob = -inf for all samples.
        prior_keys_for_lp = [
            k for k, v in self.prior.items()
            if not isinstance(v, Constraint) and not isinstance(v, DeltaFunction)
        ]
        log_prior = self.prior.ln_prob(self.samples[prior_keys_for_lp], axis=0)
        # Pass a plain dict so bilby's evaluate_constraints handles the argument
        # correctly.  bilby's evaluate_constraints mishandles a DataFrame argument:
        # its internal .values() call raises TypeError (DataFrame.values is a
        # property, not a method), causing the try/except inside bilby to fall
        # through to ``np.ones_like(out_sample)``, which returns a 2-D array and
        # causes a shape-broadcast error in the subsequent element-wise multiplication.
        constraints = self.prior.evaluate_constraints(dict(theta))
        np.putmask(log_prior, constraints == 0, -np.inf)
        within_prior = np.isfinite(log_prior)

        # Put a cap on the number of processes to avoid overhead:
        num_valid_samples = np.sum(within_prior)
        num_processes = min(
            self.synthetic_phase_kwargs.get("num_processes", 1), num_valid_samples // 10
        )

        print(f"Estimating synthetic phase for {num_valid_samples} samples.")
        t0 = time.time()

        self._sample_synthetic_phase_chain(
            theta, within_prior, approximation_22_mode, num_processes
        )
        print(f"Done. This took {time.time() - t0:.2f} s.")

    def get_samples_bilby_phase(self, num_processes=1):
        """
        Convert the spin angles phi_jl and theta_jn to account for a difference in
        phase definition compared to Bilby.

        Parameters
        ----------
        num_processes: int
            Number of parallel processes.

        Returns
        -------
        pd.DataFrame
            Samples
        """
        from dingo.gw.inference.steps import SpinConventionReparam

        return SpinConventionReparam(num_processes=num_processes).to_physical(
            self.samples, self.base_metadata
        )

    def get_pesummary_samples(
        self, num_processes=1, resampling_method="clip+rejection"
    ):
        """Samples in a form suitable for PESummary.

        These samples are adjusted to undo certain conventions used internally by
        Dingo:
            * Times are corrected by the reference time t_ref.
            * Samples are unweighted, using a fixed random seed for sampling importance
            resampling.
            * The spin angles phi_jl and theta_jn are transformed to account for a
            difference in phase definition.
            * Some columns are dropped: delta_log_prob_target, log_prob

        Parameters
        ----------
        num_processes : int
            Number of processes for spin conversion.
        resampling_method : str
            Method for producing unweighted samples from weighted ones.
            'clip+rejection': clip extreme weights then rejection sample (default).
            'sir': sampling importance resampling (old behavior).
        """
        if hasattr(self, "_pesummary_samples"):
            return self._pesummary_samples

        # Unweighted samples.
        if "weights" in self.samples:
            if resampling_method == "clip+rejection":
                samples = self.rejection_sample(
                    clip_weights=True,
                    random_state=RANDOM_STATE,
                )
            elif resampling_method == "sir":
                samples = self.sampling_importance_resampling(random_state=RANDOM_STATE)
            else:
                raise ValueError(
                    f"Unknown resampling_method '{resampling_method}'. "
                    "Use 'clip+rejection' or 'sir'."
                )
        else:
            samples = self.samples.copy()

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

        # Redefine the spin angles to the physical (Bilby) convention.
        from dingo.gw.inference.steps import SpinConventionReparam

        samples = SpinConventionReparam(num_processes=num_processes).to_physical(
            samples, self.base_metadata
        )

        self._pesummary_samples = samples

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
