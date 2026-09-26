import copy
import time
from typing import Optional

import numpy as np
from bilby.core.prior import Uniform, Constraint, PriorDict
from bilby.gw.prior import CalibrationPriorDict
from bilby_pipe.utils import CALIBRATION_CORRECTION_TYPE_LOOKUP

from dingo.core.result import Result as CoreResult
from dingo.core.utils.backward_compatibility import (
    check_minimum_version,
    update_data_config,
    update_model_config,
)
from dingo.core.inference.composer import ChainComposer
from dingo.core.inference.steps import PriorFactor, SampleTableFactor
from bilby.gw.detector import InterferometerList
from dingo.gw.frequency_updates import resolve_frequency_bounds


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
        """The analyzed detectors (arm names for a triangular detector), from the
        sampler context; the event data's detectors for a transport-only result."""
        if self.sampler_context is not None:
            return [
                ifo.name for ifo in InterferometerList(self.sampler_context.detectors)
            ]
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
        return (self.event_metadata or {}).get("minimum_frequency", self.domain.f_min)

    @minimum_frequency.setter
    def minimum_frequency(self, value: dict[str, float] | float):
        self.event_metadata["minimum_frequency"] = value

    @property
    def maximum_frequency(self) -> dict[str, float] | float:
        return (self.event_metadata or {}).get("maximum_frequency", self.domain.f_max)

    @maximum_frequency.setter
    def maximum_frequency(self, value: dict[str, float] | float):
        self.event_metadata["maximum_frequency"] = value

    def _build_domain(self):
        """Take the network's data domain from the sampler context -- its single
        owner. Called by __init__() and after reset_event()."""
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

        # Settings written by older code are mapped to the current schema in place
        # (idempotent), as the model loaders do for checkpoints.
        metadata = self.base_metadata
        update_data_config(metadata)
        if "model" in metadata["train_settings"]:
            update_model_config(metadata["train_settings"]["model"])

        # base_metadata resolves the unconditional ("base") indirection, so
        # density-recovery results reconstruct from the analysis metadata. The
        # event data and metadata determine the likelihood's grid and frequency
        # range; after reset_event they are those of the importance-sampling event.
        return GWSamplerContext.from_model_metadata(
            self.base_metadata, self.context, self.event_metadata
        )

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
        # Merge with the recorded updates (e.g. calibration priors added by
        # sample_proposal_extensions), so that a reload rebuilds all of them.
        self.importance_sampling_metadata["prior_update"] = {
            **self.importance_sampling_metadata.get("prior_update", {}),
            **prior_update,
        }
        # PriorDict instantiates in place, so work on a copy: the caller's dict
        # keeps its string form.
        prior_update = PriorDict(prior_update.copy())

        if self.log_evidence is None:
            # Save old prior evaluations.
            log_prior_old = self._log_prior()

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
        log_prior = self._log_prior()
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
        # the sample-free context cannot see; the bounds the context requires are
        # set below from that prior.
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

        # The sampler context builds the likelihood on the event data it holds.
        # Validated marginalization bounds enter as arguments.
        #
        # TODO: Add functionality to update other waveform settings, i.e.,
        #  approximant, generation minimum and maximum frequencies, reference
        #  frequency, and starting frequency.
        self.likelihood = self.sampler_context.likelihood(
            time_marginalization_kwargs=time_marginalization_kwargs,
            phase_marginalization_kwargs=phase_marginalization_kwargs,
            calibration_marginalization_kwargs=calibration_marginalization_kwargs,
            use_base_domain=self.use_base_domain,
        )

        # The likelihood places its own calibration nodes, log-spaced across each
        # detector's frequency range. Check that this range matches the end nodes
        # the calibration parameters were drawn at.
        if self.calibration_sampling_kwargs is not None:
            update = self.likelihood.frequency_update or {}
            bounds = resolve_frequency_bounds(
                self.interferometers,
                self.likelihood.data_domain,
                minimum_frequency=update.get("minimum_frequency"),
                maximum_frequency=update.get("maximum_frequency"),
            )
            n = self.calibration_sampling_kwargs["num_calibration_nodes"]
            for ifo, (f_min, f_max) in bounds.items():
                if f"recalib_{ifo}_frequency_0" not in self.prior:
                    # Results from earlier Dingo versions did not store the nodes.
                    print(
                        f"No calibration node frequencies stored for {ifo}; not "
                        f"checking them against the likelihood."
                    )
                    continue
                f_first = self.prior[f"recalib_{ifo}_frequency_0"].peak
                f_last = self.prior[f"recalib_{ifo}_frequency_{n - 1}"].peak
                if not np.allclose([f_first, f_last], [f_min, f_max]):
                    raise ValueError(
                        f"The likelihood's calibration nodes for {ifo} span "
                        f"[{f_min}, {f_max}] Hz, but the calibration parameters "
                        f"were drawn at nodes spanning [{f_first}, {f_last}] Hz."
                    )

    def sample_proposal_extensions(
        self,
        calibration_sampling_kwargs: Optional[dict] = None,
        synthetic_phase_kwargs: Optional[dict] = None,
    ):
        """
        Extend the proposal samples with calibration parameters and / or a synthetic
        phase, in preparation for importance sampling.

        Both run as one chain rooted in the proposal samples,

            [SampleTableFactor, PriorFactor (per detector), SyntheticPhaseFactor],

        with each optional step present only if its settings are given (see
        `_calibration_steps` and `_synthetic_phase_step`). The chain folds every
        step's log probability into `log_prob`, the joint proposal density
        `log q(theta) + log q(calibration) + log q(phase | theta, calibration, d)`.
        The calibration parameters are drawn first, so the phase distribution is
        built from the likelihood including the drawn calibration curve.

        The chain runs on the within-prior samples only. Out-of-prior samples
        receive placeholder values 0 for the new parameters and `log_prob = nan`,
        and carry zero weight in importance sampling.
        Afterwards the prior includes the new parameters.

        This method modifies self.samples in place.

        Note: as discussed in the sampler revamp (dingo-gw/dingo#389), this
        importance-sampling preparation, like the evolving prior, is to move out of
        the Result object.

        Parameters
        ----------
        calibration_sampling_kwargs : dict, optional
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
        synthetic_phase_kwargs : dict, optional
            Synthetic phase parameters. Keys:

            n_grid : int
                Number of phase grid points on [0, 2pi).
            approximation_22_mode : bool, default True
                Assume a (2, 2)-dominated waveform. Otherwise the exact mode sum is
                used, which requires the waveform generator's
                spin_conversion_phase = 0.
            uniform_weight : float, default 0.01
                Weight of the uniform floor added to the phase distribution for
                mass coverage.
            num_processes : int, default 1
                Number of parallel processes.
            use_dft_phase_decomposition : bool, optional
                Overrides the waveform generator setting of the same name for this
                step only, selecting how the m-components are obtained (see
                WaveformGenerator).
            cache_log_likelihood : bool, default False
                Also store the log likelihood at the drawn phase in
                samples["log_likelihood"], for importance_sample(
                use_cached_log_likelihood=True). Exact mode only. It includes the
                calibration drawn in the same chain.
        """
        if calibration_sampling_kwargs is None and synthetic_phase_kwargs is None:
            raise ValueError(
                "Pass calibration_sampling_kwargs and / or synthetic_phase_kwargs."
            )
        if "log_likelihood" in self.samples:
            # A stored log likelihood (cached earlier, or from a previous importance
            # sampling run) does not describe the redrawn phases or calibration.
            self.samples = self.samples.drop(columns="log_likelihood")

        param_keys = [k for k, v in self.prior.items() if not isinstance(v, Constraint)]
        theta = self.samples[param_keys]
        # Out-of-prior samples carry zero weight in importance sampling (which uses
        # the same _log_prior), and it may not even be possible to generate signals
        # for them (e.g., for BH spins > 1).
        within_prior = np.isfinite(self._log_prior())

        steps = []
        if calibration_sampling_kwargs is not None:
            steps += self._calibration_steps(calibration_sampling_kwargs)
        if synthetic_phase_kwargs is not None:
            steps.append(
                self._synthetic_phase_step(
                    synthetic_phase_kwargs,
                    conditioning=param_keys + [p for s in steps for p in s.parameters],
                    num_samples=np.sum(within_prior),
                )
            )

        table = SampleTableFactor(
            {k: theta[k].to_numpy()[within_prior] for k in param_keys},
            log_prob=self.samples["log_prob"].to_numpy()[within_prior],
        )
        print(
            f"Sampling {', '.join(type(s).__name__ for s in steps)} for "
            f"{np.sum(within_prior)} samples."
        )
        t0 = time.time()
        # One draw per proposal sample (the table root is emitted once).
        out, log_prob = ChainComposer([table] + steps).sample_and_log_prob(
            1, self.sampler_context
        )

        # Out-of-prior samples get placeholder values 0 (finite, so that their prior
        # is -inf rather than nan), log_prob = nan and no cached log likelihood.
        for k in [c for s in steps for c in s.produces]:
            column = np.full(len(theta), np.nan if k == "log_likelihood" else 0.0)
            column[within_prior] = out[k].cpu().numpy()
            self.samples[k] = column
        log_prob_array = np.full(len(theta), np.nan)
        log_prob_array[within_prior] = log_prob.cpu().numpy()
        self.samples["log_prob"] = log_prob_array

        # Record the settings and the calibration priors only now that the chain has
        # run, so that a failure leaves the result unchanged.
        prior_update = self.importance_sampling_metadata.get("prior_update", {})
        for s in steps:
            if isinstance(s, PriorFactor):
                # Recorded in string form, for persistence when saving to hdf5.
                prior_update.update({k: repr(s.prior[k]) for k in s.parameters})
        self.importance_sampling_metadata["prior_update"] = prior_update
        if calibration_sampling_kwargs is not None:
            self.calibration_sampling_kwargs = calibration_sampling_kwargs
        if synthetic_phase_kwargs is not None:
            self.synthetic_phase_kwargs = synthetic_phase_kwargs

        # Rebuild the prior: it now includes the calibration priors (from
        # prior_update), and the phase prior rejoins it once phase is in the samples.
        self._build_prior()
        if synthetic_phase_kwargs is not None:
            # Any previously built likelihood does not describe the now-phase-full
            # samples; importance sampling rebuilds with its own marginalization
            # settings.
            self.likelihood = None
        print(f"Done. This took {time.time() - t0:.2f} s.")

    def _calibration_steps(self, calibration_sampling_kwargs: dict) -> list:
        """
        Set up the calibration steps of `sample_proposal_extensions`: one
        `PriorFactor` per detector, drawing the calibration parameters (e.g.
        `recalib_H1_amplitude_0`) from the calibration prior, which acts as their
        proposal. Since the calibration parameters are new to the target as well,
        `sample_proposal_extensions` also adds these priors to `self.prior`.

        Parameters
        ----------
        calibration_sampling_kwargs : dict
            See `sample_proposal_extensions`.

        Returns
        -------
        list[PriorFactor]
        """
        # Handle correction_type defaults
        correction_type = calibration_sampling_kwargs.get("correction_type", "data")
        if correction_type is None:
            correction_type_dict = {
                ifo: CALIBRATION_CORRECTION_TYPE_LOOKUP[ifo]
                for ifo in self.interferometers
            }
        elif correction_type == "data" or correction_type == "template":
            correction_type_dict = {ifo: correction_type for ifo in self.interferometers}
        elif isinstance(correction_type, dict):
            correction_type_dict = correction_type
        else:
            raise ValueError(f"{correction_type} not understood")

        # Build the calibration priors. As in Bilby, the spline nodes are placed
        # across each detector's frequency range, the same range the likelihood
        # masks the ASDs to. Without a range in the event metadata the domain
        # bounds are used.
        frequency_bounds = resolve_frequency_bounds(
            self.interferometers,
            self.domain,
            minimum_frequency=self.minimum_frequency,
            maximum_frequency=self.maximum_frequency,
        )
        steps = []
        for ifo in self.interferometers:
            f_min, f_max = frequency_bounds[ifo]
            calibration_prior = CalibrationPriorDict.from_envelope_file(
                calibration_sampling_kwargs["calibration_envelope"][ifo],
                f_min,
                f_max,
                calibration_sampling_kwargs["num_calibration_nodes"],
                ifo,
                correction_type=correction_type_dict[ifo],
            )
            # bilby's Prior.__repr__ isn't parseable for numpy scalars on
            # numpy>2.0. Upstream fix: https://github.com/bilby-dev/bilby/pull/1108
            # Can be removed once dingo requires a bilby release that includes it.
            for prior_obj in calibration_prior.values():
                for attr, value in prior_obj.get_instantiation_dict().items():
                    if isinstance(value, np.generic):
                        setattr(prior_obj, attr, value.item())
            steps.append(PriorFactor(calibration_prior))
        return steps

    def _synthetic_phase_step(
        self, synthetic_phase_kwargs: dict, conditioning: list[str], num_samples: int
    ) -> "SyntheticPhaseFactor":
        """
        Set up the synthetic phase step of `sample_proposal_extensions`: a
        `SyntheticPhaseFactor`, which constructs `q(phase | theta, d)` per sample
        from the likelihood on a phase grid (with a uniform floor for mass coverage,
        so importance sampling remains exact even where the conditional is
        approximate). It applies to samples in the full parameter space except the
        phase.

        Parameters
        ----------
        synthetic_phase_kwargs : dict
            See `sample_proposal_extensions`.
        conditioning : list[str]
            The columns the phase distribution conditions on: the proposal
            parameters and any calibration parameters drawn earlier in the chain.
        num_samples : int
            Number of samples the chain runs on, used to cap the number of processes.

        Returns
        -------
        SyntheticPhaseFactor
        """
        from dingo.gw.inference.steps import SyntheticPhaseFactor

        if self.sampler_context is None:
            raise ValueError(
                "Synthetic phase requires a sampler context; this result does not "
                "carry full model metadata."
            )
        if not (
            isinstance(self.phase_prior, Uniform)
            and (self.phase_prior._minimum, self.phase_prior._maximum) == (0, 2 * np.pi)
        ):
            raise ValueError(
                f"Phase prior should be uniform [0, 2pi) to work with synthetic phase."
                f" However, the prior is {self.phase_prior}."
            )

        wfg_updates = None
        if "use_dft_phase_decomposition" in synthetic_phase_kwargs:
            wfg_updates = {
                "use_dft_phase_decomposition": synthetic_phase_kwargs[
                    "use_dft_phase_decomposition"
                ]
            }
        step = SyntheticPhaseFactor(
            conditioning=conditioning,
            n_grid=synthetic_phase_kwargs["n_grid"],
            approximation_22_mode=synthetic_phase_kwargs.get(
                "approximation_22_mode", True
            ),
            uniform_weight=synthetic_phase_kwargs.get("uniform_weight", 0.01),
            # Put a cap on the number of processes to avoid overhead.
            num_processes=min(
                synthetic_phase_kwargs.get("num_processes", 1),
                num_samples // 10,
            ),
            use_base_domain=self.use_base_domain,
            wfg_updates=wfg_updates,
            cache_log_likelihood=synthetic_phase_kwargs.get(
                "cache_log_likelihood", False
            ),
        )
        return step

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
