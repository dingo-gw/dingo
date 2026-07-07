"""
Gravitational-wave factors, steps, and samplers for the factorized sampler.

The domain-agnostic pieces (`Factor`, `FlowFactor`, `Reparametrization`,
`TargetCorrection`, and the composers) live in `dingo.core.factors`. This module adds
the GW-specific ones: `GWSamplerContext` (the data-preprocessing conditioning map), the
GNPE factors and steps (`GNPEKernelFactor`, `GNPEFlowFactor`, `GNPEKernelCorrection`),
the `RAReparam` sky-frame reparametrization, and `GWComposedSampler`, which assembles
the chain for plain NPE, multi-iteration GNPE, or single-step GNPE.
"""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import yaml
from astropy.time import Time
from bilby.core.prior import DeltaFunction, PriorDict, Uniform
from bilby.gw.detector import InterferometerList
from torchvision.transforms import Compose

from dingo.core.density import (
    interpolated_log_prob_multi,
    interpolated_sample_and_log_prob_multi,
)
from dingo.core.factors import (
    ChainComposer,
    ComposedSampler,
    DeltaFactor,
    Factor,
    FlowFactor,
    GibbsBlock,
    Reparametrization,
    Standardization,
    TargetCorrection,
    _base_model_metadata,
)
from dingo.core.multiprocessing import apply_func_with_multiprocessing
from dingo.core.posterior_models import BasePosteriorModel
from dingo.core.transforms import GetItem, RenameKey
from dingo.gw.conversion import change_spin_conversion_phase
from dingo.gw.domains import build_domain, MultibandedFrequencyDomain
from dingo.gw.frequency_updates import (
    _validate_maximum_frequency,
    _validate_minimum_frequency,
)
from dingo.gw.gwutils import get_extrinsic_prior_dict
from dingo.gw.likelihood import StationaryGaussianGWLikelihood
from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.transforms import (
    CopyToExtrinsicParameters,
    DecimateWaveformsAndASDS,
    GetDetectorTimes,
    GNPEBase,
    GNPECoalescenceTimes,
    MaskDataForFrequencyRangeUpdate,
    PostCorrectGeocentTime,
    RepackageStrainsAndASDS,
    SelectStandardizeRepackageParameters,
    TimeShiftStrain,
    ToTorch,
    WhitenAndScaleStrain,
)

logger = logging.getLogger(__name__)


def _frequency_range_update(domain, event_metadata) -> Optional[dict]:
    """The event's requested frequency range when it differs from the data-domain
    bounds, else `None`. Values may be floats or per-detector dicts; defaults are
    the domain bounds. A request merely *wider* than the domain also triggers --
    data generation writes base-domain bounds, which can exceed a multibanded
    domain's quantized band edge (e.g. 1099.0 vs 1098.875) -- and the resulting
    mask is then an identity."""
    if event_metadata is None:
        return None
    minimum = event_metadata.get("minimum_frequency", domain.f_min)
    maximum = event_metadata.get("maximum_frequency", domain.f_max)

    def normalize(value):
        return set(value.values()) if isinstance(value, dict) else {value}

    if normalize(minimum) == {domain.f_min} and normalize(maximum) == {domain.f_max}:
        return None
    return {"minimum_frequency": minimum, "maximum_frequency": maximum}


class GWSamplerContext:
    """
    Per-event shared GW state: the data `d` and everything derived from it. Referenced
    by every factor in a chain, and serialized as the transport state between pipe
    stages.

    This implements the `dingo.core.factors.SamplerContext` protocol. It owns the
    one-time data preprocessing (`prepared_data`) and builds the exact likelihood
    (`likelihood`) used by likelihood-based factors (synthetic phase) and importance
    sampling.

    Event metadata lives here (not on individual factors): it is a property of the data,
    and it drives frequency cropping, the t_ref/RA correction, and the likelihood.

    The data representation is part of a context's identity: importance-sampling
    settings updates produce a *derived* context (`derive`) rather than arguments to
    `likelihood()`. Contexts are treated as immutable -- to change the representation,
    derive a new one.

    Note that the representation vocabulary here (frequency domains, multibanded
    decimation, the base-domain likelihood view, frequency-range masking) is specific
    to this domain family. A new domain family should get its own context class
    implementing the same interface (`prepared_data` / `likelihood` / `prior` /
    `derive`) rather than extending this one.
    """

    def __init__(
        self,
        domain,
        data_prep: Compose,
        event_data: dict,
        event_metadata: Optional[dict] = None,
        model_metadata: Optional[dict] = None,
        device: Union[torch.device, str] = "cpu",
        use_base_domain: bool = False,
        wfg_delta_f: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        domain : Domain
            The data frequency domain.
        data_prep : Compose
            The one-time data-preprocessing transform chain (whiten / decimate /
            repackage).
        event_data : dict
            The raw event data `d` (strain + ASDs per detector), i.e. `EventDataset.data`.
            Consumed lazily by `prepared_data()` and reused for the likelihood.
        event_metadata : dict, optional
            Per-event metadata; drives frequency cropping, the RA correction, and the
            likelihood reference time.
        model_metadata : dict, optional
            The metadata of the model defining this analysis (dataset + train
            settings); the source for the prior, the likelihood, the detector
            names, and the reference time.
        device : torch.device or str, default "cpu"
            The torch device the chain runs on (the model device); steps that create
            fresh tensors (e.g. `DeltaFactor`) create them here.
        use_base_domain : bool, default False
            For a multibanded domain, evaluate the likelihood on the base
            (undecimated) frequency domain rather than the decimated one. Set via
            `derive`.
        wfg_delta_f : float, optional
            Override for the waveform-generator domain `delta_f` (an updated
            `T = 1/delta_f` cannot be handled by domain projection). Set via `derive`.
        """
        self.domain = domain
        self._data_prep = data_prep
        self.model_metadata = model_metadata
        self.event_data = event_data
        self.event_metadata = event_metadata
        self.device = device
        self.use_base_domain = use_base_domain
        self.wfg_delta_f = wfg_delta_f
        self._prepared: Optional[torch.Tensor] = None
        self._prior: Optional[PriorDict] = None
        self._likelihood: Optional[StationaryGaussianGWLikelihood] = None
        self._likelihood_settings: Optional[dict] = None

    @property
    def detectors(self) -> list[str]:
        """Detector names, read from the model metadata."""
        return self.model_metadata["train_settings"]["data"]["detectors"]

    @property
    def t_ref(self) -> float:
        """Training reference GPS time, read from the model metadata."""
        return self.model_metadata["train_settings"]["data"]["ref_time"]

    @classmethod
    def from_model_metadata(
        cls,
        metadata: dict,
        event_data: dict,
        event_metadata: Optional[dict] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> "GWSamplerContext":
        """Build the context from a model-metadata dict -- no model required. The
        domain, one-time data-prep chain, prior, and likelihood are all defined by
        the metadata (e.g. a saved `Result.settings`); `device` only sets where
        `prepared_data()` and chain-created tensors live.

        Parameters
        ----------
        metadata : dict
            Conditional-model metadata (`dataset_settings` + `train_settings`),
            e.g. the `settings` of a saved `Result`.
        event_data : dict
            The raw event data (strain + ASDs).
        event_metadata : dict, optional
            Per-event metadata.
        device : torch.device or str, default "cpu"
            Device for `prepared_data()` and chain-created tensors.

        Returns
        -------
        GWSamplerContext
        """
        data_settings = metadata["train_settings"]["data"]

        domain = build_domain(metadata["dataset_settings"]["domain"])
        if "domain_update" in data_settings:
            domain.update(data_settings["domain_update"])
        detectors = data_settings["detectors"]

        transforms = []
        # Decimate from the base domain when using a multibanded frequency domain.
        if isinstance(domain, MultibandedFrequencyDomain):
            transforms.append(
                DecimateWaveformsAndASDS(domain, decimation_mode="whitened")
            )
        # Whiten and scale (the network expects standardized data).
        transforms.append(WhitenAndScaleStrain(domain.noise_std))
        # Event frequency-range update: mask the whitened strain/ASDs outside the
        # requested range. Must precede repackaging (ranges may be per-detector).
        # The request is validated against the training crop license the first time
        # prepared_data() runs -- deliberately not here: the license governs the
        # network-input view only, and contexts are also reconstructed for
        # likelihood-only use (e.g. from saved importance-sampling results), where
        # a range that is illegal as network input is legal for ASD masking.
        range_update = _frequency_range_update(domain, event_metadata)
        if range_update is not None:
            transforms.append(
                MaskDataForFrequencyRangeUpdate(domain=domain, **range_update)
            )
        # Repackage strains/ASDs into an array, move to torch, extract the waveform.
        transforms += [
            RepackageStrainsAndASDS(ifos=detectors, first_index=domain.min_idx),
            ToTorch(device=device),
            GetItem("waveform"),
        ]

        return cls(
            domain=domain,
            data_prep=Compose(transforms),
            event_data=event_data,
            event_metadata=event_metadata,
            model_metadata=metadata,
            device=device,
        )

    @classmethod
    def from_model(
        cls,
        model: BasePosteriorModel,
        event_data: dict,
        event_metadata: Optional[dict] = None,
    ) -> "GWSamplerContext":
        """Build the context from a model: its own metadata and its device. Data
        preparation is network-bound, so the settings come from `model.metadata`
        (for a conditional model this equals the base analysis metadata). An
        unconditional model prepares no data, so no context can be built from one;
        for the prior/likelihood views alone, use
        `from_model_metadata(_base_model_metadata(model), ...)`.

        Parameters
        ----------
        model : BasePosteriorModel
            The (conditional) model whose metadata defines the domain and
            preprocessing.
        event_data : dict
            The raw event data (strain + ASDs).
        event_metadata : dict, optional
            Per-event metadata.

        Returns
        -------
        GWSamplerContext
        """
        if model.metadata["train_settings"]["data"].get("unconditional", False):
            raise ValueError(
                "An unconditional model has no data preparation, so a context "
                "cannot be built from it. For the prior/likelihood views, use "
                "GWSamplerContext.from_model_metadata(_base_model_metadata(model), ...)."
            )
        return cls.from_model_metadata(
            model.metadata, event_data, event_metadata, device=model.device
        )

    def derive(
        self,
        updates: Optional[dict] = None,
        use_base_domain: Optional[bool] = None,
    ) -> "GWSamplerContext":
        """Derive a context for the same event under a different data representation.

        The derived context shares the event payload (`event_data`, `event_metadata`),
        the analysis metadata, and the reference time by construction -- parameter
        meaning is preserved, so importance weights between the two representations
        remain well-defined. Only the representation changes: an updated duration `T`
        rebuilds the data domain at `delta_f = 1/T` (and enters waveform generation as
        `wfg_delta_f`), and `use_base_domain` switches a multibanded likelihood to the
        undecimated domain. Frequency-range updates act through `event_metadata` (ASD
        masking in the likelihood) and need no derived state. Caches start fresh.

        Parameters
        ----------
        updates : dict, optional
            Importance-sampling settings updates, as recorded in
            `Result.importance_sampling_metadata["updates"]`. May contain `T`,
            `minimum_frequency`, `maximum_frequency` (and non-domain keys, which are
            ignored here), but no other quantities that define a new domain.
        use_base_domain : bool, optional
            For a multibanded domain, evaluate the likelihood on the base
            (undecimated) frequency domain. `None` keeps this context's setting.

        Returns
        -------
        GWSamplerContext
        """
        domain = self.domain
        wfg_delta_f = self.wfg_delta_f
        if updates and any(
            k in updates for k in ("minimum_frequency", "maximum_frequency", "T")
        ):
            # TODO: Make compatible with MultibandedFrequencyDomain.
            if isinstance(domain, MultibandedFrequencyDomain):
                raise NotImplementedError()

            updates = updates.copy()
            # A duration update is generation-level on both sides: the pipe
            # regenerates the event data at the new T, and the waveform must be
            # generated natively at delta_f = 1/T (a resolution change is not a
            # projection of existing samples). wfg_delta_f persists on the derived
            # context, reaching every downstream likelihood build and further
            # derivations.
            if "T" in updates:
                updates["delta_f"] = 1.0 / updates["T"]
                wfg_delta_f = updates["delta_f"]
                print(
                    f"Updating waveform generation delta_f from "
                    f'{self.model_metadata["dataset_settings"]["domain"]["delta_f"]} '
                    f"to {wfg_delta_f}."
                )

            domain_dict = domain.domain_dict  # Existing settings
            domain_dict.update(
                (k, updates[k]) for k in set(domain_dict).intersection(updates)
            )
            print("Rebuilding domain as follows:")
            print(yaml.dump(domain_dict, default_flow_style=False, sort_keys=False))
            domain = build_domain(domain_dict)

        return type(self)(
            domain=domain,
            data_prep=self._data_prep,
            event_data=self.event_data,
            event_metadata=self.event_metadata,
            model_metadata=self.model_metadata,
            device=self.device,
            use_base_domain=(
                self.use_base_domain if use_base_domain is None else use_base_domain
            ),
            wfg_delta_f=wfg_delta_f,
        )

    def prepared_data(self) -> torch.Tensor:
        """One-time data preprocessing, computed once and cached. An event
        frequency-range update is validated against the training crop license
        before the first preparation."""
        if self._prepared is None:
            self._validate_frequency_range()
            self._prepared = self._data_prep(self.event_data)
        return self._prepared

    def _validate_frequency_range(self):
        """Validate an event frequency-range update: hard bounds against the (base)
        domain, and narrowing only when the network was trained with random strain
        cropping covering the requested range. Applies to the network-input view
        only -- the likelihood view applies the range independently via ASD
        masking."""
        update = _frequency_range_update(self.domain, self.event_metadata)
        if update is None:
            return
        domain = getattr(self.domain, "base_domain", self.domain)
        crop_settings = self.model_metadata["train_settings"]["data"].get(
            "random_strain_cropping"
        )
        _validate_minimum_frequency(
            update["minimum_frequency"], self.detectors, domain, crop_settings
        )
        _validate_maximum_frequency(
            update["maximum_frequency"], self.detectors, domain, crop_settings
        )

    @property
    def prior(self) -> PriorDict:
        """The static prior over all parameters, built once from the model metadata
        (intrinsic + extrinsic priors with Dingo defaults).

        This is the event-independent prior fixed at training time. Importance-sampling
        prior-bound updates and the time / phase split-off for marginalized networks are
        applied downstream (they depend on the evolving analysis state), not here.
        """
        if self._prior is None:
            data_settings = self.model_metadata["train_settings"]["data"]
            intrinsic_prior = self.model_metadata["dataset_settings"]["intrinsic_prior"]
            extrinsic_prior = get_extrinsic_prior_dict(data_settings["extrinsic_prior"])
            self._prior = build_prior_with_defaults(
                {**intrinsic_prior, **extrinsic_prior}
            )
        return self._prior

    def likelihood(
        self,
        time_marginalization_kwargs: Optional[dict] = None,
        phase_marginalization_kwargs: Optional[dict] = None,
        calibration_marginalization_kwargs: Optional[dict] = None,
    ) -> StationaryGaussianGWLikelihood:
        """
        Build the exact GW likelihood on this event's data, in physical parameter space.

        The network's standardized, decimated view of the data is `prepared_data()`; the
        likelihood instead takes the raw event data (`event_data`) and builds its own
        representation -- decimating to the multibanded domain unless `use_base_domain`,
        and masking the ASDs to the event's frequency range. Its reference time is the
        event time (the training-frame right-ascension correction is already applied to the
        samples), or the training reference time when no event time is set.

        The data representation (the domain, `use_base_domain`, `wfg_delta_f`, the
        event's frequency range) is context state -- importance-sampling settings
        updates enter by deriving a new context (`derive`), not as arguments here.
        Marginalization is per-request and enters as arguments.

        The most recently built likelihood is cached with its arguments: a repeated
        call with the same arguments returns the shared instance (the synthetic-phase
        factor requests one per chain chunk), and a call with different arguments
        builds a replacement. In the exact synthetic-phase mode the consumer assigns
        a `phase_grid` attribute on the shared instance.

        Parameters
        ----------
        time_marginalization_kwargs : dict, optional
            Analytically marginalize over `geocent_time`. `t_lower` / `t_upper` are
            filled from the network's (uniform) time prior when not already provided
            (a caller with an updated prior passes its own bounds). Requires a
            time-marginalized network.
        phase_marginalization_kwargs : dict, optional
            Analytically marginalize over `phase`. Requires a uniform [0, 2 pi) phase prior.
        calibration_marginalization_kwargs : dict, optional
            Marginalize over detector calibration uncertainty.

        Returns
        -------
        StationaryGaussianGWLikelihood
        """
        # Capture the call's arguments for the cache comparison; this must stay the
        # first statement so locals() holds exactly the arguments. The comparison
        # happens before the marginalization bounds are filled in below (which is
        # deterministic given the same arguments). The representation fields join the
        # key as a guard against off-contract in-place mutation.
        settings = dict(locals())
        settings.pop("self")
        settings["use_base_domain"] = self.use_base_domain
        settings["wfg_delta_f"] = self.wfg_delta_f
        if settings == self._likelihood_settings:
            return self._likelihood

        if self.event_data is None:
            raise ValueError(
                "Building the likelihood requires event data (strain + ASDs), "
                "which this context does not carry."
            )

        # Marginalizing over a parameter needs the (uniform) prior the network
        # marginalized over; use it to parameterize the requested marginalization.
        # Bounds already provided by the caller win: the importance-sampling layer
        # validates against its evolved prior (prior updates, time/phase
        # split-offs), which this sample-free context cannot see, and passes
        # explicit bounds. The fill below covers standalone (chain) use from the
        # network's static prior.
        if time_marginalization_kwargs is not None and not (
            "t_lower" in time_marginalization_kwargs
            and "t_upper" in time_marginalization_kwargs
        ):
            time_prior = self._marginalized_prior("geocent_time")
            if time_prior is None:
                raise NotImplementedError(
                    "Time marginalization is not compatible with a non-marginalized "
                    "network."
                )
            if type(time_prior) != Uniform:
                raise NotImplementedError(
                    "Only uniform time prior is supported for time marginalization."
                )
            time_marginalization_kwargs = {
                **time_marginalization_kwargs,
                "t_lower": time_prior.minimum,
                "t_upper": time_prior.maximum,
            }
        if phase_marginalization_kwargs is not None:
            phase_prior = self._marginalized_prior("phase")
            if not (
                isinstance(phase_prior, Uniform)
                and (phase_prior._minimum, phase_prior._maximum) == (0, 2 * np.pi)
            ):
                raise ValueError(
                    f"Phase prior should be uniform [0, 2pi) for phase marginalization, "
                    f"but is {phase_prior}."
                )

        dataset_settings = self.model_metadata["dataset_settings"]
        # WaveformGenerator domain -- deliberately the dataset domain WITHOUT any
        # domain_update, so waveform generation mirrors how the training set was
        # generated (possibly wider than the network's data domain); the likelihood
        # projects the generated waveform onto the data domain. An updated
        # T = 1/delta_f enters here: unlike range masking or decimation, a
        # resolution change needs samples that do not exist on the old grid, so it
        # cannot be handled by domain projection and must apply at generation.
        wfg_domain_dict = dataset_settings["domain"]
        if self.wfg_delta_f is not None:
            wfg_domain_dict = {**wfg_domain_dict, "delta_f": self.wfg_delta_f}
        wfg_domain = build_domain(wfg_domain_dict)

        # Likelihood reference time: the event time (the training-frame RA correction has
        # already been applied to the samples), falling back to the training reference.
        if self.event_metadata is not None and "time_event" in self.event_metadata:
            t_ref = self.event_metadata["time_event"]
        else:
            t_ref = self.t_ref

        frequency_update = dict(
            minimum_frequency=self._frequency("minimum_frequency", self.domain.f_min),
            maximum_frequency=self._frequency("maximum_frequency", self.domain.f_max),
        )

        likelihood = StationaryGaussianGWLikelihood(
            wfg_kwargs=dataset_settings["waveform_generator"],
            wfg_domain=wfg_domain,
            data_domain=self.domain,
            event_data=self.event_data,
            t_ref=t_ref,
            time_marginalization_kwargs=time_marginalization_kwargs,
            phase_marginalization_kwargs=phase_marginalization_kwargs,
            calibration_marginalization_kwargs=calibration_marginalization_kwargs,
            use_base_domain=self.use_base_domain,
            frequency_update=frequency_update,
        )
        self._likelihood = likelihood
        self._likelihood_settings = settings
        return likelihood

    def _frequency(self, key: str, default: float):
        """The event's frequency-range override for `key` (min/max), else `default`."""
        if self.event_metadata is None:
            return default
        return self.event_metadata.get(key, default)

    def _marginalized_prior(self, name: str):
        """The prior over `name` if the network marginalized it (i.e. `name` is not an
        inference parameter), else `None` -- used to parameterize likelihood
        marginalization over time / phase."""
        if (
            name
            in self.model_metadata["train_settings"]["data"]["inference_parameters"]
        ):
            return None
        return self.prior.get(name)


def _to_numpy(v) -> np.ndarray:
    """Detach a torch tensor (or coerce anything) to a numpy array."""
    if torch.is_tensor(v):
        return v.detach().cpu().numpy()
    return np.asarray(v)


class SyntheticPhaseFactor(Factor):
    """
    Reconstruct the coalescence phase for a phase-marginalized network:
    `q(phase | theta_rest, d)` from the likelihood on a phase grid. The terminal factor of
    the chain.

    For each incoming `theta_rest` it builds the phase-full likelihood
    (`context.likelihood()`) and evaluates `log L` on a grid over `[0, 2 pi)`, exploiting
    that the waveform modes computed once at `phase = 0` each transform as `exp(-i m phase)`
    -- so the whole grid follows from a single waveform evaluation. The grid is
    exponentiated into a conditional phase distribution, a uniform floor (weight
    `uniform_weight`) is added to keep it mass-covering, and one phase is drawn per sample
    from the interpolated distribution. The returned proposal log-prob
    `log q(phase | theta_rest, d)` joins the chain's proposal density; importance sampling
    then targets the phase-full posterior (with the phase prior re-added).

    Two grid modes: `approximation_22_mode=True` assumes a (2, 2)-dominated signal (the
    whole waveform transforms as `exp(2i phase)`), giving `log L` from the complex overlap
    `Re[(d | h(phase=0)) exp(2i phase)]`; `False` sums the modes exactly and requires the
    waveform generator's `spin_conversion_phase = 0`. Note the entry points differ on the
    default: this factor and `dingo_pipe`'s `PhaseRecoveryDefault` use the exact mode,
    while `Result.sample_synthetic_phase` defaults to the (2, 2) approximation when the
    key is omitted.
    """

    def __init__(
        self,
        conditioning: list[str],
        n_grid: int = 5001,
        approximation_22_mode: bool = False,
        uniform_weight: float = 0.01,
        num_processes: int = 1,
    ):
        """
        Parameters
        ----------
        conditioning : list[str]
            The physical parameters the likelihood needs to generate the waveform
            (everything the chain has produced except `phase`).
        n_grid : int, default 5001
            Number of phase grid points on `[0, 2 pi)`.
        approximation_22_mode : bool, default False
            Use the (2, 2)-mode approximation instead of the exact mode sum.
        uniform_weight : float, default 0.01
            Weight of the uniform floor added to the phase distribution for mass coverage.
        num_processes : int, default 1
            Parallel processes for the per-sample likelihood evaluation and phase sampling.
        """
        self.parameters = ["phase"]
        self.conditioning = list(conditioning)
        self.n_grid = n_grid
        self.approximation_22_mode = approximation_22_mode
        self.uniform_weight = uniform_weight
        self.num_processes = num_processes

    def sample_and_log_prob(self, num_samples, context, given=None):
        """Draw one phase per `theta_rest` row (`num_samples` must be 1); return the phases
        and their proposal log-prob `log q(phase | theta_rest, d)`."""
        if num_samples != 1:
            raise ValueError(
                "Synthetic phase is 1:1; draw one phase per sample (fan_out=1)."
            )
        reference = next(iter(given.values()))
        device = reference.device if torch.is_tensor(reference) else None
        n = len(reference)
        logger.info(f"Estimating synthetic phase for {n} samples.")
        t0 = time.time()
        phases, phase_posterior = self._phase_profile(given, context)
        new_phase, log_prob = interpolated_sample_and_log_prob_multi(
            phases, phase_posterior, self.num_processes
        )
        logger.info(f"Done. This took {time.time() - t0:.2f} s.")
        return (
            {"phase": torch.as_tensor(new_phase, device=device)},
            torch.as_tensor(log_prob, device=device),
        )

    def log_prob(self, theta_i, context, given=None):
        """Evaluate `log q(phase | theta_rest, d)` at the given phases (re-plug / IS)."""
        reference = next(iter(given.values()))
        device = reference.device if torch.is_tensor(reference) else None
        phases, phase_posterior = self._phase_profile(given, context)
        log_prob = interpolated_log_prob_multi(
            phases, phase_posterior, _to_numpy(theta_i["phase"]), self.num_processes
        )
        return torch.as_tensor(log_prob, device=device)

    def describe(self) -> dict:
        return {
            "step": type(self).__name__,
            "parameters": list(self.parameters),
            "conditioning": list(self.conditioning),
            "n_grid": self.n_grid,
            "approximation_22_mode": self.approximation_22_mode,
            "uniform_weight": self.uniform_weight,
        }

    def _phase_profile(self, given, context):
        """The phase grid and the mass-covered (un-normalized) phase distribution, one row
        per sample: evaluate `log L` on the grid, exponentiate (shifted by the per-row
        max), and add the uniform floor."""
        theta = pd.DataFrame({k: _to_numpy(v) for k, v in given.items()})
        # The context (possibly derived for importance sampling) carries the
        # representation; the phase-full likelihood needs no arguments.
        likelihood = context.likelihood()
        phases = np.linspace(0, 2 * np.pi, self.n_grid)
        if self.approximation_22_mode:
            # Assume the waveform is (2, 2)-dominated (transforms as exp(2i phase)), so the
            # phase-dependent log-posterior is Re[(d | h(phase=0)) exp(2i phase)].
            theta = theta.copy()
            theta["phase"] = 0.0
            d_inner_h = likelihood.d_inner_h_complex_multi(theta, self.num_processes)
            phase_log_posterior = np.outer(d_inner_h, np.exp(2j * phases)).real
        else:
            # Exact: each mode m contributes exp(-i m phase); needs spin_conversion_phase=0.
            likelihood.phase_grid = phases
            phase_log_posterior = apply_func_with_multiprocessing(
                likelihood.log_likelihood_phase_grid,
                theta,
                num_processes=self.num_processes,
            )
        phase_posterior = np.exp(
            phase_log_posterior - np.amax(phase_log_posterior, axis=1, keepdims=True)
        )
        # Uniform floor: keep q(phase) > 0 everywhere so importance sampling stays finite.
        phase_posterior += (
            phase_posterior.mean(axis=-1, keepdims=True) * self.uniform_weight
        )
        return phases, phase_posterior


def _build_gnpe_transforms(model: BasePosteriorModel):
    """Build the time-shift GNPE per-step transforms from a model's metadata: the
    proxy blur, the per-row time-shift alignment applied before the network, and the
    post-network geocent-time correction.

    Returns
    -------
    transform_pre, transform_post : Compose
    gnpe_parameters : list[str]
        The GNPE input parameters (detector times).
    inference_parameters : list[str]
    kernel : PriorDict
        The proxy perturbation kernel.
    gnpe_transform : GNPECoalescenceTimes
        The blur transform itself, shared so the kernel factor can call `sample_proxies`.
    """
    meta = _base_model_metadata(model)
    data_settings = meta["train_settings"]["data"]
    ifo_list = InterferometerList(data_settings["detectors"])
    domain = build_domain(meta["dataset_settings"]["domain"])
    if "domain_update" in data_settings:
        domain.update(data_settings["domain_update"])

    gnpe_time_settings = data_settings.get("gnpe_time_shifts")
    if not gnpe_time_settings:
        raise NotImplementedError(
            "Only time-shift GNPE is supported here so far (no gnpe_chirp / gnpe_phase)."
        )

    gnpe_transform = GNPECoalescenceTimes(
        ifo_list,
        gnpe_time_settings["kernel"],
        gnpe_time_settings["exact_equiv"],
        inference=True,
    )
    transform_pre = [
        RenameKey("data", "waveform"),
        gnpe_transform,
        TimeShiftStrain(ifo_list, domain),
        SelectStandardizeRepackageParameters(
            {"context_parameters": data_settings["context_parameters"]},
            data_settings["standardization"],
            device=model.device,
        ),
        RenameKey("waveform", "data"),
    ]
    gnpe_parameters: list[str] = []
    kernel = PriorDict()
    for transform in transform_pre:
        if isinstance(transform, GNPEBase):
            gnpe_parameters += transform.input_parameter_names
            for k, v in transform.kernel.items():
                kernel[k] = v

    inference_parameters = data_settings["inference_parameters"]
    transform_post = [
        SelectStandardizeRepackageParameters(
            {"inference_parameters": inference_parameters},
            data_settings["standardization"],
            inverse=True,
            as_type="dict",
        ),
        PostCorrectGeocentTime(),
        CopyToExtrinsicParameters(
            "ra", "dec", "geocent_time", "chirp_mass", "mass_ratio", "phase"
        ),
        GetDetectorTimes(ifo_list, data_settings["ref_time"]),
    ]
    return (
        Compose(transform_pre),
        Compose(transform_post),
        gnpe_parameters,
        inference_parameters,
        kernel,
        gnpe_transform,
    )


class GNPEKernelFactor(Factor):
    """
    The GNPE perturbation kernel `p(theta_hat | theta)` as a non-network factor.

    `theta` are the detector coalescence times; the kernel adds a bounded perturbation to
    each, giving the proxies `theta_hat` the main network conditions on. The parameter
    block is the proxies, the conditioning is the detector times.
    `sample_and_log_prob` blurs the times into proxies (the proxy update of a Gibbs
    sweep); `log_prob` returns the kernel density `log p(theta_hat | theta)` at the
    proxies and the detector times. One proxy per detector-time row.
    """

    def __init__(self, gnpe_transform: GNPEBase, gnpe_parameters: list[str]):
        """
        Parameters
        ----------
        gnpe_transform : GNPEBase
            The blur transform supplying the kernel and `sample_proxies`.
        gnpe_parameters : list[str]
            The detector-time parameters perturbed into proxies.
        """
        self.gnpe = gnpe_transform
        self.gnpe_parameters = gnpe_parameters
        self.parameters = [p + "_proxy" for p in gnpe_parameters]
        self.conditioning = list(gnpe_parameters)
        self.kernel = gnpe_transform.kernel

    @classmethod
    def from_model(cls, model: BasePosteriorModel) -> "GNPEKernelFactor":
        """Build from the main model's metadata (the kernel / blur transform)."""
        _, _, gnpe_parameters, _, _, gnpe_transform = _build_gnpe_transforms(model)
        return cls(gnpe_transform, gnpe_parameters)

    def sample_and_log_prob(self, num_samples, context, given=None):
        """Blur the conditioning detector times into proxies; `num_samples` must be 1
        (GNPE is 1:1). Returns the proxies and their kernel log-prob."""
        if num_samples != 1:
            raise ValueError("GNPE proxy is 1:1; use fan_out=1.")
        times = {k: given[k] for k in self.gnpe_parameters}
        proxies = self.gnpe.sample_proxies(times)
        return proxies, self.log_prob(proxies, context, given)

    def log_prob(self, theta_i, context, given=None):
        """`log p(theta_hat | theta)` from the kernel, at the proxies (`theta_i`) and
        the detector times (`given`).

        The kernel is a bilby `PriorDict` -- the same object that samples the blur --
        so the density is evaluated in numpy (converting each side first: the times
        and proxies may live on different devices) and returned on the detector
        times' device."""
        reference = next(iter(given.values()))
        device = reference.device if torch.is_tensor(reference) else None
        diffs = {
            k: _to_numpy(given[k]) - _to_numpy(theta_i[f"{k}_proxy"])
            for k in self.kernel.keys()
        }
        return torch.as_tensor(
            self.kernel.ln_prob(diffs, axis=0), dtype=torch.float32, device=device
        )


class GNPEFlowFactor(Factor):
    """
    The GNPE main network `q(theta | theta_hat, d)` as a factor.

    Conditions on the detector-time proxies from `GNPEKernelFactor`: it shifts each
    detector's strain by the corresponding proxy time (standardizing the network input),
    samples the network, and recomputes the detector times from the sampled sky position
    and geocent time. The proxies are supplied, so no blurring happens here.

    The single network factor in either GNPE mode: cycled by a `GibbsBlock` for
    multi-iteration GNPE, or a `ChainComposer` factor for single-step GNPE. The recomputed
    detector times are emitted as extra columns (`produces`): the next Gibbs iteration
    blurs them into fresh proxies, and single-step GNPE evaluates the kernel correction at
    them. One sample per proxy row.
    """

    def __init__(
        self,
        model: BasePosteriorModel,
        transform_pre: Compose,
        transform_post: Compose,
        gnpe_parameters: list[str],
        parameters: list[str],
        aliases: Optional[dict[str, str]] = None,
    ):
        """
        Parameters
        ----------
        model : BasePosteriorModel
            The GNPE main network.
        transform_pre : Compose
            Per-iteration pre-network transforms (proxy bookkeeping, time shift,
            standardization).
        transform_post : Compose
            Post-network transforms (de-standardize, recompute detector times).
        gnpe_parameters : list[str]
            The detector-time parameters.
        parameters : list[str]
            The network's trained inference parameters.
        aliases : dict[str, str], optional
            Trained-name to exposed-name map (e.g. `{"ra": "ra@t_ref"}`).
        """
        self.model = model
        self.transform_pre = transform_pre
        self.transform_post = transform_post
        self.gnpe_parameters = gnpe_parameters
        self.proxy_parameters = [p + "_proxy" for p in gnpe_parameters]
        self.aliases = aliases or {}
        self._net_parameters = parameters
        self.parameters = [self.aliases.get(p, p) for p in parameters]
        self.conditioning = list(self.proxy_parameters)
        # For log_prob: the sampling path de-standardizes (and corrects the log-prob)
        # inside transform_post, but evaluating at a point needs the forward map too.
        # The model's own standardization (network-bound, like FlowFactor's).
        std = model.metadata["train_settings"]["data"]["standardization"]
        self.standardization = Standardization(std["mean"], std["std"])

    @property
    def produces(self) -> list[str]:
        """Emitted columns: the inference block plus the recomputed detector times."""
        return self.parameters + self.gnpe_parameters

    @classmethod
    def from_model(
        cls, model: BasePosteriorModel, aliases: Optional[dict[str, str]] = None
    ) -> "GNPEFlowFactor":
        """Build the GNPE per-iteration transforms from the main model's metadata.

        Parameters
        ----------
        model : BasePosteriorModel
            The GNPE main model.
        aliases : dict[str, str], optional
            Trained-name to canonical-name map (e.g. `{"ra": "ra@t_ref"}`).

        Returns
        -------
        GNPEFlowFactor
        """
        pre, post, gnpe_parameters, inference_parameters, _, _ = _build_gnpe_transforms(
            model
        )
        return cls(
            model, pre, post, gnpe_parameters, inference_parameters, aliases=aliases
        )

    def sample_and_log_prob(self, num_samples, context, given=None):
        """Sample one parameter set per proxy row; `num_samples` must be 1 (GNPE is 1:1).
        Returns theta plus the recomputed detector times, and the network log-prob."""
        if num_samples != 1:
            raise ValueError("GNPE is 1:1; draw one sample per proxy (fan_out=1).")
        proxies = {p: given[p] for p in self.proxy_parameters}
        n_rows = next(iter(proxies.values())).shape[0]
        x = {"extrinsic_parameters": dict(proxies), "parameters": {}}
        d = context.prepared_data().clone()
        x["data"] = d.expand(n_rows, *d.shape)
        x = self.transform_pre(x)
        self.model.network.eval()
        with torch.no_grad():
            if "context_parameters" in x:
                y, log_prob = self.model.sample_and_log_prob(
                    x["data"], x["context_parameters"]
                )
            else:
                y, log_prob = self.model.sample_and_log_prob(x["data"])
        # sample_and_log_prob(num_samples=1) adds a singleton dim; the batch is the proxy
        # rows.
        x["parameters"] = y.squeeze(1)
        x["log_prob"] = log_prob.squeeze(1)
        x = self.transform_post(x)
        params = dict(x["parameters"])
        # Expose trained names under their canonical aliases (e.g. ra -> ra@t_ref).
        params = {self.aliases.get(k, k): v for k, v in params.items()}
        # Surface the recomputed detector times: the next Gibbs iteration's input, and the
        # evaluation point for GNPEKernelFactor's importance-sampling correction.
        for k in self.gnpe_parameters:
            params[k] = x["extrinsic_parameters"][k]
        return params, x["log_prob"]

    def log_prob(self, theta_i, context, given=None):
        """Evaluate the network density `log q(theta | theta_hat, d)` in physical space
        at given `theta_i` (exposed / aliased names), one row per proxy row in `given`.
        Runs the same proxies-present data preparation as sampling (time-shift by the
        proxies, standardize the conditioning), then scores the standardized parameters
        under the network."""
        proxies = {p: given[p] for p in self.proxy_parameters}
        n_rows = next(iter(proxies.values())).shape[0]
        x = {"extrinsic_parameters": dict(proxies), "parameters": {}}
        d = context.prepared_data().clone()
        x["data"] = d.expand(n_rows, *d.shape)
        x = self.transform_pre(x)
        theta_net = {
            net: theta_i[self.aliases.get(net, net)] for net in self._net_parameters
        }
        # Mirror transform_post: sampling shifts geocent_time by the preferred proxy
        # after the network (PostCorrectGeocentTime, using the extrinsic geocent_time
        # set up by the GNPE transform), so score the network in its own output frame
        # by applying the inverse correction first.
        y = {
            "parameters": dict(theta_net),
            "extrinsic_parameters": dict(x["extrinsic_parameters"]),
        }
        theta_net = PostCorrectGeocentTime(inverse=True)(y)["parameters"]
        z = self.standardization.standardize(theta_net, self._net_parameters)
        self.model.network.eval()
        with torch.no_grad():
            if "context_parameters" in x:
                log_prob = self.model.log_prob(z, x["data"], x["context_parameters"])
            else:
                log_prob = self.model.log_prob(z, x["data"])
        return log_prob + self.standardization.log_det(self._net_parameters)


class RAReparam(Reparametrization):
    """
    Rotate right ascension from the network's training reference frame (`ra@t_ref`) to the
    event frame (`ra`).

    The network is trained at a fixed reference time; an event at a different GPS time needs
    the sky rotated by the sidereal-time difference. This is a measure-preserving shift
    modulo 2*pi (`log_det = 0`), so it contributes nothing to the density. `forward`
    produces the event-frame `ra`, `inverse` recovers `ra@t_ref`. The sidereal
    correction is read from the shared context (`t_ref` and the event time).

    The modulo makes the map a bijection on the circle, while the flow's density lives
    on the real line: a sample drawn outside `[0, 2 pi)` is wrapped, so `inverse`
    recovers its principal-branch representative and a re-evaluated `log_prob` refers
    to that branch. Only tail samples outside the bounded `ra` prior are affected.
    """

    def __init__(self):
        self.conditioning = ["ra@t_ref"]
        self.parameters = ["ra"]

    @staticmethod
    def _correction(context) -> float:
        """Sidereal-time difference (event minus reference) in radians; 0 when the event
        time is unset or equal to the reference time."""
        event_metadata = context.event_metadata
        t_event = None if event_metadata is None else event_metadata.get("time_event")
        t_ref = context.t_ref
        if t_event is None or t_event == t_ref:
            return 0.0
        longitude_event = Time(t_event, format="gps", scale="utc").sidereal_time(
            "apparent", "greenwich"
        )
        longitude_reference = Time(t_ref, format="gps", scale="utc").sidereal_time(
            "apparent", "greenwich"
        )
        return (longitude_event - longitude_reference).rad

    def forward(self, given, context):
        correction = self._correction(context)
        if correction == 0.0:
            return {"ra": given["ra@t_ref"]}
        # ra is a bounded angle -> float32 is plenty. The correction is a difference of
        # absolute GPS times, so compute it in float64, but store the wrapped angle float32.
        ra = (given["ra@t_ref"].double() + correction) % (2 * np.pi)
        return {"ra": ra.float()}

    def inverse(self, params, context, given=None):
        correction = self._correction(context)
        if correction == 0.0:
            return {"ra@t_ref": params["ra"]}
        ra_tref = (params["ra"].double() - correction) % (2 * np.pi)
        return {"ra@t_ref": ra_tref.float()}


class SpinConventionReparam(Reparametrization):
    """
    Relabel the precessing-spin angles between Dingo's internal spin-phase
    convention and the physical (Bilby) one.

    Dingo fixes the spin-conversion phase (usually to 0) so that the Cartesian
    spins decouple from the coalescence phase; the whole density / likelihood /
    synthetic-phase pipeline requires that convention, so stored samples keep the
    plain names `theta_jn` / `phi_jl` in the *network* convention throughout. The
    physical convention (spin conversion at the sample's own phase) is what Bilby
    and PESummary mean by the same names, so the relabel happens at the export
    boundary -- this class is its single home. Only `theta_jn` and `phi_jl`
    change; the conversion phase and reference frequency are read from the model
    metadata, and a model trained without a fixed conversion phase (`None`)
    relabels to the identity.

    Unlike `RAReparam` no marked intermediate name is needed: the two conventions
    never coexist in one table -- each world's plain names denote its own
    convention, and this bijection is the boundary crossing.

    Exporting a finished weighted sample set needs no Jacobian (proposal, prior,
    and likelihood transform together), which is how `to_physical` is used. As a
    chain `Step` the bijection is *not* measure-preserving in the flat
    `(theta_jn, phi_jl)` coordinates: it rotates the line of sight rigidly about
    the orbital angular momentum, preserving the spherical measure
    `sin(theta_jn) dtheta dphi`, so
    `log_det = log sin(theta_jn) - log sin(theta_jn')` (verified numerically
    against finite differences through the LAL conversion), and `inverse`
    rebuilds the network convention from the physical one using the invariant
    conditioning the reverse fold supplies.
    """

    def __init__(self, num_processes: int = 1):
        """
        Parameters
        ----------
        num_processes : int, default 1
            Parallel processes for the per-sample LAL spin conversion.
        """
        self.parameters = ["theta_jn", "phi_jl"]
        self.conditioning = [
            "theta_jn",
            "phi_jl",
            "phase",
            "chirp_mass",
            "mass_ratio",
            "a_1",
            "a_2",
            "tilt_1",
            "tilt_2",
            "phi_12",
        ]
        self.num_processes = num_processes

    @property
    def consumes(self) -> list[str]:
        # The bijection overwrites theta_jn / phi_jl in place; the remaining
        # conditioning (phase, masses, tilts, ...) is read-only and must stay in
        # the chain -- the default conditioning-minus-parameters would drop it.
        return []

    def log_det(self, given, context):
        """`log|det J|` of `forward`, per row. The map preserves the spherical
        measure, so the flat-coordinate Jacobian is
        `sin(theta_jn) / sin(theta_jn')` -- verified numerically against finite
        differences through the LAL conversion in
        `test_jacobian_matches_sin_ratio` (agreement ~1e-9)."""
        converted = self.forward(given, context)
        return self._log_det(given["theta_jn"], converted["theta_jn"])

    @staticmethod
    def _log_det(theta_jn_in, theta_jn_out):
        # Compute in double (the LAL conversion is double precision anyway),
        # return in the input dtype: a reparametrization preserves the chain's
        # dtype rather than promoting the summed log_prob.
        log_det = torch.log(torch.sin(theta_jn_in.double())) - torch.log(
            torch.sin(theta_jn_out.double())
        )
        return log_det.to(theta_jn_in.dtype)

    def sample_and_log_prob(self, num_samples, context, given=None):
        """Apply `forward`; contribute `-log|det J|`. Overridden to share the
        single LAL conversion between the transform and its Jacobian (the base
        implementation would convert twice)."""
        if num_samples != 1:
            raise ValueError("A reparametrization is 1:1; use fan_out=1.")
        out = self.forward(given, context)
        return out, -self._log_det(given["theta_jn"], out["theta_jn"])

    @staticmethod
    def _model_convention(model_metadata: dict) -> tuple[float, Optional[float]]:
        """The reference frequency and spin-conversion phase the model trained with."""
        wfg_settings = model_metadata["dataset_settings"]["waveform_generator"]
        return wfg_settings["f_ref"], wfg_settings.get("spin_conversion_phase")

    def to_physical(self, samples: pd.DataFrame, model_metadata: dict) -> pd.DataFrame:
        """Relabel samples from the model's convention to the physical (Bilby) one."""
        f_ref, sc_phase = self._model_convention(model_metadata)
        return change_spin_conversion_phase(
            samples, f_ref, sc_phase, None, num_processes=self.num_processes
        )

    def to_network(self, samples: pd.DataFrame, model_metadata: dict) -> pd.DataFrame:
        """Relabel samples from the physical (Bilby) convention to the model's,
        e.g. to ingest external posteriors for comparison."""
        f_ref, sc_phase = self._model_convention(model_metadata)
        return change_spin_conversion_phase(
            samples, f_ref, None, sc_phase, num_processes=self.num_processes
        )

    def forward(self, given, context):
        # The conversion runs in double; the outputs return in the input dtype
        # and device (cf. RAReparam: compute in float64, store the chain dtype).
        reference = given["theta_jn"]
        theta = pd.DataFrame({k: _to_numpy(v) for k, v in given.items()})
        converted = self.to_physical(theta, context.model_metadata)
        return {
            k: torch.as_tensor(converted[k].to_numpy()).to(
                dtype=reference.dtype, device=reference.device
            )
            for k in self.parameters
        }

    def inverse(self, params, context, given=None):
        # The physical -> network direction also needs the invariant
        # conditioning (phase, masses, tilts), which the reverse fold supplies
        # as `given`.
        if given is None:
            raise ValueError(
                "The spin-convention inverse needs the conditioning block "
                "(phase, masses, tilts); pass it as `given`, or convert "
                "DataFrames with to_network()."
            )
        reference = params["theta_jn"]
        rows = {**given, **params}
        theta = pd.DataFrame({k: _to_numpy(v) for k, v in rows.items()})
        converted = self.to_network(theta, context.model_metadata)
        return {
            k: torch.as_tensor(converted[k].to_numpy()).to(
                dtype=reference.dtype, device=reference.device
            )
            for k in self.parameters
        }


class ProxyOffsetReparam(Reparametrization):
    """
    Reconstruct a physical parameter from a network's offset output and its proxy:
    `X = delta_X + X_proxy`.

    A proxy-conditioned network (e.g. the chirp-mass prior conditioning of
    DINGO-BNS) infers the offset `delta_X = X - X_proxy` rather than `X` itself.
    This step rebuilds `X`, consuming the offset column while keeping the proxy in
    the chain (it is recorded with the samples, like the GNPE time proxies). A pure
    shift at fixed proxy, so `log_det = 0`; `inverse` recovers the offset from the
    proxy the reverse fold supplies.
    """

    def __init__(self, parameter_name: str):
        """
        Parameters
        ----------
        parameter_name : str
            The physical parameter name `X`; the step reads `delta_X` and
            `X_proxy` and produces `X`.
        """
        self.parameter_name = parameter_name
        self.delta_name = f"delta_{parameter_name}"
        self.proxy_name = f"{parameter_name}_proxy"
        self.parameters = [parameter_name]
        self.conditioning = [self.delta_name, self.proxy_name]

    @property
    def consumes(self) -> list[str]:
        # The offset is replaced by the physical parameter; the proxy stays in
        # the chain (recorded with the samples).
        return [self.delta_name]

    def forward(self, given, context):
        return {self.parameter_name: given[self.delta_name] + given[self.proxy_name]}

    def inverse(self, params, context, given=None):
        if given is None or self.proxy_name not in given:
            raise ValueError(
                f"Inverting {self.parameter_name} = {self.delta_name} + "
                f"{self.proxy_name} requires the proxy in `given`."
            )
        return {self.delta_name: params[self.parameter_name] - given[self.proxy_name]}

    def describe(self) -> dict:
        return {
            "step": type(self).__name__,
            "parameters": list(self.parameters),
            "conditioning": list(self.conditioning),
        }


class GNPEKernelCorrection(TargetCorrection):
    """
    The single-step GNPE kernel correction as a target-side chain step.

    Emits `delta_log_prob_target = log p(theta_hat | theta)` -- the GNPE kernel evaluated
    at the proxies and the detector times recomputed from theta -- for importance sampling
    on the joint proposal `q(theta, theta_hat | d)`. Contributes 0 to the proposal
    density and consumes the intermediate detector times.
    """

    def __init__(self, kernel_factor: GNPEKernelFactor):
        self.kernel_factor = kernel_factor
        self.parameters = ["delta_log_prob_target"]
        self.conditioning = list(kernel_factor.parameters) + list(
            kernel_factor.gnpe_parameters
        )
        self.consumes = list(kernel_factor.gnpe_parameters)

    def correction(self, given, context):
        proxies = {p: given[p] for p in self.kernel_factor.parameters}
        times = {k: given[k] for k in self.kernel_factor.gnpe_parameters}
        correction = self.kernel_factor.log_prob(proxies, context, times)
        return {"delta_log_prob_target": correction}


def _ra_aliases(inference_parameters: list[str]) -> dict[str, str]:
    """The RA frame alias (`ra` -> `ra@t_ref`), applied only when the model infers
    `ra`; paired with an `RAReparam` step that maps it back to the event frame."""
    return {"ra": "ra@t_ref"} if "ra" in inference_parameters else {}


def _ra_reparam_steps(inference_parameters: list[str]) -> list:
    """The `RAReparam` step, appended to a chain only when the model infers `ra`."""
    return [RAReparam()] if "ra" in inference_parameters else []


def _delta_prior_steps(prior, inference_parameters: list[str]) -> list:
    """Delta-prior parameters the chain does not produce, as a single `DeltaFactor` step
    (or none). These are pinned constants (e.g. an aligned-spin component fixed to 0).

    Parameters
    ----------
    prior : PriorDict
        The static prior (`GWSamplerContext.prior`); its delta-function entries that are
        not inference parameters become the pinned constants.
    inference_parameters : list of str
        The inferred parameter names.
    """
    fixed = {
        k: p.peak
        for k, p in prior.items()
        if isinstance(p, DeltaFunction) and k not in inference_parameters
    }
    return [DeltaFactor(fixed)] if fixed else []


def _assert_consistent_gnpe_data_prep(init_model, main_model):
    """Assert the init and main GNPE models agree on the data-preprocessing view.

    Multi-iteration GNPE shares one `GWSamplerContext` (built from the main model)
    between the init and main factors, so both read the same `prepared_data()` and
    reference time. That is only valid when the two models agree on everything that
    determines those: the domain, the detectors, and the reference time. Raises
    `ValueError` on any mismatch.

    Parameters
    ----------
    init_model, main_model : BasePosteriorModel
        The GNPE init and main networks.
    """
    init = _base_model_metadata(init_model)
    main = _base_model_metadata(main_model)
    fields = {
        "domain": (
            init["dataset_settings"]["domain"],
            main["dataset_settings"]["domain"],
        ),
        "domain_update": (
            init["train_settings"]["data"].get("domain_update"),
            main["train_settings"]["data"].get("domain_update"),
        ),
        "detectors": (
            init["train_settings"]["data"]["detectors"],
            main["train_settings"]["data"]["detectors"],
        ),
        "ref_time": (
            init["train_settings"]["data"]["ref_time"],
            main["train_settings"]["data"]["ref_time"],
        ),
    }
    mismatched = {k: (i, m) for k, (i, m) in fields.items() if i != m}
    if mismatched:
        details = "; ".join(
            f"{k}: init={i!r} vs main={m!r}" for k, (i, m) in mismatched.items()
        )
        raise ValueError(
            f"GNPE init and main models disagree on the data-preprocessing view "
            f"({details}). They share one context, so they must agree on the domain, "
            f"detectors, and reference time."
        )


class GWComposedSampler(ComposedSampler):
    """
    GW builder and exporter over the generic `ComposedSampler` runner. The `from_*`
    constructors assemble the chain for plain NPE, multi-iteration GNPE, or single-step
    GNPE from model metadata; `to_result` exports the samples to a gw `Result`. All
    GW-specific processing (RA frame, fixed parameters, kernel correction) is expressed as
    chain steps, so there is no post-processing.
    """

    def __init__(
        self,
        composer: ChainComposer,
        context: GWSamplerContext,
        metadata: dict,
        inference_parameters: list[str],
    ):
        """
        Parameters
        ----------
        composer : ChainComposer
            The assembled chain of steps.
        context : GWSamplerContext
            Per-event shared state.
        metadata : dict
            Model metadata, carried through to the exported `Result`.
        inference_parameters : list[str]
            The inferred parameter names.
        """
        super().__init__(composer, context)
        self.metadata = metadata
        self.inference_parameters = inference_parameters
        # Extra provenance merged into settings["sampler"] by to_result -- e.g. the
        # pipe records model checkpoint paths and the density-recovery recipe.
        # Literal-only values (the settings dict round-trips through str/literal_eval).
        self.provenance_extra: dict = {}

    def sampler_provenance(self) -> dict:
        """Provenance of how the samples were made, stored as `settings["sampler"]`
        in the exported `Result`: the executed chain in order (one descriptor per
        step, via `Step.describe()`), plus anything in `provenance_extra`. The
        block is purely a record -- nothing consumes it at load time -- and the
        `version` field allows future consumers (e.g. chain reconstruction from
        file) to evolve the format safely."""
        return {
            "version": 1,
            "implementation": "composed",
            "chain": [step.describe() for step in self.composer.steps],
            **copy.deepcopy(self.provenance_extra),
        }

    @classmethod
    def from_model(
        cls,
        model: BasePosteriorModel,
        event_data: dict,
        event_metadata: Optional[dict] = None,
    ) -> "GWComposedSampler":
        """Build a plain-NPE GW sampler from a model and event data: the flow exposes
        `ra` as `ra@t_ref`, followed by an `RAReparam` to the event frame.

        Parameters
        ----------
        model : BasePosteriorModel
            The NPE model.
        event_data : dict
            The raw event data (strain + ASDs).
        event_metadata : dict, optional
            Per-event metadata.

        Returns
        -------
        GWComposedSampler
        """
        context = GWSamplerContext.from_model(model, event_data, event_metadata)
        metadata = _base_model_metadata(model)
        inference_parameters = metadata["train_settings"]["data"][
            "inference_parameters"
        ]
        factor = FlowFactor.from_model(model, aliases=_ra_aliases(inference_parameters))
        steps = (
            [factor]
            + _ra_reparam_steps(inference_parameters)
            + _delta_prior_steps(context.prior, inference_parameters)
        )
        return cls(
            composer=ChainComposer(steps),
            context=context,
            metadata=metadata,
            inference_parameters=inference_parameters,
        )

    @classmethod
    def from_gnpe_models(
        cls,
        init_model: BasePosteriorModel,
        main_model: BasePosteriorModel,
        event_data: dict,
        event_metadata: Optional[dict] = None,
        num_iterations: int = 30,
    ) -> "GWComposedSampler":
        """Build a multi-iteration time-GNPE sampler from an init + main model pair: the
        init model's data preprocessing, an init `FlowFactor` to seed, and a single
        `GibbsBlock` step -- cycling the GNPE kernel and main-network factors -- in a
        `ChainComposer`, then an `RAReparam` to the event frame. Returns samples without
        a log_prob (Gibbs breaks density access).

        Parameters
        ----------
        init_model : BasePosteriorModel
            The init network (detector times); seeds the Gibbs loop and defines the data
            preprocessing.
        main_model : BasePosteriorModel
            The GNPE main network.
        event_data : dict
            The raw event data (strain + ASDs).
        event_metadata : dict, optional
            Per-event metadata.
        num_iterations : int, default 30
            Number of Gibbs sweeps.

        Returns
        -------
        GWComposedSampler
        """
        _assert_consistent_gnpe_data_prep(init_model, main_model)
        # Build the context from the main model: it owns the analysis (likelihood,
        # prior, inference parameters). The init model shares the data domain and
        # preprocessing (asserted above), so prepared_data() is identical either way.
        context = GWSamplerContext.from_model(main_model, event_data, event_metadata)
        metadata = _base_model_metadata(main_model)
        inference_parameters = metadata["train_settings"]["data"][
            "inference_parameters"
        ]
        init_factor = FlowFactor.from_model(init_model)
        kernel_factor = GNPEKernelFactor.from_model(main_model)
        flow_factor = GNPEFlowFactor.from_model(
            main_model, aliases=_ra_aliases(inference_parameters)
        )
        gibbs = GibbsBlock(init_factor, [kernel_factor, flow_factor], num_iterations)
        steps = (
            [gibbs]
            + _ra_reparam_steps(inference_parameters)
            + _delta_prior_steps(context.prior, inference_parameters)
        )
        return cls(
            ChainComposer(steps),
            context,
            metadata,
            inference_parameters,
        )

    @classmethod
    def from_singlestep_gnpe(
        cls,
        main_model: BasePosteriorModel,
        proxy_source: Factor,
        event_data: dict,
        event_metadata: Optional[dict] = None,
    ) -> "GWComposedSampler":
        """Build a single-step (density-preserving) time-GNPE sampler: a `ChainComposer`
        of `[proxy_source, GNPEFlowFactor, GNPEKernelCorrection, RAReparam]`. The chain is
        autoregressive, so log_prob is preserved, and `GNPEKernelCorrection` emits the
        `delta_log_prob_target` correction that importance sampling adds to the target.

        Parameters
        ----------
        main_model : BasePosteriorModel
            The GNPE main network.
        proxy_source : Factor
            Supplies the detector-time proxies -- a `DeltaFactor` for prior conditioning
            (BNS), or an unconditional NDE for density recovery.
        event_data : dict
            The raw event data (strain + ASDs).
        event_metadata : dict, optional
            Per-event metadata.

        Returns
        -------
        GWComposedSampler
        """
        context = GWSamplerContext.from_model(main_model, event_data, event_metadata)
        metadata = _base_model_metadata(main_model)
        inference_parameters = metadata["train_settings"]["data"][
            "inference_parameters"
        ]
        flow_factor = GNPEFlowFactor.from_model(
            main_model, aliases=_ra_aliases(inference_parameters)
        )
        kernel_factor = GNPEKernelFactor.from_model(main_model)
        steps = (
            [proxy_source, flow_factor, GNPEKernelCorrection(kernel_factor)]
            + _ra_reparam_steps(inference_parameters)
            + _delta_prior_steps(context.prior, inference_parameters)
        )
        return cls(ChainComposer(steps), context, metadata, inference_parameters)

    def to_result(self):
        """Export to a gw `Result` (samples + raw event data + metadata), so the
        existing post-processing pipeline -- synthetic phase, importance sampling,
        evidence, plotting -- runs on the factorized sampler's output unchanged.

        The raw event-data dict (`GWSamplerContext.event_data`) is stored as the
        `Result` context (serialized), and the live `GWSamplerContext` is passed as
        `sampler_context` so `Result` can pull the prior (and, later, the likelihood)
        from it rather than rebuilding them from metadata.
        """
        from dingo.gw.result import Result

        settings = copy.deepcopy(self.metadata)
        settings["sampler"] = self.sampler_provenance()
        data_dict = {
            "samples": self.samples,
            "context": self.context.event_data,
            "event_metadata": self.context.event_metadata,
            "importance_sampling_metadata": None,
            "log_evidence": None,
            "log_noise_evidence": None,
            "settings": settings,
        }
        return Result(dictionary=data_dict, sampler_context=self.context)

    def to_hdf5(self, label="result", outdir="."):
        """Export via `to_result` and save to `<outdir>/<label>.hdf5`."""
        result = self.to_result()
        Path(outdir).mkdir(parents=True, exist_ok=True)
        result.to_file(file_name=Path(outdir, label + ".hdf5"))
