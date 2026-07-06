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
    """

    def __init__(
        self,
        domain,
        detectors: list[str],
        t_ref: float,
        data_prep: Compose,
        event_data: dict,
        event_metadata: Optional[dict] = None,
        base_metadata: Optional[dict] = None,
        device: Union[torch.device, str] = "cpu",
    ):
        """
        Parameters
        ----------
        domain : Domain
            The data frequency domain.
        detectors : list[str]
            Detector names.
        t_ref : float
            Training reference GPS time.
        data_prep : Compose
            The one-time data-preprocessing transform chain (whiten / decimate /
            repackage).
        event_data : dict
            The raw event data `d` (strain + ASDs per detector), i.e. `EventDataset.data`.
            Consumed lazily by `prepared_data()` and reused for the likelihood.
        event_metadata : dict, optional
            Per-event metadata; drives frequency cropping, the RA correction, and the
            likelihood reference time.
        base_metadata : dict, optional
            The base (parameter / waveform / domain) metadata, used to build the
            likelihood.
        device : torch.device or str, default "cpu"
            The torch device the chain runs on (the model device); steps that create
            fresh tensors (e.g. `DeltaFactor`) create them here.
        """
        self.domain = domain
        self.detectors = detectors
        self.t_ref = t_ref
        self._data_prep = data_prep
        self.base_metadata = base_metadata
        self.event_data = event_data
        self.event_metadata = event_metadata
        self.device = device
        self._prepared: Optional[torch.Tensor] = None
        self._prior: Optional[PriorDict] = None
        self._likelihood: Optional[StationaryGaussianGWLikelihood] = None
        self._likelihood_settings: Optional[dict] = None

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
        # prepared_data() runs.
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
            detectors=detectors,
            t_ref=data_settings["ref_time"],
            data_prep=Compose(transforms),
            event_data=event_data,
            event_metadata=event_metadata,
            base_metadata=metadata,
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
        crop_settings = self.base_metadata["train_settings"]["data"].get(
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
            data_settings = self.base_metadata["train_settings"]["data"]
            intrinsic_prior = self.base_metadata["dataset_settings"]["intrinsic_prior"]
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
        use_base_domain: bool = False,
        data_domain=None,
        wfg_delta_f: Optional[float] = None,
        frequency_update: Optional[dict] = None,
    ) -> StationaryGaussianGWLikelihood:
        """
        Build the exact GW likelihood on this event's data, in physical parameter space.

        The network's standardized, decimated view of the data is `prepared_data()`; the
        likelihood instead takes the raw event data (`event_data`) and builds its own
        representation -- decimating to the multibanded domain unless `use_base_domain`,
        and masking the ASDs to the event's frequency range. Its reference time is the
        event time (the training-frame right-ascension correction is already applied to the
        samples), or the training reference time when no event time is set.

        The most recently built likelihood is cached with its arguments: a repeated
        call with the same arguments returns the shared instance (the synthetic-phase
        factor requests one per chain chunk), and a call with different arguments
        builds a replacement. In the exact synthetic-phase mode the consumer assigns
        a `phase_grid` attribute on the shared instance, as
        `Result.sample_synthetic_phase` does.

        The context itself is importance-sampling-state-free: IS-time settings
        updates (a rebuilt data domain, an updated `T` / `delta_f`, an explicit
        frequency range) enter as arguments, supplied by the IS layer
        (`Result._build_likelihood`).

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
        use_base_domain : bool, default False
            For a multibanded domain, evaluate on the base (undecimated) frequency domain
            rather than the decimated one.
        data_domain : Domain, optional
            Override for the data domain (e.g. rebuilt with importance-sampling
            updates); defaults to this context's domain.
        wfg_delta_f : float, optional
            Override for the waveform-generator domain `delta_f` (an updated
            `T = 1/delta_f` cannot be handled by domain projection).
        frequency_update : dict, optional
            `minimum_frequency` / `maximum_frequency` for ASD masking; defaults to
            the event metadata's values with this context's domain as fallback.

        Returns
        -------
        StationaryGaussianGWLikelihood
        """
        # Capture the call's arguments for the cache comparison; this must stay the
        # first statement so locals() holds exactly the arguments. The comparison
        # happens before the marginalization bounds are filled in below (which is
        # deterministic given the same arguments).
        settings = dict(locals())
        settings.pop("self")
        if settings == self._likelihood_settings:
            return self._likelihood

        # Marginalizing over a parameter needs the (uniform) prior the network
        # marginalized over; use it to parameterize the requested marginalization.
        # Bounds already provided by the caller (e.g. from an updated prior) win.
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

        dataset_settings = self.base_metadata["dataset_settings"]
        # WaveformGenerator domain -- generally the dataset domain (it may be wider than
        # the data domain for generation). An updated T = 1/delta_f enters here (it
        # cannot be handled by domain projection).
        wfg_domain_dict = dataset_settings["domain"]
        if wfg_delta_f is not None:
            wfg_domain_dict = {**wfg_domain_dict, "delta_f": wfg_delta_f}
        wfg_domain = build_domain(wfg_domain_dict)

        # Likelihood reference time: the event time (the training-frame RA correction has
        # already been applied to the samples), falling back to the training reference.
        if self.event_metadata is not None and "time_event" in self.event_metadata:
            t_ref = self.event_metadata["time_event"]
        else:
            t_ref = self.t_ref

        if data_domain is None:
            data_domain = self.domain
        if frequency_update is None:
            frequency_update = dict(
                minimum_frequency=self._frequency(
                    "minimum_frequency", self.domain.f_min
                ),
                maximum_frequency=self._frequency(
                    "maximum_frequency", self.domain.f_max
                ),
            )

        likelihood = StationaryGaussianGWLikelihood(
            wfg_kwargs=dataset_settings["waveform_generator"],
            wfg_domain=wfg_domain,
            data_domain=data_domain,
            event_data=self.event_data,
            t_ref=t_ref,
            time_marginalization_kwargs=time_marginalization_kwargs,
            phase_marginalization_kwargs=phase_marginalization_kwargs,
            calibration_marginalization_kwargs=calibration_marginalization_kwargs,
            use_base_domain=use_base_domain,
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
        if name in self.base_metadata["train_settings"]["data"]["inference_parameters"]:
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
    `Re[(d | h(phase=0)) exp(2i phase)]`; `False` (the production default) sums the modes
    exactly and requires the waveform generator's `spin_conversion_phase = 0`.
    """

    def __init__(
        self,
        conditioning: list[str],
        n_grid: int = 5001,
        approximation_22_mode: bool = False,
        uniform_weight: float = 0.01,
        num_processes: int = 1,
        likelihood_kwargs: Optional[dict] = None,
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
        likelihood_kwargs : dict, optional
            Arguments for `context.likelihood()` selecting the likelihood view (e.g.
            `use_base_domain`, a rebuilt `data_domain`, frequency updates), supplied
            by the importance-sampling layer. Defaults to the context's own views.
        """
        self.parameters = ["phase"]
        self.conditioning = list(conditioning)
        self.n_grid = n_grid
        self.approximation_22_mode = approximation_22_mode
        self.uniform_weight = uniform_weight
        self.num_processes = num_processes
        self.likelihood_kwargs = likelihood_kwargs

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
        likelihood = context.likelihood(**(self.likelihood_kwargs or {}))
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

    def inverse(self, params, context):
        correction = self._correction(context)
        if correction == 0.0:
            return {"ra@t_ref": params["ra"]}
        ra_tref = (params["ra"].double() - correction) % (2 * np.pi)
        return {"ra@t_ref": ra_tref.float()}


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
