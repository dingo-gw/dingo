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
from dingo.gw.gwutils import get_extrinsic_prior_dict
from dingo.gw.likelihood import StationaryGaussianGWLikelihood
from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.transforms import (
    CopyToExtrinsicParameters,
    DecimateWaveformsAndASDS,
    GetDetectorTimes,
    GNPEBase,
    GNPECoalescenceTimes,
    PostCorrectGeocentTime,
    RepackageStrainsAndASDS,
    SelectStandardizeRepackageParameters,
    TimeShiftStrain,
    ToTorch,
    WhitenAndScaleStrain,
)

logger = logging.getLogger(__name__)


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
        raw_context: dict,
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
        raw_context : dict
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
        self.raw_context = raw_context
        self.event_metadata = event_metadata
        self.device = device
        self._prepared: Optional[torch.Tensor] = None
        self._prior: Optional[PriorDict] = None
        self._likelihood: Optional[StationaryGaussianGWLikelihood] = None
        self._likelihood_settings: Optional[dict] = None

    @classmethod
    def from_model(
        cls,
        model: BasePosteriorModel,
        raw_context: dict,
        event_metadata: Optional[dict] = None,
    ) -> "GWSamplerContext":
        """Build the context (domain + one-time data-prep chain) from a model's metadata.
        The data-prep chain reproduces `GWSampler` preprocessing for the plain-NPE case
        (parameter-dependent transforms are handled per-factor).

        Parameters
        ----------
        model : BasePosteriorModel
            The model whose metadata defines the domain and preprocessing.
        raw_context : dict
            The raw event data (strain + ASDs).
        event_metadata : dict, optional
            Per-event metadata.

        Returns
        -------
        GWSamplerContext
        """
        meta = _base_model_metadata(model)
        data_settings = meta["train_settings"]["data"]

        domain = build_domain(meta["dataset_settings"]["domain"])
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
        # Repackage strains/ASDs into an array, move to torch, extract the waveform.
        # TODO: frequency-range cropping (MaskDataForFrequencyRangeUpdate) -- follow-up.
        transforms += [
            RepackageStrainsAndASDS(ifos=detectors, first_index=domain.min_idx),
            ToTorch(device=model.device),
            GetItem("waveform"),
        ]

        return cls(
            domain=domain,
            detectors=detectors,
            t_ref=data_settings["ref_time"],
            data_prep=Compose(transforms),
            raw_context=raw_context,
            event_metadata=event_metadata,
            base_metadata=meta,
            device=model.device,
        )

    def prepared_data(self) -> torch.Tensor:
        """One-time data preprocessing, computed once and cached."""
        if self._prepared is None:
            self._prepared = self._data_prep(self.raw_context)
        return self._prepared

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
    ) -> StationaryGaussianGWLikelihood:
        """
        Build the exact GW likelihood on this event's data, in physical parameter space.

        The network's standardized, decimated view of the data is `prepared_data()`; the
        likelihood instead takes the raw event data (`raw_context`) and builds its own
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

        Importance-sampling domain rebuilds (an updated `T` / `delta_f` or frequency range)
        are not wired here yet: the likelihood uses this context's domain.

        Parameters
        ----------
        time_marginalization_kwargs : dict, optional
            Analytically marginalize over `geocent_time`; `t_lower` / `t_upper` are filled
            from the network's (uniform) time prior. Requires a time-marginalized network.
        phase_marginalization_kwargs : dict, optional
            Analytically marginalize over `phase`. Requires a uniform [0, 2 pi) phase prior.
        calibration_marginalization_kwargs : dict, optional
            Marginalize over detector calibration uncertainty.
        use_base_domain : bool, default False
            For a multibanded domain, evaluate on the base (undecimated) frequency domain
            rather than the decimated one.

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
        if time_marginalization_kwargs is not None:
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
        # the data domain for generation). delta_f / T importance-sampling updates: TODO.
        wfg_domain = build_domain(dataset_settings["domain"])

        # Likelihood reference time: the event time (the training-frame RA correction has
        # already been applied to the samples), falling back to the training reference.
        if self.event_metadata is not None and "time_event" in self.event_metadata:
            t_ref = self.event_metadata["time_event"]
        else:
            t_ref = self.t_ref

        likelihood = StationaryGaussianGWLikelihood(
            wfg_kwargs=dataset_settings["waveform_generator"],
            wfg_domain=wfg_domain,
            data_domain=self.domain,
            event_data=self.raw_context,
            t_ref=t_ref,
            time_marginalization_kwargs=time_marginalization_kwargs,
            phase_marginalization_kwargs=phase_marginalization_kwargs,
            calibration_marginalization_kwargs=calibration_marginalization_kwargs,
            use_base_domain=use_base_domain,
            frequency_update=dict(
                minimum_frequency=self._frequency(
                    "minimum_frequency", self.domain.f_min
                ),
                maximum_frequency=self._frequency(
                    "maximum_frequency", self.domain.f_max
                ),
            ),
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

    def _phase_profile(self, given, context):
        """The phase grid and the mass-covered (un-normalized) phase distribution, one row
        per sample: evaluate `log L` on the grid, exponentiate (shifted by the per-row
        max), and add the uniform floor."""
        theta = pd.DataFrame({k: _to_numpy(v) for k, v in given.items()})
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
    """Build the time-shift GNPE per-step transforms from a model's metadata (the same
    chains as `GWSamplerGNPE._initialize_transforms`).

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
        std = _base_model_metadata(model)["train_settings"]["data"]["standardization"]
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

    @classmethod
    def from_model(
        cls,
        model: BasePosteriorModel,
        raw_context: dict,
        event_metadata: Optional[dict] = None,
    ) -> "GWComposedSampler":
        """Build a plain-NPE GW sampler from a model and event data: the flow exposes
        `ra` as `ra@t_ref`, followed by an `RAReparam` to the event frame.

        Parameters
        ----------
        model : BasePosteriorModel
            The NPE model.
        raw_context : dict
            The raw event data (strain + ASDs).
        event_metadata : dict, optional
            Per-event metadata.

        Returns
        -------
        GWComposedSampler
        """
        context = GWSamplerContext.from_model(model, raw_context, event_metadata)
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
        raw_context: dict,
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
        raw_context : dict
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
        context = GWSamplerContext.from_model(main_model, raw_context, event_metadata)
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
        raw_context: dict,
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
        raw_context : dict
            The raw event data (strain + ASDs).
        event_metadata : dict, optional
            Per-event metadata.

        Returns
        -------
        GWComposedSampler
        """
        context = GWSamplerContext.from_model(main_model, raw_context, event_metadata)
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

        The raw event-data dict (`GWSamplerContext.raw_context`) is stored as the
        `Result` context (serialized), and the live `GWSamplerContext` is passed as
        `sampler_context` so `Result` can pull the prior (and, later, the likelihood)
        from it rather than rebuilding them from metadata.
        """
        from dingo.gw.result import Result

        data_dict = {
            "samples": self.samples,
            "context": self.context.raw_context,
            "event_metadata": self.context.event_metadata,
            "importance_sampling_metadata": None,
            "log_evidence": None,
            "log_noise_evidence": None,
            "settings": copy.deepcopy(self.metadata),
        }
        return Result(dictionary=data_dict, sampler_context=self.context)
