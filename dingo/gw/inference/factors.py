"""
Gravitational-wave factors for the factorized sampler.

GW-specific pieces of the factorized-sampler design (see
vault/Hackathon/Factorized_Sampler_Design.md). The generic ``FlowFactor`` is
domain-agnostic and lives in ``dingo.core.factors``; what is GW-specific is the
``GWSamplerContext`` (which builds the data-preprocessing conditioning map ``f_i``) and
the post-processing in ``GWComposedSampler``. Covers plain NPE (``FlowFactor`` via a
``ChainComposer``) and multi-iteration time-shift GNPE (``GNPEKernelFactor`` +
``GNPEFlowFactor`` cycled by a ``GibbsBlock``); the synthetic-phase factor and
single-step GNPE importance sampling remain for later steps.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
from astropy.time import Time
from bilby.core.prior import DeltaFunction, PriorDict
from bilby.gw.detector import InterferometerList
from torchvision.transforms import Compose

from dingo.core.factors import (
    ChainComposer,
    ComposedSampler,
    Conditioning,
    Factor,
    FlowFactor,
    GibbsBlock,
    Reparametrization,
    _base_model_metadata,
)
from dingo.core.posterior_models import BasePosteriorModel
from dingo.core.transforms import GetItem, RenameKey
from dingo.gw.domains import build_domain, MultibandedFrequencyDomain
from dingo.gw.gwutils import get_extrinsic_prior_dict
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


class GWSamplerContext:
    """
    Per-event shared GW state: the data ``d`` and everything derived from it. Referenced
    by every factor in a chain, and serialized as the transport state between pipe
    stages.

    This implements the ``dingo.core.factors.SamplerContext`` protocol. It owns the
    one-time data preprocessing (``prepared_data``) and -- once wired in a later step --
    the likelihood used by likelihood-based factors and importance sampling.

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
    ):
        self.domain = domain
        self.detectors = detectors
        self.t_ref = t_ref
        self._data_prep = data_prep
        # The raw event data `d` (strain + ASDs per detector), i.e. EventDataset.data --
        # the same object set as GWSampler.context today. Consumed lazily by
        # prepared_data(); also retained for the likelihood (synthetic-phase / IS factors)
        # and as part of the serialized transport state, neither of which is wired yet.
        self.raw_context = raw_context
        self.event_metadata = event_metadata
        self._prepared: Optional[torch.Tensor] = None

    @classmethod
    def from_model(
        cls,
        model: BasePosteriorModel,
        raw_context: dict,
        event_metadata: Optional[dict] = None,
    ) -> "GWSamplerContext":
        """Build the context (domain + one-time data-prep chain) from a model's
        metadata. The data-prep chain reproduces ``GWSampler`` preprocessing for the
        plain-NPE case (parameter-dependent transforms are handled per-factor)."""
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
        )

    def prepared_data(self) -> torch.Tensor:
        """One-time data preprocessing, computed once and cached."""
        if self._prepared is None:
            self._prepared = self._data_prep(self.raw_context)
        return self._prepared

    def likelihood(self):
        raise NotImplementedError(
            "Likelihood construction moves onto the context in a later step "
            "(synthetic-phase / importance-sampling factors)."
        )


class FixedFactor(Factor):
    """``q_i = delta(theta_i - c)``: pins parameters to fixed values, contributing 0 to the
    proposal log-prob.

    Serves two roles in a chain: the **root**, for prior-conditioning / known proxies
    (single-step GNPE, where later factors condition on the pinned values -- subsumes
    ``FixedInitSampler``), and a non-root **filler** for delta-prior parameters the network
    does not infer (one constant per current row). ``log_prob`` (re-plug) is a later step.
    """

    def __init__(self, values: dict[str, float]):
        self.values = values
        self.parameters = list(values)
        self.conditioning = []

    def sample_and_log_prob(self, num_samples, cond):
        samples = {
            p: torch.full((num_samples,), float(v)) for p, v in self.values.items()
        }
        return samples, torch.zeros(num_samples)

    def log_prob(self, theta_i, cond):
        raise NotImplementedError("FixedFactor.log_prob -- later step.")


# --- Stubs for later steps -------------------------------------------------------------


class SyntheticPhaseFactor(Factor):
    """``q(phase | theta_rest, d)`` from the likelihood on a phase grid; the final,
    non-NN factor, run in the CPU post-processing stage. TODO: port from
    ``Result.sample_synthetic_phase``."""

    def __init__(self):
        self.parameters = ["phase"]
        self.conditioning = []  # conditions on all preceding params via the likelihood

    def sample_and_log_prob(self, num_samples, cond):
        raise NotImplementedError("SyntheticPhaseFactor -- later step.")

    def log_prob(self, theta_i, cond):
        raise NotImplementedError("SyntheticPhaseFactor -- later step.")


def _build_gnpe_transforms(model: BasePosteriorModel):
    """Build the time-shift GNPE per-step transforms from a model's metadata (the same
    chains as ``GWSamplerGNPE._initialize_transforms``).

    Returns
    -------
    transform_pre, transform_post : Compose
    gnpe_parameters : list[str]
        The GNPE input parameters (detector times).
    inference_parameters : list[str]
    kernel : PriorDict
        The proxy perturbation kernel.
    gnpe_transform : GNPECoalescenceTimes
        The blur transform itself, shared so the kernel factor can call ``sample_proxies``.
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
    The GNPE perturbation kernel p(theta_hat | theta) as a factor (non-NN).

    GNPE (arXiv:2111.13139) conditions the main network on "proxy" parameters theta_hat
    drawn from a fixed kernel p(theta_hat | theta). Here theta are the detector coalescence
    times and the kernel adds a bounded perturbation to each. Conditioning on the proxies
    frees the network input to be simplified by them -- ``GNPEFlowFactor`` shifts each
    detector's strain to its proxy time -- at the price of inferring theta and theta_hat
    jointly.

    The parameter block is the proxies; the conditioning is the detector times.
    ``sample_and_log_prob`` blurs the times into proxies (the proxy update of a Gibbs
    sweep). ``log_prob`` returns the kernel density log p(theta_hat | theta); for
    single-step GNPE this is the ``delta_log_prob_target`` importance-sampling correction,
    evaluated at the proxies and the detector times recomputed from theta. One proxy per
    detector-time row.
    """

    def __init__(self, gnpe_transform: GNPEBase, gnpe_parameters: list[str]):
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

    def sample_and_log_prob(self, num_samples, cond):
        """Blur the conditioning detector times into proxies; ``num_samples`` must be 1
        (GNPE is 1:1). Returns the proxies and their kernel log-prob."""
        if num_samples != 1:
            raise ValueError("GNPE proxy is 1:1; use fan_out=1.")
        times = {k: cond.given[k] for k in self.gnpe_parameters}
        proxies = self.gnpe.sample_proxies(times)
        return proxies, self.log_prob(proxies, cond)

    def log_prob(self, theta_i, cond):
        """``log p(theta_hat | theta)`` from the kernel, at the proxies (``theta_i``) and
        the detector times (``cond.given``)."""
        diffs = {}
        for k in self.kernel.keys():
            diff = cond.given[k] - theta_i[f"{k}_proxy"]
            if torch.is_tensor(diff):
                diff = diff.detach().cpu().numpy()
            diffs[k] = np.asarray(diff)
        return torch.as_tensor(self.kernel.ln_prob(diffs, axis=0), dtype=torch.float32)


class GNPEFlowFactor(Factor):
    """
    The GNPE main network q(theta | theta_hat, d) as a factor.

    Conditions on the detector-time proxies from ``GNPEKernelFactor``: it shifts each
    detector's strain by the corresponding proxy time (standardizing the network input),
    samples the network, and recomputes the detector times from the sampled sky position
    and geocent time. The proxies are supplied, so no blurring happens here. Wraps the main
    model and the per-iteration transforms built for ``GWSamplerGNPE``.

    The single network factor in either GNPE mode: cycled by a ``GibbsBlock`` for
    multi-iteration GNPE, or a ``ChainComposer`` factor for single-step GNPE. The recomputed
    detector times are returned as extra columns: the next Gibbs iteration blurs them into
    fresh proxies, and single-step GNPE evaluates the kernel correction at them. One sample
    per proxy row.
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
        self.model = model
        self.transform_pre = transform_pre
        self.transform_post = transform_post
        self.gnpe_parameters = gnpe_parameters
        self.proxy_parameters = [p + "_proxy" for p in gnpe_parameters]
        self.aliases = aliases or {}
        self._net_parameters = parameters
        self.parameters = [self.aliases.get(p, p) for p in parameters]
        self.conditioning = list(self.proxy_parameters)

    @classmethod
    def from_model(
        cls, model: BasePosteriorModel, aliases: Optional[dict[str, str]] = None
    ) -> "GNPEFlowFactor":
        """Build the GNPE per-iteration transforms from the main model's metadata.
        ``aliases`` maps trained names to canonical names (e.g. ``{"ra": "ra@t_ref"}``).
        """
        pre, post, gnpe_parameters, inference_parameters, _, _ = _build_gnpe_transforms(
            model
        )
        return cls(
            model, pre, post, gnpe_parameters, inference_parameters, aliases=aliases
        )

    def sample_and_log_prob(self, num_samples, cond):
        """Sample one parameter set per proxy row; ``num_samples`` must be 1 (GNPE is 1:1).
        Returns theta plus the recomputed detector times, and the network log-prob."""
        if num_samples != 1:
            raise ValueError("GNPE is 1:1; draw one sample per proxy (fan_out=1).")
        proxies = {p: cond.given[p] for p in self.proxy_parameters}
        n_rows = next(iter(proxies.values())).shape[0]
        x = {"extrinsic_parameters": dict(proxies), "parameters": {}}
        d = cond.context.prepared_data().clone()
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

    def log_prob(self, theta_i, cond):
        raise NotImplementedError(
            "GNPEFlowFactor.log_prob -- later step (single-step GNPE importance sampling)."
        )


class RAReparam(Reparametrization):
    """
    Rotate right ascension from the network's training reference frame (``ra@t_ref``) to the
    event frame (``ra``).

    The network is trained at a fixed reference time; an event at a different GPS time needs
    the sky rotated by the sidereal-time difference. This is a measure-preserving shift
    modulo 2*pi (``log_det = 0``), so it contributes nothing to the density. Ported from
    ``GWSamplerMixin._correct_reference_time``: ``forward`` produces the event-frame ``ra``
    for downstream, ``inverse`` recovers ``ra@t_ref`` for re-plug / importance sampling
    (design Q#7). The sidereal correction is read from the shared context (``t_ref`` and the
    event time).
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


def _ra_aliases(inference_parameters: list[str]) -> dict[str, str]:
    """The RA frame alias (``ra`` -> ``ra@t_ref``), applied only when the model infers
    ``ra``; paired with an ``RAReparam`` step that maps it back to the event frame."""
    return {"ra": "ra@t_ref"} if "ra" in inference_parameters else {}


def _ra_reparam_steps(inference_parameters: list[str]) -> list:
    """The ``RAReparam`` step, appended to a chain only when the model infers ``ra``."""
    return [RAReparam()] if "ra" in inference_parameters else []


def _fixed_prior_steps(metadata: dict, inference_parameters: list[str]) -> list:
    """Delta-prior parameters the chain does not produce, as a single ``FixedFactor`` step
    (or none). These are pinned constants (e.g. an aligned-spin component fixed to 0); the
    old sampler injected them in ``_post_process``."""
    intrinsic_prior = metadata["dataset_settings"]["intrinsic_prior"]
    extrinsic_prior = get_extrinsic_prior_dict(
        metadata["train_settings"]["data"]["extrinsic_prior"]
    )
    prior = build_prior_with_defaults({**intrinsic_prior, **extrinsic_prior})
    fixed = {
        k: p.peak
        for k, p in prior.items()
        if isinstance(p, DeltaFunction) and k not in inference_parameters
    }
    return [FixedFactor(fixed)] if fixed else []


class GWComposedSampler(ComposedSampler):
    """
    GW façade: a ``ComposedSampler`` that adds the gravitational-wave post-processing --
    injecting fixed (delta-prior) parameters and correcting the sky position for the
    model's fixed reference time. Ported from ``GWSamplerMixin._post_process``; in the
    longer run the reference-time correction becomes a deterministic reparametrization
    factor in the chain (see the design doc, Q#4/Q#7).
    """

    def __init__(
        self,
        composer: ChainComposer,
        context: GWSamplerContext,
        metadata: dict,
        inference_parameters: list[str],
        kernel_factor: Optional[GNPEKernelFactor] = None,
    ):
        super().__init__(composer, context)
        self.metadata = metadata
        self.inference_parameters = inference_parameters
        # Set for single-step GNPE. Supplies the delta_log_prob_target kernel correction
        # (evaluated in post-processing, out of the proposal sum) that importance sampling
        # applies to the joint target q(theta, theta_hat | d).
        self.kernel_factor = kernel_factor

    @classmethod
    def from_model(
        cls,
        model: BasePosteriorModel,
        raw_context: dict,
        event_metadata: Optional[dict] = None,
    ) -> "GWComposedSampler":
        """Build a plain-NPE GW sampler from a model and event data: the flow exposes
        ``ra`` as ``ra@t_ref``, followed by an ``RAReparam`` to the event frame."""
        context = GWSamplerContext.from_model(model, raw_context, event_metadata)
        metadata = _base_model_metadata(model)
        inference_parameters = metadata["train_settings"]["data"][
            "inference_parameters"
        ]
        factor = FlowFactor.from_model(model, aliases=_ra_aliases(inference_parameters))
        steps = (
            [factor]
            + _ra_reparam_steps(inference_parameters)
            + _fixed_prior_steps(metadata, inference_parameters)
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
        init model's data preprocessing, an init ``FlowFactor`` to seed, and a single
        ``GibbsBlock`` step -- cycling the GNPE kernel and main-network factors -- in a
        ``ChainComposer``, then an ``RAReparam`` to the event frame. Returns samples without
        a log_prob (Gibbs breaks density access)."""
        context = GWSamplerContext.from_model(init_model, raw_context, event_metadata)
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
            + _fixed_prior_steps(metadata, inference_parameters)
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
        """Build a single-step (density-preserving) time-GNPE sampler: a ``ChainComposer``
        of ``[proxy_source, GNPEFlowFactor, RAReparam]``. ``proxy_source`` supplies the
        detector-time proxies -- a ``FixedFactor`` for prior conditioning (BNS), or an
        unconditional NDE for density recovery. Unlike multi-iteration GNPE the chain is
        autoregressive, so log_prob is preserved; the ``GNPEKernelFactor`` supplies the
        ``delta_log_prob_target`` correction that importance sampling adds to the target.
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
            [proxy_source, flow_factor]
            + _ra_reparam_steps(inference_parameters)
            + _fixed_prior_steps(metadata, inference_parameters)
        )
        composer = ChainComposer(steps)
        return cls(
            composer,
            context,
            metadata,
            inference_parameters,
            kernel_factor=kernel_factor,
        )

    def _add_kernel_correction(self, samples: dict):
        """Single-step GNPE: add ``delta_log_prob_target = log p(theta_hat | theta)``,
        evaluated at the proxies and the detector times recomputed from theta, then drop
        those intermediate detector times. This is the joint-target correction importance
        sampling applies; it is deliberately kept out of the proposal ``log_prob``."""
        kf = self.kernel_factor
        proxies = {p: samples[p] for p in kf.parameters}
        times = {k: samples[k] for k in kf.gnpe_parameters}
        correction = kf.log_prob(proxies, Conditioning(self.context, times))
        samples["delta_log_prob_target"] = np.asarray(correction)
        for k in kf.gnpe_parameters:
            samples.pop(k, None)

    def _post_process(self, samples: dict):
        # Fixed (delta-prior) parameters are a FixedFactor in the chain and the RA frame
        # rotation is a chain reparametrization; only the single-step-GNPE kernel correction
        # remains here (moves to a chain step next).
        if self.kernel_factor is not None:
            self._add_kernel_correction(samples)

    def to_result(self):
        """Export to a gw ``Result`` (samples + raw event data + metadata), so the
        existing post-processing pipeline -- synthetic phase, importance sampling,
        evidence, plotting -- runs on the factorized sampler's output unchanged.

        ``context`` is the *raw* event-data dict (``GWSamplerContext.raw_context``, what
        ``Result`` needs to rebuild the likelihood), not the ``SamplerContext`` object.
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
        return Result(dictionary=data_dict)
