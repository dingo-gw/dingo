"""
Gravitational-wave factors for the factorized sampler.

GW-specific pieces of the factorized-sampler design (see
vault/Hackathon/Factorized_Sampler_Design.md). The generic ``FlowFactor`` is
domain-agnostic and lives in ``dingo.core.factors``; what is GW-specific is the
``GWSamplerContext`` (which builds the data-preprocessing conditioning map ``f_i``) and
the post-processing in ``GWComposedSampler``. Covers plain NPE (``FlowFactor`` via a
``ChainComposer``) and multi-iteration time-shift GNPE (``GNPEKernelFactor`` +
``GNPEFlowFactor`` cycled by a ``GibbsComposer``); the synthetic-phase factor and
single-step GNPE importance sampling remain for later steps.
"""

from __future__ import annotations

import copy
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from astropy.time import Time
from bilby.core.prior import DeltaFunction, PriorDict
from bilby.gw.detector import InterferometerList
from torchvision.transforms import Compose

from dingo.core.factors import (
    ChainComposer,
    ComposedSampler,
    Composer,
    Factor,
    FlowFactor,
    GibbsComposer,
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


# --- Stubs for later steps -------------------------------------------------------------


class FixedFactor(Factor):
    """``q_i = delta(theta_i - c)``: pins parameters to fixed values (prior-conditioning
    / known proxies). Subsumes ``FixedInitSampler``. TODO: implement."""

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

    The single network factor in either GNPE mode: cycled by ``GibbsComposer`` for
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
    ):
        self.model = model
        self.transform_pre = transform_pre
        self.transform_post = transform_post
        self.gnpe_parameters = gnpe_parameters
        self.proxy_parameters = [p + "_proxy" for p in gnpe_parameters]
        self.parameters = parameters
        self.conditioning = list(self.proxy_parameters)

    @classmethod
    def from_model(cls, model: BasePosteriorModel) -> "GNPEFlowFactor":
        """Build the GNPE per-iteration transforms from the main model's metadata."""
        pre, post, gnpe_parameters, inference_parameters, _, _ = _build_gnpe_transforms(
            model
        )
        return cls(model, pre, post, gnpe_parameters, inference_parameters)

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
        # Surface the recomputed detector times: the next Gibbs iteration's input, and the
        # evaluation point for GNPEKernelFactor's importance-sampling correction.
        for k in self.gnpe_parameters:
            params[k] = x["extrinsic_parameters"][k]
        return params, x["log_prob"]

    def log_prob(self, theta_i, cond):
        raise NotImplementedError(
            "GNPEFlowFactor.log_prob -- later step (single-step GNPE importance sampling)."
        )


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
        composer: Composer,
        context: GWSamplerContext,
        metadata: dict,
        inference_parameters: list[str],
    ):
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
        """Build a single-factor (plain-NPE) GW sampler from a model and event data."""
        context = GWSamplerContext.from_model(model, raw_context, event_metadata)
        factor = FlowFactor.from_model(model)
        return cls(
            composer=ChainComposer([factor]),
            context=context,
            metadata=_base_model_metadata(model),
            inference_parameters=factor.parameters,
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
        init model's data preprocessing, an init ``FlowFactor`` to seed, and a
        ``GibbsComposer`` cycling the GNPE kernel and main-network factors. Returns samples
        without a log_prob (Gibbs breaks density access)."""
        context = GWSamplerContext.from_model(init_model, raw_context, event_metadata)
        init_factor = FlowFactor.from_model(init_model)
        kernel_factor = GNPEKernelFactor.from_model(main_model)
        flow_factor = GNPEFlowFactor.from_model(main_model)
        composer = GibbsComposer(
            init_factor, [kernel_factor, flow_factor], num_iterations
        )
        return cls(
            composer,
            context,
            _base_model_metadata(main_model),
            flow_factor.parameters,
        )

    def _correct_reference_time(
        self, samples: Union[dict, pd.DataFrame], inverse: bool = False
    ):
        """Correct the right ascension for the difference between the event time and the
        model's training reference time (fixed detector positions)."""
        event_metadata = self.context.event_metadata
        if event_metadata is None:
            return
        t_event = event_metadata.get("time_event")
        t_ref = self.context.t_ref
        if t_event is None or t_event == t_ref or "ra" not in samples:
            return
        ra = samples["ra"]
        longitude_event = Time(t_event, format="gps", scale="utc").sidereal_time(
            "apparent", "greenwich"
        )
        longitude_reference = Time(t_ref, format="gps", scale="utc").sidereal_time(
            "apparent", "greenwich"
        )
        ra_correction = (longitude_event - longitude_reference).rad
        if not inverse:
            samples["ra"] = (ra + ra_correction) % (2 * np.pi)
        else:
            samples["ra"] = (ra - ra_correction) % (2 * np.pi)

    def _post_process(self, samples: Union[dict, pd.DataFrame], inverse: bool = False):
        intrinsic_prior = self.metadata["dataset_settings"]["intrinsic_prior"]
        extrinsic_prior = get_extrinsic_prior_dict(
            self.metadata["train_settings"]["data"]["extrinsic_prior"]
        )
        prior = build_prior_with_defaults({**intrinsic_prior, **extrinsic_prior})

        if not inverse:
            # Add fixed (delta-prior) parameters not produced by the chain.
            num_samples = len(samples[list(samples.keys())[0]])
            for k, p in prior.items():
                if isinstance(p, DeltaFunction) and k not in samples:
                    print(f"Adding fixed parameter {k} = {p.peak} from prior.")
                    samples[k] = p.peak * np.ones(num_samples)
        else:
            drop = [k for k in samples.keys() if k not in self.inference_parameters]
            if isinstance(samples, pd.DataFrame):
                samples.drop(columns=drop, inplace=True, errors="ignore")
            else:
                for k in drop:
                    samples.pop(k, None)

        self._correct_reference_time(samples, inverse)

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
