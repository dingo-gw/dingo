"""
Gravitational-wave factors for the factorized sampler.

Concrete ``Factor`` / ``SamplerContext`` implementations for GW inference (see
vault/Hackathon/Factorized_Sampler_Design.md). This first increment covers the plain-NPE
path: a single ``FlowFactor`` whose conditioning map ``f_i`` is the one-time data
preprocessing (decimate / whiten / repackage), run on a ``GWSamplerContext``. The
fixed, synthetic-phase, and GNPE-proxy factors are stubbed for later steps.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from astropy.time import Time
from bilby.core.prior import DeltaFunction
from torchvision.transforms import Compose

from dingo.core.factors import (
    ChainComposer,
    ComposedSampler,
    Conditioning,
    Factor,
    Standardization,
)
from dingo.core.posterior_models import BasePosteriorModel
from dingo.core.transforms import GetItem
from dingo.gw.domains import build_domain, MultibandedFrequencyDomain
from dingo.gw.gwutils import get_extrinsic_prior_dict
from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.transforms import (
    DecimateWaveformsAndASDS,
    RepackageStrainsAndASDS,
    ToTorch,
    WhitenAndScaleStrain,
)


def _base_model_metadata(model: BasePosteriorModel) -> dict:
    """The training metadata describing the data domain / standardization. For an
    unconditional (density-recovery) model this lives under ``metadata["base"]``."""
    metadata = model.metadata
    if metadata["train_settings"]["data"].get("unconditional", False):
        return metadata["base"]
    return metadata


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


class FlowFactor(Factor):
    """
    A factor wrapping a posterior model (NPE flow, FMPE, ...). Its conditioning map
    ``f_i`` applies the cheap parameter-dependent data transform (none for plain NPE;
    time-shift / heterodyne for GNPE -- follow-up), standardizes, then runs the network.

    Encapsulates the network's own standardization: the public interface is physical-in
    / physical-out. Standardized values never leave the factor.
    """

    def __init__(
        self,
        model: BasePosteriorModel,
        parameters: list[str],
        conditioning: Optional[list[str]] = None,
        context_parameters: Optional[list[str]] = None,
    ):
        self.model = model
        self.parameters = parameters
        self.conditioning = conditioning or []
        # Network conditioning inputs (GNPE proxies). Empty for plain NPE.
        self.context_parameters = context_parameters or []
        std = _base_model_metadata(model)["train_settings"]["data"]["standardization"]
        self.standardization = Standardization(std["mean"], std["std"])

    @classmethod
    def from_model(cls, model: BasePosteriorModel) -> "FlowFactor":
        """Build a plain-NPE factor from a model: ``parameters`` and ``context_parameters``
        come from the model's training metadata (first-class conditioning)."""
        data_settings = _base_model_metadata(model)["train_settings"]["data"]
        context_parameters = data_settings.get("context_parameters") or []
        return cls(
            model=model,
            parameters=data_settings["inference_parameters"],
            conditioning=list(context_parameters),
            context_parameters=list(context_parameters),
        )

    def _network_context(self, cond: Conditioning) -> tuple[torch.Tensor, ...]:
        """Assemble the model's positional context: the prepared data, plus standardized
        context parameters when the network conditions on proxies."""
        data = cond.context.prepared_data()
        if not self.context_parameters:
            return (data.unsqueeze(0),)
        # Standardize the (physical) conditioning values the network was trained on.
        ctx = self.standardization.standardize(cond.given, self.context_parameters)
        return data.unsqueeze(0), ctx.unsqueeze(0)

    def sample_and_log_prob(self, num_samples, cond):
        net_context = self._network_context(cond)
        self.model.network.eval()
        with torch.no_grad():
            z, log_prob = self.model.sample_and_log_prob(
                *net_context, num_samples=num_samples
            )
        # Squeeze the batch dimension added for the (single) context.
        z = z.squeeze(0)
        log_prob = log_prob.squeeze(0)
        theta = self.standardization.destandardize(z, self.parameters)
        log_prob = log_prob + self.standardization.log_det(self.parameters)
        return theta, log_prob

    def log_prob(self, theta_i, cond):
        num_samples = next(iter(theta_i.values())).shape[0]
        z = self.standardization.standardize(theta_i, self.parameters)
        data = cond.context.prepared_data()
        data = data.expand(num_samples, *data.shape)
        net_context: tuple[torch.Tensor, ...]
        if not self.context_parameters:
            net_context = (data,)
        else:
            ctx = self.standardization.standardize(cond.given, self.context_parameters)
            net_context = (data, ctx)
        self.model.network.eval()
        with torch.no_grad():
            log_prob = self.model.log_prob(z, *net_context)
        return log_prob + self.standardization.log_det(self.parameters)


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
