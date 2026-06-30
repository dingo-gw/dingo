"""
Gravitational-wave factors for the factorized sampler.

GW-specific pieces of the factorized-sampler design (see
vault/Hackathon/Factorized_Sampler_Design.md). The generic ``FlowFactor`` is
domain-agnostic and lives in ``dingo.core.factors``; what is GW-specific is the
``GWSamplerContext`` (which builds the data-preprocessing conditioning map ``f_i``) and
the post-processing in ``GWComposedSampler``. This first increment covers the plain-NPE
path; the fixed, synthetic-phase, and GNPE-proxy factors are stubbed for later steps.
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
    Factor,
    FlowFactor,
    _base_model_metadata,
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
