"""Per-event sampler context for gravitational-wave inference: the event
data and its derived views -- the network-input representation, the prior,
and the likelihood."""

from __future__ import annotations

import copy
from typing import Optional, Union
import numpy as np
import torch
from bilby.core.prior import PriorDict, Uniform
from bilby.gw.detector import InterferometerList
from torchvision.transforms import Compose
from dingo.core.inference.steps import _n_rows
from dingo.core.posterior_models import BasePosteriorModel
from dingo.core.transforms import GetItem
from dingo.gw.domains import (
    MultibandedFrequencyDomain,
    UniformFrequencyDomain,
    build_domain,
)
from dingo.gw.frequency_updates import (
    _validate_frequency_bound,
    _validate_psd_notches,
    check_detector_update,
    resolve_frequency_bounds,
)
from dingo.gw.gwutils import get_extrinsic_prior_dict
from dingo.gw.likelihood import StationaryGaussianGWLikelihood
from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.transforms import (
    DecimateWaveformsAndASDS,
    HeterodynePhase,
    MaskDataForFrequencyRangeUpdate,
    MaskTokensForFrequencyRangeUpdate,
    NormalizePosition,
    RepackageStrainsAndASDS,
    StrainTokenization,
    ToTorch,
    UnpackDict,
    WhitenAndScaleStrain,
)


def _event_detectors(data_settings: dict, event_metadata) -> list[str]:
    """The analyzed detectors: the event record's list when it carries one, else the
    training list. Per-detector event settings are expanded over this list: a float
    bound applies to all of them, a dict may name any subset of them. A strict
    subset of the training detectors is allowed only for a network trained with
    detector masking (checked with the other event settings before network input
    is prepared)."""
    detectors = (event_metadata or {}).get("detectors")
    return list(detectors) if detectors is not None else data_settings["detectors"]


def _frequency_range_update(domain, event_metadata, detectors) -> Optional[dict]:
    """The event's requested frequency range as complete per-detector dicts over
    `detectors` when it differs from the data-domain bounds, else `None`. The
    request may be floats or dicts naming some of the detectors; defaults are the
    domain bounds. A request merely *wider* than the domain also triggers -- data
    generation writes base-domain bounds, which can exceed a multibanded domain's
    quantized band edge (e.g. 1099.0 vs 1098.875) -- and the resulting mask is
    then an identity."""
    if event_metadata is None:
        return None
    minimum = event_metadata.get("minimum_frequency", domain.f_min)
    maximum = event_metadata.get("maximum_frequency", domain.f_max)

    def normalize(value):
        return set(value.values()) if isinstance(value, dict) else {value}

    if normalize(minimum) == {domain.f_min} and normalize(maximum) == {domain.f_max}:
        return None
    bounds = resolve_frequency_bounds(detectors, domain, minimum, maximum)
    return {
        "minimum_frequency": {d: f_min for d, (f_min, _) in bounds.items()},
        "maximum_frequency": {d: f_max for d, (_, f_max) in bounds.items()},
    }


class GWSamplerContext:
    """
    Per-event shared state for a chain of gravitational-wave steps: the event data
    and everything derived from it.

    The context implements the `dingo.core.inference.context.SamplerContext`
    protocol for this domain family. It prepares the network-input view of the data
    (`prepared_data`), builds the prior (`prior`) and the exact likelihood
    (`likelihood`), and carries the per-event metadata: the event time, which sets
    the likelihood reference time and the right-ascension frame correction, and any
    per-event analysis settings such as a frequency-range update.

    A context is immutable. It is built once from an event dataset and the model
    metadata. The likelihood works on whatever frequency grid the event data are
    on: for importance sampling, the pipe generates data for the requested
    frequency range and duration and builds a new context from them. Whether a
    multibanded model's likelihood uses the base domain is an argument of
    `likelihood()`, like the marginalizations.

    The representation vocabulary here (frequency domains, multibanded decimation,
    the base-domain likelihood view, frequency-range masking) is specific to this
    domain family. A new domain family should get its own context class
    implementing the same interface rather than extending this one.
    """

    def __init__(
        self,
        domain,
        data_prep: Compose,
        event_data: dict,
        event_metadata: Optional[dict] = None,
        model_metadata: Optional[dict] = None,
        device: Union[torch.device, str] = "cpu",
        data_prep_conditioning: Optional[list[str]] = None,
    ):
        """
        Parameters
        ----------
        domain : Domain
            The frequency domain the network was trained on, used to prepare the
            network input. The likelihood uses the grid of the event data instead.
        data_prep : Compose
            The one-time data-preprocessing transform chain (whiten / decimate /
            repackage).
        event_data : dict
            The raw event data `d` (strain + ASDs per detector), i.e. `EventDataset.data`.
            Consumed lazily by `prepared_data()` and reused for the likelihood. An
            injection dict (`Injection.injection()`) may also carry its truths under
            `"parameters"`; these are moved to `event_metadata["injection_parameters"]`.
        event_metadata : dict, optional
            Per-event metadata: the grid the event data are on, the analyzed
            detectors, the per-detector frequency range and PSD notches, the RA
            correction, and the likelihood reference time.
        model_metadata : dict, optional
            The metadata of the model defining this analysis (dataset + train
            settings); the source for the prior, the likelihood, the detector
            names, and the reference time.
        device : torch.device or str, default "cpu"
            The torch device the chain runs on (the model device); steps that create
            fresh tensors (e.g. `DeltaFactor`) create them here.
        data_prep_conditioning : list[str], optional
            Names of the chain-conditioning parameters the data preparation is a
            function of (e.g. `["chirp_mass_proxy"]` for a heterodyning model).
            `prepared_data` requires their values, injects them into the
            transform chain, and keys its cache on them; the values themselves
            have a single owner -- the chain.
        """
        # An injection dict carries its truths under "parameters". Keep them as the
        # event record's injection_parameters (the Result reads them for truth lines
        # and credible levels), leaving the data dict -- and the caller's -- to the
        # strain and ASDs. An existing record wins, so a context rebuilt from a saved
        # Result stays consistent with it.
        if event_data is not None and "parameters" in event_data:
            event_data = dict(event_data)
            truths = event_data.pop("parameters")
            event_metadata = {**(event_metadata or {})}
            event_metadata.setdefault("injection_parameters", truths)
        self.domain = domain
        self._data_prep = data_prep
        self.model_metadata = model_metadata
        self.event_data = event_data
        self.event_metadata = event_metadata
        self.device = device
        self.data_prep_conditioning = list(data_prep_conditioning or [])
        self._prepared_key: Optional[dict] = None
        self._prepared: Optional[torch.Tensor] = None
        self._prior: Optional[PriorDict] = None
        self._likelihood: Optional[StationaryGaussianGWLikelihood] = None
        self._likelihood_settings: Optional[dict] = None

    @property
    def detectors(self) -> list[str]:
        """The analyzed detectors: the event record's, else the training list."""
        return _event_detectors(
            self.model_metadata["train_settings"]["data"], self.event_metadata
        )

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
        detectors = _event_detectors(data_settings, event_metadata)

        transforms = []
        # Chirp-mass GNPE (BNS): heterodyne the raw strain -- before decimation
        # (they do not commute) and on the base domain. The transform draws the
        # chirp mass from the sample's "parameters", which `prepared_data`
        # injects from the chain's conditioning: the proxy value has a single
        # owner (the chain's DeltaFactor), and the preparation is a function of
        # it. Iterated chirp GNPE (heterodyning inside a Gibbs loop) is not
        # implemented: it would require carrying the undecimated strain per
        # sample.
        gnpe_chirp = data_settings.get("gnpe_chirp")
        data_prep_conditioning = []
        if gnpe_chirp is not None:
            data_prep_conditioning = [k + "_proxy" for k in gnpe_chirp["kernel"]]
            transforms.append(
                HeterodynePhase(
                    domain=getattr(domain, "base_domain", domain),
                    order=gnpe_chirp.get("order", 0),
                )
            )
        # Decimate from the base domain when using a multibanded frequency domain.
        if isinstance(domain, MultibandedFrequencyDomain):
            transforms.append(
                DecimateWaveformsAndASDS(domain, decimation_mode="whitened")
            )
        # Whiten and scale (the network expects standardized data).
        transforms.append(WhitenAndScaleStrain(domain.noise_std))
        # Event frequency-range update, expanded over the analyzed detectors (a
        # triangular detector's arms are keyed by arm name). The request is
        # validated against the training license the first time prepared_data()
        # runs -- deliberately not here: the license governs the network-input view
        # only, and contexts are also reconstructed for likelihood-only use (e.g.
        # from saved importance-sampling results), where a range that is illegal as
        # network input is legal for ASD masking. The analyzed detectors and the
        # PSD notches are checked at the same time.
        ifo_names = [ifo.name for ifo in InterferometerList(detectors)]
        range_update = _frequency_range_update(domain, event_metadata, ifo_names)
        tokenization = data_settings.get("tokenization")
        if range_update is not None and tokenization is None:
            # Mask the whitened strain/ASDs outside the requested range. Must
            # precede repackaging (ranges may be per-detector). A tokenized network
            # masks whole tokens instead (below); bin masking would be inert there.
            transforms.append(
                MaskDataForFrequencyRangeUpdate(domain=domain, **range_update)
            )
        # Repackage the analyzed detectors' strains/ASDs into an array.
        transforms.append(
            RepackageStrainsAndASDS(ifos=ifo_names, first_index=domain.min_idx)
        )
        if tokenization is None:
            # Move to torch and extract the waveform.
            transforms += [ToTorch(device=device), GetItem("waveform")]
        else:
            # Tokenize (on numpy arrays, so before ToTorch); a token's detector
            # index is its detector's position in the training list. Then mask the
            # tokens outside the requested range or overlapping a PSD notch, rescale
            # the token positions if the network was trained on normalized ones
            # (after the masks, which compare positions in Hz), and hand the
            # network its three inputs.
            if ifo_names != detectors:
                raise NotImplementedError(
                    f"Tokenized networks do not support multi-arm detectors "
                    f"({detectors} -> {ifo_names})."
                )
            training_detectors = data_settings["detectors"]
            transforms.append(
                StrainTokenization(
                    domain=domain,
                    detectors=detectors,
                    token_size=tokenization.get("token_size"),
                    num_tokens_per_block=tokenization.get("num_tokens_per_block"),
                    drop_last_token=tokenization.get("drop_last_token", False),
                    training_detectors=training_detectors,
                )
            )
            notches = (event_metadata or {}).get("psd_notch_dict")
            if range_update is not None or notches:
                transforms.append(
                    MaskTokensForFrequencyRangeUpdate(
                        domain=domain,
                        detectors=detectors,
                        **(range_update or {}),
                        psd_notch_dict=notches,
                        training_detectors=training_detectors,
                    )
                )
            if tokenization["normalize_position"]:
                transforms.append(NormalizePosition(domain.f_min, domain.f_max))
            transforms += [
                ToTorch(device=device),
                UnpackDict(["waveform", "position", "token_mask"]),
            ]

        return cls(
            domain=domain,
            data_prep=Compose(transforms),
            event_data=event_data,
            event_metadata=event_metadata,
            model_metadata=metadata,
            device=device,
            data_prep_conditioning=data_prep_conditioning,
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
        `from_model_metadata(model.base_metadata, ...)`.

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
                "GWSamplerContext.from_model_metadata(model.base_metadata, ...)."
            )
        return cls.from_model_metadata(
            model.metadata,
            event_data,
            event_metadata,
            device=model.device,
        )

    def prepared_data(self, conditioning=None):
        """The event data in the representation the networks condition on: one
        tensor, or for a tokenized (transformer) network the list
        `[waveform, position, token_mask]`.

        Called without `conditioning`, this returns the single shared
        representation, computed once and cached. Called with `conditioning` (the
        chain columns available to a conditioned factor), the result has one data
        row per conditioning row. Only the columns named in
        `data_prep_conditioning` affect the preparation (for example the
        chirp-mass heterodyne proxy); the other columns condition the network
        alone. When the consumed value is the same in every row (a pinned proxy),
        the data are prepared once and viewed across the rows; when it varies (a
        sweep), the whole batch runs through the transform chain in one pass,
        uncached, so a caller sweeping a large grid should split it into blocks.

        The event's network-input settings (analyzed detectors, frequency range,
        PSD notches) are validated against the training licenses before any
        preparation.

        Parameters
        ----------
        conditioning : dict[str, torch.Tensor], optional
            The chain conditioning available to the calling factor, one value per
            row. May contain columns irrelevant to the preparation.

        Returns
        -------
        torch.Tensor or list of torch.Tensor
        """
        if self._data_prep is None:
            raise ValueError("This context carries no network-input preparation.")
        if self._event_grid() != getattr(self.domain, "base_domain", self.domain):
            raise ValueError(
                "These event data are not on the network's grid; they were "
                "generated for a different frequency range or duration, for "
                "importance sampling. Preparing network input needs the event data "
                "from the sampling stage."
            )
        if not self.data_prep_conditioning:
            if self._prepared is None:
                self._validate_event_settings()
                self._prepared = self._data_prep(self.event_data)
            if conditioning is None:
                return self._prepared
            n_rows = _n_rows(conditioning)
        else:
            columns = self._conditioning_columns(conditioning)
            self._validate_event_settings()
            n_rows = _n_rows(columns)
            if not all(torch.all(c == c[0]) for c in columns.values()):
                parameters = {name: column.numpy() for name, column in columns.items()}
                return self._data_prep(
                    {**self._broadcast_event(n_rows), "parameters": parameters}
                )
            # N rows of one pinned value: prepare once, view it across the rows.
            key = {name: float(c[0]) for name, c in columns.items()}
            if key != self._prepared_key:
                self._prepared = self._data_prep({**self.event_data, "parameters": key})
                self._prepared_key = key
        # The shared representation viewed across the rows.
        if isinstance(self._prepared, list):
            return [p.expand(n_rows, *p.shape) for p in self._prepared]
        return self._prepared.expand(n_rows, *self._prepared.shape)

    def _conditioning_columns(self, conditioning) -> dict[str, torch.Tensor]:
        """Collect the conditioning columns the preparation consumes, keyed by
        their physical names (the `_proxy` suffix names the chain column; the
        transform chain reads the physical parameter), as float64 on the host
        (the heterodyne phase is computed in float64, and a float32 chain
        column must not degrade it; the preparation is a numpy transform
        chain, so columns from a CUDA chain come back to the CPU here)."""
        conditioning = conditioning or {}
        columns = {}
        for name in self.data_prep_conditioning:
            if name not in conditioning:
                raise ValueError(
                    f"This model's data preparation is a function of the chain "
                    f"conditioning `{name}`, which the caller does not provide "
                    f"(pass it to prepared_data, e.g. from the chain's pins)."
                )
            columns[name[: -len("_proxy")]] = (
                torch.as_tensor(conditioning[name], dtype=torch.float64)
                .cpu()
                .reshape(-1)
            )
        return columns

    def _broadcast_event(self, n_rows: int) -> dict:
        """The event arrays broadcast (as read-only views) across `n_rows`
        rows, forming the batched sample dict for a single transform-chain
        pass."""
        return {
            part: (
                {k: np.broadcast_to(v, (n_rows, *np.shape(v))) for k, v in data.items()}
                if isinstance(data, dict)
                else data
            )
            for part, data in self.event_data.items()
        }

    def _validate_event_settings(self):
        """Validate the event's network-input settings against the training
        licenses: the analyzed detectors (a strict subset of the training detectors
        only for a network trained with detector masking), a frequency-range update
        (hard bounds against the base domain; narrowing only under random strain
        cropping or, for a tokenized network, token-range masking), and PSD notches.
        Applies to the network-input view only -- the likelihood applies the range
        and the notched ASDs independently."""
        domain = getattr(self.domain, "base_domain", self.domain)
        data_settings = self.model_metadata["train_settings"]["data"]
        check_detector_update(self.model_metadata, self.detectors)
        update = _frequency_range_update(
            self.domain, self.event_metadata, self.detectors
        )
        if update is not None:
            for bound in ("minimum_frequency", "maximum_frequency"):
                _validate_frequency_bound(
                    update[bound], bound, domain, data_settings, self.detectors
                )
        notches = (self.event_metadata or {}).get("psd_notch_dict")
        if notches:
            _validate_psd_notches(notches, domain, data_settings)

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
        use_base_domain: bool = False,
    ) -> StationaryGaussianGWLikelihood:
        """
        Build the exact GW likelihood on this event's data, in physical parameter
        space.

        The likelihood does not depend on the network. It works on the frequency
        grid of the event data, which the pipe generates to cover the requested
        frequency range, whether inside the network's band or beyond it. Each
        detector's frequency range comes from the event metadata: the ASDs are
        masked outside it, and the calibration spline nodes are placed across it.
        For a multibanded model the data are decimated onto the bands unless
        `use_base_domain` is set; decimation needs the data on the network's own
        grid. The reference time is the event time, or the training reference
        time when no event time is set.

        The most recently built likelihood is cached: a repeated call with the
        same arguments returns the shared instance, and a call with different
        arguments builds a replacement.

        Parameters
        ----------
        time_marginalization_kwargs : dict, optional
            Analytically marginalize over `geocent_time`. Must carry explicit
            `t_lower` / `t_upper` bounds; the caller (the importance-sampling layer
            of `Result`) supplies them from its prior. Requires a time-marginalized
            network.
        phase_marginalization_kwargs : dict, optional
            Analytically marginalize over `phase`. Requires a uniform [0, 2 pi) phase prior.
        calibration_marginalization_kwargs : dict, optional
            Marginalize over detector calibration uncertainty.
        use_base_domain : bool, default False
            For a multibanded model, evaluate on the undecimated base domain.

        Returns
        -------
        StationaryGaussianGWLikelihood
        """
        # The most recently built likelihood is cached, keyed on the requested
        # marginalizations (copied, so a caller mutating its dict later cannot alias
        # the cache). The comparison happens before the bounds are filled in below,
        # which is deterministic given the same arguments.
        settings = copy.deepcopy(
            {
                "time_marginalization_kwargs": time_marginalization_kwargs,
                "phase_marginalization_kwargs": phase_marginalization_kwargs,
                "calibration_marginalization_kwargs": calibration_marginalization_kwargs,
                "use_base_domain": use_base_domain,
            }
        )
        if settings == self._likelihood_settings:
            return self._likelihood

        if self.event_data is None:
            raise ValueError(
                "Building the likelihood requires event data (strain + ASDs), "
                "which this context does not carry."
            )

        # The marginalization bounds are the caller's responsibility: the
        # importance-sampling layer validates them against its evolved prior
        # (prior updates, time/phase split-offs), which this sample-free context
        # cannot see.
        if time_marginalization_kwargs is not None and not (
            "t_lower" in time_marginalization_kwargs
            and "t_upper" in time_marginalization_kwargs
        ):
            raise ValueError(
                "time_marginalization_kwargs requires explicit t_lower / t_upper "
                "bounds."
            )
        if phase_marginalization_kwargs is not None:
            # Requires a phase-marginalized network (phase not inferred) with the
            # standard uniform phase prior.
            data_settings = self.model_metadata["train_settings"]["data"]
            if "phase" in data_settings["inference_parameters"]:
                raise ValueError(
                    "Phase marginalization requires a phase-marginalized network, "
                    "but this network infers phase."
                )
            phase_prior = self.prior.get("phase")
            if not (
                isinstance(phase_prior, Uniform)
                and (phase_prior._minimum, phase_prior._maximum) == (0, 2 * np.pi)
            ):
                raise ValueError(
                    f"Phase prior should be uniform [0, 2pi) for phase marginalization, "
                    f"but is {phase_prior}."
                )

        dataset_settings = self.model_metadata["dataset_settings"]
        # The pipe records the grid it generated the event data on: the network's
        # grid, or a wider one if importance sampling asked for a wider frequency
        # range or a different duration.
        # Waveforms are generated as for the training set: on the dataset's domain,
        # before any domain_update.
        wfg_domain = build_domain(dataset_settings["domain"])
        network = getattr(self.domain, "base_domain", self.domain)
        grid = self._event_grid()
        if grid == network:
            data_domain = self.domain
        else:
            if (
                isinstance(self.domain, MultibandedFrequencyDomain)
                and not use_base_domain
            ):
                raise ValueError(
                    "Data generated for another frequency range or duration cannot "
                    "be decimated onto the network's bands; evaluate on the base "
                    "domain (use_base_domain=True)."
                )
            data_domain = grid
            # The waveform generator has to cover the data grid at its resolution.
            wfg_domain = getattr(wfg_domain, "base_domain", wfg_domain)
            wfg_domain = UniformFrequencyDomain(
                min(wfg_domain.f_min, grid.f_min),
                max(wfg_domain.f_max, grid.f_max),
                grid.delta_f,
            )

        # Likelihood reference time: the event time (the training-frame RA correction has
        # already been applied to the samples), falling back to the training reference.
        if self.event_metadata is not None and "time_event" in self.event_metadata:
            t_ref = self.event_metadata["time_event"]
        else:
            t_ref = self.t_ref

        metadata = self.event_metadata or {}
        bounds = resolve_frequency_bounds(
            list(self.event_data["waveform"]),
            data_domain,
            metadata.get("minimum_frequency"),
            metadata.get("maximum_frequency"),
        )
        frequency_update = dict(
            minimum_frequency={d: f_min for d, (f_min, _) in bounds.items()},
            maximum_frequency={d: f_max for d, (_, f_max) in bounds.items()},
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

    def _event_grid(self) -> UniformFrequencyDomain:
        """The grid the event data are on, as recorded in the event metadata by the
        pipe. Older event files carry no record: they are on the network's grid, or,
        after a duration update, on the network's band at the new resolution."""
        metadata = self.event_metadata or {}
        if metadata.get("domain") is not None:
            return build_domain(metadata["domain"])
        network = getattr(self.domain, "base_domain", self.domain)
        T = metadata.get("T")
        if T is not None and abs(1.0 / T - network.delta_f) > 1e-12:
            return UniformFrequencyDomain(network.f_min, network.f_max, 1.0 / T)
        return network
