"""Per-event sampler context for gravitational-wave inference: the event
data and its derived views -- the network-input representation, the prior,
and the likelihood."""

from __future__ import annotations
import logging
from typing import Optional, Union
import numpy as np
import torch
import yaml
from bilby.core.prior import PriorDict, Uniform
from torchvision.transforms import Compose
from dingo.core.factors import _n_rows
from dingo.core.posterior_models import BasePosteriorModel
from dingo.core.transforms import GetItem
from dingo.gw.domains import build_domain, MultibandedFrequencyDomain
from dingo.gw.frequency_updates import (
    _validate_maximum_frequency,
    _validate_minimum_frequency,
)
from dingo.gw.gwutils import get_extrinsic_prior_dict
from dingo.gw.likelihood import StationaryGaussianGWLikelihood
from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.transforms import (
    DecimateWaveformsAndASDS,
    MaskDataForFrequencyRangeUpdate,
    HeterodynePhase,
    RepackageStrainsAndASDS,
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
        data_prep_conditioning: Optional[list[str]] = None,
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
        data_prep_conditioning : list[str], optional
            Names of the chain-conditioning parameters the data preparation is a
            function of (e.g. `["chirp_mass_proxy"]` for a heterodyning model).
            `prepared_data` requires their values, injects them into the
            transform chain, and keys its cache on them; the values themselves
            have a single owner -- the chain.
        """
        self.domain = domain
        self._data_prep = data_prep
        self.model_metadata = model_metadata
        self.event_data = event_data
        self.event_metadata = event_metadata
        self.device = device
        self.use_base_domain = use_base_domain
        self.wfg_delta_f = wfg_delta_f
        self.data_prep_conditioning = list(data_prep_conditioning or [])
        self._prepared_key: Optional[dict] = None
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
            model.metadata,
            event_data,
            event_metadata,
            device=model.device,
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
            data_prep_conditioning=self.data_prep_conditioning,
        )

    def prepared_data(self, conditioning=None) -> torch.Tensor:
        """The network-input data representation of this event.

        Without `conditioning`: the single shared representation, computed once
        and cached. With `conditioning` (the chain columns available to a
        conditioned factor): row-aligned, one data row per conditioning row.
        Only the columns named in `data_prep_conditioning` are consumed by the
        preparation (e.g. the chirp-mass heterodyne proxy, injected under its
        physical name); the remaining columns condition the network only, and a
        context that consumes nothing serves the shared representation viewed
        across the rows. A constant consumed value (a pinned proxy) is prepared
        once and viewed across the rows; varying values (a sweep) run through
        the batch-native transform chain in one vectorized pass, uncached -- a
        caller sweeping more rows than memory allows blocks its own request.

        An event frequency-range update is validated against the training crop
        license before any preparation.

        Parameters
        ----------
        conditioning : dict[str, torch.Tensor], optional
            The chain conditioning available to the calling factor, one value
            per row. May contain columns irrelevant to the preparation.
        """
        if not self.data_prep_conditioning:
            if self._prepared is None:
                self._validate_frequency_range()
                self._prepared = self._data_prep(self.event_data)
            if conditioning is None:
                return self._prepared
            return self._prepared.expand(_n_rows(conditioning), *self._prepared.shape)

        columns = self._conditioning_columns(conditioning)
        self._validate_frequency_range()
        n_rows = _n_rows(columns)
        if all(torch.all(c == c[0]) for c in columns.values()):
            # N rows of one pinned value: prepare once, view it across the rows.
            key = {name: float(c[0]) for name, c in columns.items()}
            if key != self._prepared_key:
                self._prepared = self._data_prep({**self.event_data, "parameters": key})
                self._prepared_key = key
            return self._prepared.expand(n_rows, *self._prepared.shape)
        parameters = {name: column.numpy() for name, column in columns.items()}
        return self._data_prep(
            {**self._broadcast_event(n_rows), "parameters": parameters}
        )

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
