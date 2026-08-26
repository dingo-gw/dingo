"""The composed gravitational-wave sampler and its chain builders."""

from __future__ import annotations
import copy
from pathlib import Path
from typing import Optional
from bilby.core.prior import DeltaFunction
from dingo.core.inference.composer import ChainComposer, ComposedSampler, GibbsBlock
from dingo.core.inference.steps import (
    DeltaFactor,
    Factor,
    FlowFactor,
    ProxyOffsetReparam,
)
from dingo.core.posterior_models import BasePosteriorModel
from dingo.gw.inference.context import GWSamplerContext
from dingo.gw.inference.steps import (
    RAToTrainingFrame,
    GNPEFlowFactor,
    GNPEKernelCorrection,
    GNPEKernelFactor,
    RAToEventFrame,
)


def _ra_aliases(inference_parameters: list[str]) -> dict[str, str]:
    """The RA frame alias (`ra` -> `ra@t_ref`), applied only when the model infers
    `ra`; paired with an `RAToEventFrame` step that maps it back to the event frame."""
    return {"ra": "ra@t_ref"} if "ra" in inference_parameters else {}


def _proxy_offset_steps(
    inference_parameters: list[str], context_parameters: list[str]
) -> list:
    """One offset reconstruction (`X = delta_X + X_proxy`) per `delta_X` the
    network infers whose proxy it conditions on."""
    return [
        ProxyOffsetReparam(p[len("delta_") :])
        for p in inference_parameters
        if p.startswith("delta_")
        and p[len("delta_") :] + "_proxy" in context_parameters
    ]


def _ra_to_event_steps(inference_parameters: list[str]) -> list:
    """The `RAToEventFrame` step, appended to a chain only when the model infers `ra`."""
    return [RAToEventFrame()] if "ra" in inference_parameters else []


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


def single_network_steps(model, prior) -> list:
    """The steps of a single-network chain, without its root.

    That is: the flow (with the right-ascension frame adjustments for a pinned or
    inferred sky position), the proxy-offset reconstructions, the rotation to the
    event frame, and the delta-prior fillers. `from_model` prepends a `DeltaFactor`
    of the pins when the model has context parameters; the chirp-mass scan prepends
    a `SampleTableFactor` with one row per proxy grid point.

    Parameters
    ----------
    model : BasePosteriorModel
        The (conditional) model.
    prior : PriorDict
        The static prior (`GWSamplerContext.prior`), for the delta-prior fillers.

    Returns
    -------
    list of chain steps
    """
    data_settings = model.base_metadata["train_settings"]["data"]
    inference_parameters = data_settings["inference_parameters"]
    context_parameters = data_settings.get("context_parameters") or []
    flow = FlowFactor(
        model, aliases=_ra_aliases(inference_parameters + context_parameters)
    )
    # A pinned event-frame sky position is rotated into the training frame before
    # the network, and back into the event frame after it.
    ra_pinned = "ra" in context_parameters
    return (
        ([RAToTrainingFrame()] if ra_pinned else [])
        + [flow]
        + _proxy_offset_steps(inference_parameters, context_parameters)
        + ([RAToEventFrame()] if ra_pinned else [])
        + _ra_to_event_steps(inference_parameters)
        + _delta_prior_steps(prior, inference_parameters)
    )


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
    init = init_model.base_metadata
    main = main_model.base_metadata
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
    GNPE from model metadata; `to_result` exports the samples to a gw `Result`, with
    the model metadata and the sampler provenance as its settings. All GW-specific
    processing (RA frame, fixed parameters, kernel correction) is expressed as chain
    steps, so there is no post-processing.
    """

    def __init__(self, composer: ChainComposer, context: GWSamplerContext):
        """
        Parameters
        ----------
        composer : ChainComposer
            The assembled chain of steps.
        context : GWSamplerContext
            Per-event shared state; its model metadata is carried through to the
            exported `Result`.
        """
        super().__init__(composer, context)

    @property
    def metadata(self) -> dict:
        """The model metadata defining the analysis (from the context)."""
        return self.context.model_metadata

    @classmethod
    def from_model(
        cls,
        model: BasePosteriorModel,
        event_data: dict,
        event_metadata: Optional[dict] = None,
        fixed_context_parameters: Optional[dict] = None,
    ) -> "GWComposedSampler":
        """Build a single-network GW sampler from a model and event data.

        For a plain NPE model the chain is the flow, followed by an
        `RAToEventFrame` rotation to the event frame. A model with
        `context_parameters` (e.g. the DINGO-BNS chirp-mass prior conditioning)
        requires `fixed_context_parameters` pinning all of them: the chain is
        then rooted in a `DeltaFactor` of the pins, the flow conditions on them,
        and each inferred offset `delta_X` with a pinned proxy is reconstructed
        by a `ProxyOffsetReparam` (`X = delta_X + X_proxy`). Proxies that
        parameterize the data preparation (the chirp-mass heterodyne) are read
        from the chain by `prepared_data`. A time-GNPE model is rejected: its
        data must be time-shifted by the proxies, which is the job of the GNPE
        builders.

        Parameters
        ----------
        model : BasePosteriorModel
            The model.
        event_data : dict
            The raw event data (strain + ASDs).
        event_metadata : dict, optional
            Per-event metadata.
        fixed_context_parameters : dict, optional
            Pinned values for the model's `context_parameters`, e.g.
            `{"chirp_mass_proxy": 1.1975, "ra": 3.446, "dec": -0.408}`.

        Returns
        -------
        GWComposedSampler

        Raises
        ------
        ValueError
            For a time-GNPE model, or when the pinned keys do not match the
            model's `context_parameters`.
        """
        context = GWSamplerContext.from_model(model, event_data, event_metadata)
        data_settings = model.base_metadata["train_settings"]["data"]
        context_parameters = data_settings.get("context_parameters") or []
        if data_settings.get("gnpe_time_shifts"):
            raise ValueError(
                "This is a time-GNPE main model (its data are time-shifted by the "
                "proxies): use from_gnpe_models with the init model, or "
                "from_singlestep_gnpe with a proxy source."
            )
        if set(fixed_context_parameters or {}) != set(context_parameters):
            raise ValueError(
                f"The model conditions on {context_parameters}; provide "
                f"fixed_context_parameters with exactly these keys, got "
                f"{sorted(fixed_context_parameters or {})}."
            )
        steps = single_network_steps(model, context.prior)
        if context_parameters:
            steps = [DeltaFactor(fixed_context_parameters)] + steps
        return cls(ChainComposer(steps), context)

    @classmethod
    def from_gnpe_models(
        cls,
        init_model: BasePosteriorModel,
        main_model: BasePosteriorModel,
        event_data: dict,
        event_metadata: Optional[dict] = None,
        num_iterations: int = 30,
    ) -> "GWComposedSampler":
        """Build a multi-iteration time-GNPE sampler from an init + main model pair.

        The chain is a single `GibbsBlock` -- seeded by the init network, then
        cycling the GNPE kernel and the main network for `num_iterations` sweeps
        -- followed by an `RAToEventFrame` rotation. The context is built from
        the main model; the init model must share its data preprocessing
        (asserted). The chain is density-free: the samples carry no log_prob,
        and the density must be recovered before importance sampling.

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
        inference_parameters = main_model.base_metadata["train_settings"]["data"][
            "inference_parameters"
        ]
        init_factor = FlowFactor(init_model)
        kernel_factor = GNPEKernelFactor(main_model)
        flow_factor = GNPEFlowFactor(
            main_model, aliases=_ra_aliases(inference_parameters)
        )
        gibbs = GibbsBlock(init_factor, [kernel_factor, flow_factor], num_iterations)
        steps = (
            [gibbs]
            + _ra_to_event_steps(inference_parameters)
            + _delta_prior_steps(context.prior, inference_parameters)
        )
        return cls(ChainComposer(steps), context)

    @classmethod
    def from_singlestep_gnpe(
        cls,
        main_model: BasePosteriorModel,
        proxy_source: Factor,
        event_data: dict,
        event_metadata: Optional[dict] = None,
    ) -> "GWComposedSampler":
        """Build a single-step (density-preserving) time-GNPE sampler.

        The chain is `[proxy_source, GNPEFlowFactor, GNPEKernelCorrection,
        RAToEventFrame]`: the proxy source supplies the detector-time proxies,
        the main network draws conditioned on them, and the kernel correction
        emits the `delta_log_prob_target` column that importance sampling adds
        to the target. Every step has a tractable density, so the samples carry
        a log_prob.

        Parameters
        ----------
        main_model : BasePosteriorModel
            The GNPE main network.
        proxy_source : Factor
            Supplies the detector-time proxies: a `DeltaFactor` of fixed proxies, or
            an unconditional NDE for density recovery.
        event_data : dict
            The raw event data (strain + ASDs).
        event_metadata : dict, optional
            Per-event metadata.

        Returns
        -------
        GWComposedSampler
        """
        context = GWSamplerContext.from_model(main_model, event_data, event_metadata)
        inference_parameters = main_model.base_metadata["train_settings"]["data"][
            "inference_parameters"
        ]
        flow_factor = GNPEFlowFactor(
            main_model, aliases=_ra_aliases(inference_parameters)
        )
        kernel_factor = GNPEKernelFactor(main_model)
        steps = (
            [proxy_source, flow_factor, GNPEKernelCorrection(kernel_factor)]
            + _ra_to_event_steps(inference_parameters)
            + _delta_prior_steps(context.prior, inference_parameters)
        )
        return cls(ChainComposer(steps), context)

    def to_result(self):
        """Export to a gw `Result` (samples + raw event data + metadata), so the
        existing post-processing pipeline -- synthetic phase, importance sampling,
        evidence, plotting -- runs on the factorized sampler's output unchanged.

        The raw event-data dict (`GWSamplerContext.event_data`) is stored as the
        `Result` context (serialized), and the live `GWSamplerContext` is passed
        as `sampler_context`, so `Result` uses its prior and likelihood rather
        than rebuilding them from metadata.
        """
        from dingo.gw.result import Result

        settings = copy.deepcopy(self.context.model_metadata)
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
