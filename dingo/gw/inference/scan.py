"""Chirp-mass scan for prior-conditioned (DINGO-BNS) networks.

A chirp-mass-conditioned network infers parameters relative to a proxy value
that must be supplied per event. When no external trigger value is available,
this scan determines it from the data: sweep the proxy over the training
chirp-mass prior, draw a few samples per grid point in a single network pass,
and take the chirp mass of the maximum-likelihood draw (Dax et al., Nature 639,
49 (2025), Methods).

The sweep runs on the ordinary chain machinery: a `SampleTableFactor` roots the
chain in the pin table (one row per grid point), the conditioned `FlowFactor`
draws `num_samples` per row against per-row prepared data (the row-varying
heterodyne), and the usual reconstruction steps follow. The winner is selected
with a phase-marginalized likelihood on the (base-domain) event data.
"""

import math

import numpy as np
import pandas as pd
from bilby.gw.prior import BBHPriorDict

from dingo.core.factors import (
    ChainComposer,
    FlowFactor,
    SampleTableFactor,
    Stage,
    _base_model_metadata,
)
from dingo.gw.inference.context import GWSamplerContext
from dingo.gw.inference.sampler import (
    _delta_prior_steps,
    _ra_adjustments,
    _proxy_offset_steps,
    _ra_aliases,
    _ra_to_event_steps,
)


def chirp_mass_scan_grid(model_metadata: dict, overlap_factor: int = 2) -> np.ndarray:
    """Proxy grid spanning the training chirp-mass prior at `kernel_width /
    overlap_factor` spacing, with the bounds inset by the kernel edges so that
    every chirp mass in the prior is within the kernel of some grid point.

    Parameters
    ----------
    model_metadata : dict
        The model metadata (with `dataset_settings` and `train_settings`).
    overlap_factor : int, default 2
        Grid points per kernel width.

    Returns
    -------
    np.ndarray
        The proxy grid (float64).
    """
    prior = BBHPriorDict(dict(model_metadata["dataset_settings"]["intrinsic_prior"]))[
        "chirp_mass"
    ]
    kernel = BBHPriorDict(
        dict(model_metadata["train_settings"]["data"]["gnpe_chirp"]["kernel"])
    )["chirp_mass"]
    num_points = math.ceil(
        (prior.maximum - prior.minimum)
        / (kernel.maximum - kernel.minimum)
        * overlap_factor
    )
    return np.linspace(
        prior.minimum - kernel.minimum, prior.maximum - kernel.maximum, num_points
    )


def chirp_mass_scan(
    model,
    event_data: dict,
    event_metadata: dict = None,
    fixed_context_parameters: dict = None,
    *,
    num_samples: int = 10,
    overlap_factor: int = 2,
    block_size: int = 32,
    num_processes: int = 1,
) -> dict:
    """Scan for the trigger chirp mass of an event.

    Draws `num_samples` posterior samples at each grid value of the chirp-mass
    proxy (one batched network pass over the whole grid), evaluates a
    phase-marginalized likelihood for every within-prior draw, and returns the
    maximum-likelihood draw's chirp mass as the trigger value.

    Parameters
    ----------
    model : BasePosteriorModel
        A chirp-mass-conditioned model (trained with `gnpe_chirp`).
    event_data : dict
        The raw event data (strain + ASDs).
    event_metadata : dict, optional
        Per-event metadata.
    fixed_context_parameters : dict, optional
        Fixed values for the model's remaining (non-proxy) context parameters,
        e.g. the sky position for a sky-conditioned network. The scan fills
        `chirp_mass_proxy` itself.
    num_samples : int, default 10
        Draws per grid point.
    overlap_factor : int, default 2
        Grid points per kernel width.
    block_size : int, default 32
        Grid points per sweep block; bounds the transient memory of the
        row-wise data preparation and the network batch.
    num_processes : int, default 1
        Parallel processes for the likelihood evaluations.

    Returns
    -------
    dict
        `chirp_mass_trigger` (the winner), `snr` and `max_log_likelihood` (its
        trigger quality), `grid`, `samples` (all within-prior draws with their
        `log_likelihood` and `snr`), and `settings` (the scan configuration).
    """
    metadata = _base_model_metadata(model)
    data_settings = metadata["train_settings"]["data"]
    inference_parameters = data_settings["inference_parameters"]
    context_parameters = data_settings.get("context_parameters") or []
    if (
        "chirp_mass_proxy" not in context_parameters
        or "gnpe_chirp" not in data_settings
    ):
        raise ValueError(
            "The chirp-mass scan requires a model conditioned on chirp_mass_proxy "
            "(trained with gnpe_chirp)."
        )
    pins = dict(fixed_context_parameters or {})
    pins.pop("chirp_mass_proxy", None)
    expected = set(context_parameters) - {"chirp_mass_proxy"}
    if set(pins) != expected:
        raise ValueError(
            f"The scan fills chirp_mass_proxy; provide fixed_context_parameters "
            f"for the remaining context parameters {sorted(expected)}, "
            f"got {sorted(pins)}."
        )

    grid = chirp_mass_scan_grid(metadata, overlap_factor)

    context = GWSamplerContext.from_model(model, event_data, event_metadata)
    flow = FlowFactor.from_model(
        model, aliases=_ra_aliases(inference_parameters + context_parameters)
    )
    ra_to_training, ra_to_event = _ra_adjustments(context_parameters)
    tail_steps = (
        _proxy_offset_steps(inference_parameters, context_parameters)
        + ra_to_event
        + _ra_to_event_steps(inference_parameters)
        + _delta_prior_steps(context.prior, inference_parameters)
    )
    # Sweep the grid in blocks: a fixed table roots the chain (its length is
    # the base count, so composer-level chunking does not apply), and the block
    # size bounds both the transient base-domain memory of the row-wise data
    # preparation and the network batch. The chain is density-free (no stored
    # table log-prob) -- the scan selects on the likelihood, not on a proposal
    # density.
    frames = []
    for block in np.array_split(grid, math.ceil(len(grid) / block_size)):
        # Chain columns carry the network dtype (float32); the data preparation
        # extracts pin values at this precision, exactly as for fixed-proxy
        # pins.
        table = {"chirp_mass_proxy": block.astype(np.float32)}
        table.update(
            {k: np.full(len(block), v, dtype=np.float32) for k, v in pins.items()}
        )
        composer = ChainComposer(
            [SampleTableFactor(table)]
            + ra_to_training
            + [Stage(flow, fan_out=num_samples)]
            + tail_steps
        )
        out, _ = composer.sample_and_log_prob(len(block), context)
        frames.append(pd.DataFrame({k: v.cpu().numpy() for k, v in out.items()}))
    theta = pd.concat(frames, ignore_index=True)

    # Restrict to draws within the prior (unphysical combinations cannot be
    # evaluated); the constraint factors are irrelevant for an argmax.
    prior_keys = [k for k in context.prior if k in theta.columns]
    log_prior = context.prior.ln_prob(theta[prior_keys], axis=0)
    theta = theta.iloc[np.flatnonzero(np.isfinite(log_prior))]

    # Exact likelihood on the event data -- on the base domain for multibanded
    # models -- marginalized over the (unsampled) phase.
    likelihood_context = (
        context.derive(use_base_domain=True)
        if hasattr(context.domain, "base_domain")
        else context
    )
    likelihood = likelihood_context.likelihood(
        phase_marginalization_kwargs={"approximation_22_mode": True}
    )
    likelihood.return_aux_snr = True
    theta_likelihood = theta[[c for c in theta.columns if c != "chirp_mass_proxy"]]
    log_likelihood, snr = likelihood.log_likelihood_multi(
        theta_likelihood, num_processes=num_processes
    ).T

    winner = int(np.argmax(log_likelihood))
    samples = theta.assign(log_likelihood=log_likelihood, snr=snr)
    return {
        "chirp_mass_trigger": float(samples["chirp_mass"].iloc[winner]),
        "snr": float(snr[winner]),
        "max_log_likelihood": float(log_likelihood[winner]),
        "grid": grid,
        "samples": samples,
        "settings": {
            "num_samples": num_samples,
            "overlap_factor": overlap_factor,
            "block_size": block_size,
            "grid_size": len(grid),
        },
    }
