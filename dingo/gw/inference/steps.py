"""Gravitational-wave chain steps: the GNPE factors, the synthetic-phase and
synthetic-phase-psi factors, and the coordinate reparametrizations."""

from __future__ import annotations
import logging
import time
from typing import Optional
import bilby
import lalsimulation as LS
import numpy as np
import pandas as pd
import torch
from astropy.time import Time
from bilby.gw.detector import InterferometerList
from bilby.gw.utils import ln_i0
from torchvision.transforms import Compose
from dingo.core.density import (
    interpolated_log_prob_multi,
    interpolated_sample_and_log_prob_multi,
)
from dingo.core.inference.steps import (
    Factor,
    Reparametrization,
    Standardization,
    TargetCorrection,
    _n_rows,
)
from dingo.core.multiprocessing import apply_func_with_multiprocessing
from dingo.core.posterior_models import BasePosteriorModel
from dingo.core.transforms import RenameKey
from dingo.gw.conversion import change_spin_conversion_phase
from dingo.gw.domains import build_domain
from dingo.gw.transforms import (
    CopyToExtrinsicParameters,
    GetDetectorTimes,
    GNPEBase,
    GNPECoalescenceTimes,
    PostCorrectGeocentTime,
    SelectStandardizeRepackageParameters,
    TimeShiftStrain,
)

logger = logging.getLogger(__name__)


def _to_numpy(v) -> np.ndarray:
    """Detach a torch tensor (or coerce anything) to a numpy array."""
    if torch.is_tensor(v):
        return v.detach().cpu().numpy()
    return np.asarray(v)


def _stacked_phase_grid_terms(likelihood, theta, num_processes):
    """Compute `likelihood.phase_grid_terms` for each row of `theta`, in parallel, and
    stack the per-sample arrays along a leading axis for a single call to
    `log_likelihood_from_phase_grid_terms`."""
    terms_per_sample = apply_func_with_multiprocessing(
        likelihood.phase_grid_terms,
        theta,
        num_processes,
    )
    return {
        # The mode orders are the same for every sample.
        "m_vals": terms_per_sample[0]["m_vals"],
        "deltas": terms_per_sample[0]["deltas"],
        **{
            k: np.array([t[k] for t in terms_per_sample])
            for k in ("kappa2_modes", "rho2opt_const", "rho2opt_crossterms")
        },
    }


def _floored_density(log_density, uniform_weight):
    """Exponentiate log densities, rescaled to a maximum of 1 along the last axis, and
    add a constant floor of `uniform_weight` times the mean. The result is
    unnormalized and strictly positive, so importance weights stay finite."""
    density = np.exp(log_density - np.amax(log_density, axis=-1, keepdims=True))
    return density + density.mean(axis=-1, keepdims=True) * uniform_weight


def _phase_global_factor_mismatch(waveform_generator, theta, delta=0.9):
    """Mismatch of `h(theta, delta)` against `h(theta, 0) exp(2i delta)`: zero to
    round-off iff a phase shift multiplies the waveform by `exp(2i phase)`, which is
    what makes the (2, 2) synthetic phase exact. The inclination, and for precessing
    models the tilts, are set to a generic configuration, since at aligned spins the
    comparison cannot tell the spin conventions apart and would pass for any model
    with a single (2, |m| = 2) pair."""
    from dingo.gw.gwutils import get_mismatch

    theta = {k: float(v) for k, v in theta.items()}
    generic = {"theta_jn": 1.0}
    # In-plane spins only for models that take them; aligned ones reject them.
    if (
        LS.SimInspiralGetSpinSupportFromApproximant(waveform_generator.approximant)
        == LS.SIM_INSPIRAL_PRECESSINGSPIN
    ):
        generic.update({"tilt_1": 1.0, "tilt_2": 0.7, "phi_jl": 0.4})
    theta.update({k: v for k, v in generic.items() if k in theta})
    h0 = waveform_generator.generate_hplus_hcross({**theta, "phase": 0.0})
    h1 = waveform_generator.generate_hplus_hcross({**theta, "phase": delta})
    return max(
        get_mismatch(h1[pol], h0[pol] * np.exp(2j * delta), waveform_generator.domain)
        for pol in ("h_plus", "h_cross")
    )


def _check_cached_log_likelihood_is_exact(
    waveform_generator, approximation_22_mode, given
):
    """Raise unless the log likelihood a synthetic phase factor computes at the drawn
    angles equals a direct likelihood call: the phase dependence must be exact, as
    for the mode sum via the DFT decomposition, or the (2, 2) path for a model whose
    phase shift is a global exp(2i phase) factor. The model's declared property
    decides, a probe on the first row of `given` verifies it."""
    if not approximation_22_mode:
        if not waveform_generator.uses_dft_phase_decomposition:
            raise ValueError(
                "cache_log_likelihood requires the DFT phase decomposition, "
                f"which {waveform_generator.approximant_str} does not use here."
            )
    elif not waveform_generator.phase_is_global_factor:
        raise ValueError(
            "cache_log_likelihood with approximation_22_mode requires a phase "
            "shift to be a global exp(2i phase) factor, which "
            f"{waveform_generator.approximant_str} is not with "
            f"spin_conversion_phase = {waveform_generator.spin_conversion_phase}."
        )
    elif len(next(iter(given.values()))):
        mismatch = _phase_global_factor_mismatch(
            waveform_generator, {k: v[0] for k, v in given.items()}
        )
        # A NaN waveform fails the probe like any other disagreement.
        if not mismatch <= 1e-10:
            raise ValueError(
                f"{waveform_generator.approximant_str} is listed as having a "
                f"global exp(2i phase) factor, but a phase shift changes the "
                f"waveform by a mismatch of {mismatch:.1e}; refusing to cache "
                f"the (2, 2) log likelihood."
            )


class SyntheticPhaseFactor(Factor):
    """
    Reconstruct the coalescence phase for a phase-marginalized network: the factor
    `q(phase | theta_rest, d)`, built from the likelihood on a phase grid.

    For each incoming sample the factor evaluates `log L` on a grid over
    `[0, 2 pi]`. A single waveform evaluation per sample suffices, because the
    waveform modes computed at `phase = 0` each transform as `exp(-i m phase)`.
    The grid is exponentiated into a conditional phase distribution, a uniform
    floor (weight `uniform_weight`) keeps it positive everywhere so that
    importance sampling stays exact, and one phase is drawn per sample from the
    interpolated distribution. The returned log probability joins the chain's
    proposal density; importance sampling then targets the phase-full posterior.

    There are two grid modes. With `approximation_22_mode=True` the signal is
    assumed to be (2, 2)-dominated: the whole waveform transforms as
    `exp(2i phase)`, so the grid follows from the complex overlap
    `(d | h(phase=0))`. With `False` the modes are summed exactly, which requires
    the waveform generator's `spin_conversion_phase = 0`. The entry points differ
    on the default: this factor and `dingo_pipe`'s `PhaseRecoveryDefault` use the
    exact mode, while `Result.sample_proposal_extensions` defaults to the (2, 2)
    approximation when the key is omitted.

    With `cache_log_likelihood=True` the factor also emits the log
    likelihood at the drawn phase as the annotation column `log_likelihood_cache`, so
    that importance sampling need not evaluate the waveform again. The phase is drawn
    from the grid distribution *interpolated* between grid points, so the drawn phase
    generally lies between them and its likelihood is not one of the grid values.
    Rather, it is evaluated exactly at the drawn phase from the same mode inner
    products that produced the grid (a cheap sum over modes, no waveform call).
    Snapping the draws to the grid points instead would make the phase proposal
    discrete, inconsistent with the continuous interpolated density returned as the
    log probability, and interpolating the grid of log likelihoods would only be
    approximate. The value equals a direct likelihood call only if the m-components
    sum exactly to the direct waveform, as with the DFT phase decomposition;
    `Result._synthetic_phase_step` enables caching only then.
    """

    def __init__(
        self,
        conditioning: list[str],
        n_grid_phase: int = 5001,
        approximation_22_mode: bool = False,
        uniform_weight: float = 0.01,
        num_processes: int = 1,
        use_base_domain: bool = False,
        wfg_updates: Optional[dict] = None,
        cache_log_likelihood: bool = False,
    ):
        """
        Parameters
        ----------
        conditioning : list[str]
            The physical parameters the likelihood needs to generate the waveform
            (everything the chain has produced except `phase`).
        n_grid_phase : int, default 5001
            Number of phase grid points on `[0, 2 pi]`, endpoints included.
        approximation_22_mode : bool, default False
            Use the (2, 2)-mode approximation instead of the exact mode sum.
        uniform_weight : float, default 0.01
            Weight of the uniform floor added to the phase distribution for mass coverage.
        num_processes : int, default 1
            Parallel processes for the per-sample likelihood evaluation and phase sampling.
        use_base_domain : bool, default False
            For a multibanded model, evaluate the likelihood on the undecimated base
            domain (passed on to `SamplerContext.likelihood`).
        wfg_updates : dict, optional
            Overrides for the waveform generator settings stored with the network,
            e.g. `use_dft_phase_decomposition` (passed on to
            `SamplerContext.likelihood`).
        cache_log_likelihood : bool, default False
            Also emit the log likelihood at the drawn phase as the column
            `log_likelihood_cache`, for reuse by importance sampling. Requires a path
            whose phase dependence is exact: the exact mode sum with the DFT phase
            decomposition, or the (2, 2) path for a model whose phase shift is a
            global `exp(2i phase)` factor (`WaveformGenerator.phase_is_global_factor`,
            verified on the waveform before the first draw).
        """
        self.parameters = ["phase"]
        self.conditioning = list(conditioning)
        self.n_grid_phase = n_grid_phase
        self.approximation_22_mode = approximation_22_mode
        self.uniform_weight = uniform_weight
        self.num_processes = num_processes
        self.use_base_domain = use_base_domain
        self.wfg_updates = wfg_updates
        self.cache_log_likelihood = cache_log_likelihood
        self.annotations = ["log_likelihood_cache"] if cache_log_likelihood else []

    def sample_and_log_prob(self, num_samples, context, given=None):
        """Draw one phase per `theta_rest` row (`num_samples` must be 1); return the phases
        and their proposal log-prob `log q(phase | theta_rest, d)`."""
        if num_samples != 1:
            raise ValueError(
                "Synthetic phase is 1:1; draw one phase per sample (num_samples=1)."
            )
        reference = next(iter(given.values()))
        device = reference.device if torch.is_tensor(reference) else None
        n = len(reference)
        logger.info(f"Estimating synthetic phase for {n} samples.")
        t0 = time.time()
        if self.cache_log_likelihood:
            _check_cached_log_likelihood_is_exact(
                context.likelihood(
                    use_base_domain=self.use_base_domain, wfg_updates=self.wfg_updates
                ).waveform_generator,
                self.approximation_22_mode,
                given,
            )
        phases, phase_posterior, terms = self._phase_profile(given, context)
        new_phase, log_prob = interpolated_sample_and_log_prob_multi(
            phases, phase_posterior, self.num_processes
        )
        samples = {"phase": torch.as_tensor(new_phase, device=device)}
        if self.cache_log_likelihood:
            # Exact log likelihood at the drawn (off-grid) phase, see class docstring.
            likelihood = context.likelihood(
                use_base_domain=self.use_base_domain, wfg_updates=self.wfg_updates
            )
            if self.approximation_22_mode:
                log_likelihood = likelihood.log_likelihood_22_from_terms(
                    terms, new_phase[:, None]
                )[:, 0]
            else:
                log_likelihood = likelihood.log_likelihood_from_phase_grid_terms(
                    terms, new_phase[:, None], _to_numpy(given["psi"])[:, None]
                )[:, 0, 0]
            samples["log_likelihood_cache"] = torch.as_tensor(
                log_likelihood, device=device
            )
        logger.info(f"Done. This took {time.time() - t0:.2f} s.")
        return samples, torch.as_tensor(log_prob, device=device)

    def log_prob(self, theta_i, context, given=None):
        """Evaluate `log q(phase | theta_rest, d)` at the given phases (re-plug / IS)."""
        reference = next(iter(given.values()))
        device = reference.device if torch.is_tensor(reference) else None
        phases, phase_posterior, _ = self._phase_profile(given, context)
        log_prob = interpolated_log_prob_multi(
            phases, phase_posterior, _to_numpy(theta_i["phase"]), self.num_processes
        )
        return torch.as_tensor(log_prob, device=device)

    def describe(self) -> dict:
        """The default descriptor plus the phase-grid settings."""
        return {
            "step": type(self).__name__,
            "parameters": list(self.parameters),
            "conditioning": list(self.conditioning),
            "n_grid_phase": self.n_grid_phase,
            "approximation_22_mode": self.approximation_22_mode,
            "uniform_weight": self.uniform_weight,
            "use_base_domain": self.use_base_domain,
            "cache_log_likelihood": self.cache_log_likelihood,
        }

    def _phase_profile(self, given, context):
        """The phase grid and the mass-covered (un-normalized) phase distribution, one row
        per sample: evaluate `log L` on the grid, exponentiate (shifted by the per-row
        max), and add the uniform floor. Also returns the stacked mode terms of the
        likelihood in exact mode (else `None`)."""
        theta = pd.DataFrame({k: _to_numpy(v) for k, v in given.items()})
        likelihood = context.likelihood(
            use_base_domain=self.use_base_domain, wfg_updates=self.wfg_updates
        )
        phases = np.linspace(0, 2 * np.pi, self.n_grid_phase)
        if self.approximation_22_mode:
            # Assume a phase shift multiplies the waveform by exp(2i phase), so that
            # log L(phase) = log_Zn + Re[(d | h_0) exp(2i phase)] - (h_0 | h_0) / 2.
            # Exact for the models WaveformGenerator.phase_is_global_factor lists.
            terms_per_sample = apply_func_with_multiprocessing(
                likelihood.phase_grid_terms_22, theta, self.num_processes
            )
            terms = {
                k: np.array([t[k] for t in terms_per_sample])
                for k in ("d_inner_h", "h_inner_h")
            }
            phase_log_posterior = likelihood.log_likelihood_22_from_terms(terms, phases)
        else:
            # Exact: each mode m contributes exp(-i m phase); needs spin_conversion_phase=0.
            # One waveform evaluation per sample gives the mode terms, which are
            # stacked and evaluated on the grid for all samples at once.
            terms = _stacked_phase_grid_terms(likelihood, theta, self.num_processes)
            phase_log_posterior = likelihood.log_likelihood_from_phase_grid_terms(
                terms, phases, theta["psi"].to_numpy()[:, None]
            )[..., 0]
        return phases, _floored_density(phase_log_posterior, self.uniform_weight), terms


class SyntheticPhasePsiFactor(Factor):
    """
    Reconstruct the phase and polarization angle for a network trained with both
    marginalized: the factor `q(phase, psi | theta_rest, d)`, built from the
    likelihood on a (phase, psi) grid.

    One waveform evaluation per sample suffices. The m-components of the signal,
    generated at phase = 0, transform as `exp(-i m phase)`, and psi enters only
    through the antenna patterns, so the detector strain at any psi is

        mu(psi) = cos(2 psi) mu(0) + sin(2 psi) mu(pi / 4).

    The inner products of the two projections give `log L` at any (phase, psi). As
    in `SyntheticPhaseFactor`, there are two modes.

    With `approximation_22_mode=False` the modes are summed exactly, which requires
    the waveform generator's `spin_conversion_phase = 0`, see
    `StationaryGaussianGWLikelihood.phase_grid_terms`. The proposal is
    `q(phase) q(psi | phase)`. `q(phase)` is the likelihood on a (phase, psi) grid
    summed over psi, evaluated in chunks of rows sized by `max_grid_elements`.
    `q(psi | phase)` is the likelihood on the psi grid at the drawn phase itself,
    not at a phase grid point.

    With `approximation_22_mode=True` the whole waveform transforms as
    `exp(2i phase)`, see `StationaryGaussianGWLikelihood.phase_psi_terms_22`, so at
    fixed psi

        log L(phase, psi) = log_Zn + |z| cos(2 phase + arg z) - (h, h) / 2,

    with z = (d, h) at phase = 0 and both terms functions of psi. The phase then
    integrates out in closed form, which fixes the order the other way round: the
    proposal is `q(psi) q(phase | psi)`. `q(psi)` is the phase-marginal likelihood
    `exp(-(h, h) / 2) I0(|z|)` on the psi grid, without any phase grid, and
    `q(phase | psi)` is the exact conditional, a von Mises distribution in
    `2 phase`, drawn directly. `n_grid_phase` is not used.

    Each density is exponentiated and given a uniform floor of weight
    `uniform_weight`. The grid densities are sampled as piecewise-constant
    densities; the von Mises conditional is mixed with the uniform density in the
    same proportion, `uniform_weight / (1 + uniform_weight)`.

    With `cache_log_likelihood=True` the factor also emits `log L` at the drawn
    (phase, psi), evaluated exactly from the same inner products, as the
    annotation column `log_likelihood_cache`. As in `SyntheticPhaseFactor`, this
    requires an exact phase dependence, which is checked before the first draw.
    """

    # Upper bound on the (chunk, n_grid_phase, n_grid_psi) grid held at once, in elements.
    max_grid_elements = 8_000_000

    def __init__(
        self,
        conditioning: list[str],
        n_grid_phase: int = 512,
        n_grid_psi: int = 128,
        approximation_22_mode: bool = False,
        uniform_weight: float = 0.01,
        num_processes: int = 1,
        use_base_domain: bool = False,
        wfg_updates: Optional[dict] = None,
        cache_log_likelihood: bool = False,
    ):
        """
        Parameters
        ----------
        conditioning : list[str]
            The physical parameters the likelihood needs to generate the waveform
            (everything the chain has produced except `phase` and `psi`).
        n_grid_phase : int, default 512
            Number of phase grid points on `[0, 2 pi]`, endpoints included. Not
            used with `approximation_22_mode`, where the phase is drawn exactly.
        n_grid_psi : int, default 128
            Number of psi grid points on `[0, pi]`, endpoints included.
        approximation_22_mode : bool, default False
            Use the (2, 2)-mode approximation instead of the exact mode sum.
        uniform_weight : float, default 0.01
            Weight of the uniform floor added to each of the two densities.
        num_processes : int, default 1
            Parallel processes for the per-sample waveform evaluations and the
            interpolated draws.
        use_base_domain : bool, default False
            For a multibanded model, evaluate the likelihood on the undecimated base
            domain (passed on to `SamplerContext.likelihood`).
        wfg_updates : dict, optional
            Overrides for the waveform generator settings stored with the network,
            e.g. `use_dft_phase_decomposition` (passed on to
            `SamplerContext.likelihood`).
        cache_log_likelihood : bool, default False
            Also emit the log likelihood at the drawn (phase, psi) as the column
            `log_likelihood_cache`, for reuse by importance sampling. Requires an
            exact phase dependence, as in `SyntheticPhaseFactor`.
        """
        self.parameters = ["phase", "psi"]
        self.conditioning = list(conditioning)
        self.n_grid_phase = n_grid_phase
        self.n_grid_psi = n_grid_psi
        self.approximation_22_mode = approximation_22_mode
        self.uniform_weight = uniform_weight
        self.num_processes = num_processes
        self.use_base_domain = use_base_domain
        self.wfg_updates = wfg_updates
        self.cache_log_likelihood = cache_log_likelihood
        self.annotations = ["log_likelihood_cache"] if cache_log_likelihood else []

    def sample_and_log_prob(self, num_samples, context, given=None):
        """Draw one (phase, psi) per row of `given` (`num_samples` must be 1): phase
        from `q(phase)`, then psi from `q(psi | phase)` at the drawn phase, or with
        `approximation_22_mode` psi from `q(psi)`, then the phase from
        `q(phase | psi)`. The log probability is that of the pair. See
        `Factor.sample_and_log_prob`."""
        if num_samples != 1:
            raise ValueError(
                "Synthetic phase and psi are 1:1; draw one pair per sample "
                "(num_samples=1)."
            )
        reference = next(iter(given.values()))
        device = reference.device if torch.is_tensor(reference) else None
        logger.info(f"Estimating synthetic phase and psi for {len(reference)} samples.")
        t0 = time.time()
        if self.cache_log_likelihood:
            _check_cached_log_likelihood_is_exact(
                context.likelihood(
                    use_base_domain=self.use_base_domain, wfg_updates=self.wfg_updates
                ).waveform_generator,
                self.approximation_22_mode,
                given,
            )
        if self.approximation_22_mode:
            likelihood, terms, psis, psi_posterior = self._psi_profile_22(
                given, context
            )
            new_psi, log_prob_psi = interpolated_sample_and_log_prob_multi(
                psis, psi_posterior, self.num_processes
            )
            terms_at_psi = {
                k: v[:, 0]
                for k, v in likelihood.terms_22_at_psi(terms, new_psi[:, None]).items()
            }
            new_phase = self._sample_phase_22(terms_at_psi)
            log_prob_phase = self._log_prob_phase_22(terms_at_psi, new_phase)
        else:
            likelihood, terms, phases, phase_posterior = self._phase_profile(
                given, context
            )
            new_phase, log_prob_phase = interpolated_sample_and_log_prob_multi(
                phases, phase_posterior, self.num_processes
            )
            psis, psi_posterior = self._psi_profile(likelihood, terms, new_phase)
            new_psi, log_prob_psi = interpolated_sample_and_log_prob_multi(
                psis, psi_posterior, self.num_processes
            )
        samples = {
            "phase": torch.as_tensor(new_phase, device=device),
            "psi": torch.as_tensor(new_psi, device=device),
        }
        if self.cache_log_likelihood:
            # Exact log likelihood at the drawn (off-grid) point, from the same terms.
            if self.approximation_22_mode:
                log_likelihood = likelihood.log_likelihood_22_from_terms(
                    terms_at_psi, new_phase[:, None]
                )[:, 0]
            else:
                log_likelihood = likelihood.log_likelihood_from_phase_grid_terms(
                    terms, new_phase[:, None], new_psi[:, None]
                )[:, 0, 0]
            samples["log_likelihood_cache"] = torch.as_tensor(
                log_likelihood, device=device
            )
        logger.info(f"Done. This took {time.time() - t0:.2f} s.")
        return samples, torch.as_tensor(log_prob_phase + log_prob_psi, device=device)

    def log_prob(self, theta_i, context, given=None):
        """Evaluate the log probability of the given angles, rebuilding both
        distributions from `given`. See `Factor.log_prob`."""
        reference = next(iter(given.values()))
        device = reference.device if torch.is_tensor(reference) else None
        phase, psi = _to_numpy(theta_i["phase"]), _to_numpy(theta_i["psi"])
        if self.approximation_22_mode:
            likelihood, terms, psis, psi_posterior = self._psi_profile_22(
                given, context
            )
            log_prob_psi = interpolated_log_prob_multi(
                psis, psi_posterior, psi, self.num_processes
            )
            terms_at_psi = {
                k: v[:, 0]
                for k, v in likelihood.terms_22_at_psi(terms, psi[:, None]).items()
            }
            log_prob_phase = self._log_prob_phase_22(terms_at_psi, phase)
        else:
            likelihood, terms, phases, phase_posterior = self._phase_profile(
                given, context
            )
            log_prob_phase = interpolated_log_prob_multi(
                phases, phase_posterior, phase, self.num_processes
            )
            psis, psi_posterior = self._psi_profile(likelihood, terms, phase)
            log_prob_psi = interpolated_log_prob_multi(
                psis, psi_posterior, psi, self.num_processes
            )
        return torch.as_tensor(log_prob_phase + log_prob_psi, device=device)

    def describe(self) -> dict:
        """The default descriptor plus the grid settings."""
        return {
            "step": type(self).__name__,
            "parameters": list(self.parameters),
            "conditioning": list(self.conditioning),
            "n_grid_phase": self.n_grid_phase,
            "n_grid_psi": self.n_grid_psi,
            "approximation_22_mode": self.approximation_22_mode,
            "uniform_weight": self.uniform_weight,
            "use_base_domain": self.use_base_domain,
            "cache_log_likelihood": self.cache_log_likelihood,
        }

    def _phase_profile(self, given, context):
        """Return the likelihood, the stacked phase grid terms of all rows of `given`,
        the phase grid, and `q(phase)` on it: the likelihood on the (phase, psi) grid
        summed over psi, then floored; unnormalized, shape (N, n_grid_phase)."""
        theta = pd.DataFrame({k: _to_numpy(v) for k, v in given.items()})
        likelihood = context.likelihood(
            use_base_domain=self.use_base_domain, wfg_updates=self.wfg_updates
        )
        terms = _stacked_phase_grid_terms(likelihood, theta, self.num_processes)
        phases = np.linspace(0, 2 * np.pi, self.n_grid_phase)
        psis = np.linspace(0, np.pi, self.n_grid_psi)
        n = len(theta)
        chunk = max(1, self.max_grid_elements // (self.n_grid_phase * self.n_grid_psi))
        log_marginal = np.empty((n, self.n_grid_phase))
        for start in range(0, n, chunk):
            sl = slice(start, start + chunk)
            grid = likelihood.log_likelihood_from_phase_grid_terms(
                {
                    **terms,
                    **{
                        k: terms[k][sl]
                        for k in ("kappa2_modes", "rho2opt_const", "rho2opt_crossterms")
                    },
                },
                phases,
                psis,
            )
            # Drop the endpoint psi = pi: psi has period pi, so it repeats the
            # phase-dependent psi = 0 column and would double-count it.
            # A plain numpy logsumexp: scipy's is 2-4x slower on this strided grid
            # when the grid size is small.
            grid = grid[..., :-1]
            peak = grid.max(axis=-1)
            log_marginal[sl] = peak + np.log(
                np.exp(grid - peak[..., None]).sum(axis=-1)
            )
        return (
            likelihood,
            terms,
            phases,
            _floored_density(log_marginal, self.uniform_weight),
        )

    def _psi_profile(self, likelihood, terms, phase):
        """Return the psi grid and `q(psi | phase)` on it: the likelihood on the psi
        grid at each row's `phase` (off the phase grid), floored; unnormalized, shape
        (N, n_grid_psi)."""
        psis = np.linspace(0, np.pi, self.n_grid_psi)
        log_conditional = likelihood.log_likelihood_from_phase_grid_terms(
            terms, np.asarray(phase)[:, None], psis
        )[:, 0, :]
        return psis, _floored_density(log_conditional, self.uniform_weight)

    def _psi_profile_22(self, given, context):
        """Return the likelihood, the stacked (2, 2) terms of all rows of `given`, the
        psi grid, and `q(psi)` on it: the likelihood integrated over the phase,
        `exp(log_Zn - (h, h) / 2) I0(|(d, h)|)` up to a constant, floored;
        unnormalized, shape (N, n_grid_psi)."""
        theta = pd.DataFrame({k: _to_numpy(v) for k, v in given.items()})
        likelihood = context.likelihood(
            use_base_domain=self.use_base_domain, wfg_updates=self.wfg_updates
        )
        terms_per_sample = apply_func_with_multiprocessing(
            likelihood.phase_psi_terms_22, theta, self.num_processes
        )
        terms = {
            k: np.array([t[k] for t in terms_per_sample])
            for k in ("d_inner_h", "h_inner_h")
        }
        psis = np.linspace(0, np.pi, self.n_grid_psi)
        terms_on_grid = likelihood.terms_22_at_psi(terms, psis)
        log_marginal = (
            ln_i0(np.abs(terms_on_grid["d_inner_h"])) - terms_on_grid["h_inner_h"] / 2
        )
        return (
            likelihood,
            terms,
            psis,
            _floored_density(log_marginal, self.uniform_weight),
        )

    def _sample_phase_22(self, terms_at_psi):
        """Draw the phase from `q(phase | psi)` given the (2, 2) terms at the drawn
        psi, one phase per row: the von Mises conditional, with probability
        `uniform_weight / (1 + uniform_weight)` replaced by a uniform draw (the
        floor). Uses bilby's random generator, like the interpolated draws."""
        rng = bilby.core.utils.random.rng
        z = terms_at_psi["d_inner_h"]
        n = len(z)
        # Re[z exp(2i phase)] = |z| cos(2 phase + arg z), so 2 phase is von Mises
        # about -arg z, and phase is either of its two halves in [0, 2 pi).
        two_phase = rng.vonmises(-np.angle(z), np.abs(z))
        phase = np.mod(two_phase / 2 + np.pi * rng.integers(2, size=n), 2 * np.pi)
        floor = rng.uniform(size=n) < self.uniform_weight / (1 + self.uniform_weight)
        return np.where(floor, rng.uniform(0, 2 * np.pi, size=n), phase)

    def _log_prob_phase_22(self, terms_at_psi, phase):
        """`log q(phase | psi)`, the density `_sample_phase_22` draws from: the von
        Mises density of `2 phase` with the uniform floor, normalized on [0, 2 pi)."""
        z = terms_at_psi["d_inner_h"]
        von_mises = np.exp((z * np.exp(2j * phase)).real - ln_i0(np.abs(z)))
        return np.log(
            (von_mises + self.uniform_weight) / (2 * np.pi * (1 + self.uniform_weight))
        )


def _build_gnpe_transforms(model: BasePosteriorModel):
    """Build the time-shift GNPE per-step transforms from a model's metadata: the
    proxy blur, the per-row time-shift alignment applied before the network, and the
    post-network geocent-time correction.

    Returns
    -------
    transform_pre, transform_post : Compose
    gnpe_transform : GNPECoalescenceTimes
        The blur transform; it carries the kernel and the detector-time parameter
        names (`input_parameter_names`).
    """
    meta = model.base_metadata
    data_settings = meta["train_settings"]["data"]
    ifo_list = InterferometerList(data_settings["detectors"])
    domain = build_domain(meta["dataset_settings"]["domain"])
    if "domain_update" in data_settings:
        domain.update(data_settings["domain_update"])

    gnpe_time_settings = data_settings.get("gnpe_time_shifts")
    if not gnpe_time_settings:
        raise NotImplementedError(
            "Only time-shift GNPE (gnpe_time_shifts) is supported here."
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
    return Compose(transform_pre), Compose(transform_post), gnpe_transform


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

    def __init__(self, model: BasePosteriorModel):
        """
        Parameters
        ----------
        model : BasePosteriorModel
            The GNPE main network; its metadata defines the kernel and the
            detector-time parameters.
        """
        _, _, gnpe_transform = _build_gnpe_transforms(model)
        self.gnpe = gnpe_transform
        self.gnpe_parameters = list(gnpe_transform.input_parameter_names)
        self.parameters = [p + "_proxy" for p in self.gnpe_parameters]
        self.conditioning = list(self.gnpe_parameters)
        self.kernel = gnpe_transform.kernel

    def sample_and_log_prob(self, num_samples, context, given=None):
        """Blur the conditioning detector times into proxies; `num_samples` must be 1
        (GNPE is 1:1). Returns the proxies and their kernel log-prob."""
        if num_samples != 1:
            raise ValueError("GNPE proxy is 1:1; num_samples must be 1.")
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
    them. Draws `num_samples` per proxy row (one in the Gibbs loop).
    """

    def __init__(
        self, model: BasePosteriorModel, aliases: Optional[dict[str, str]] = None
    ):
        """
        Parameters
        ----------
        model : BasePosteriorModel
            The GNPE main network; the per-iteration transforms are built from its
            metadata.
        aliases : dict[str, str], optional
            Trained-name to exposed-name map (e.g. `{"ra": "ra@t_ref"}`).
        """
        if model.metadata["train_settings"]["data"].get("tokenization"):
            raise NotImplementedError(
                "GNPE with a tokenized (transformer) network is not supported."
            )
        self.model = model
        self.transform_pre, self.transform_post, gnpe_transform = (
            _build_gnpe_transforms(model)
        )
        self.gnpe_parameters = list(gnpe_transform.input_parameter_names)
        self.proxy_parameters = [p + "_proxy" for p in self.gnpe_parameters]
        self.aliases = aliases or {}
        self._net_parameters = model.base_metadata["train_settings"]["data"][
            "inference_parameters"
        ]
        self.parameters = [self.aliases.get(p, p) for p in self._net_parameters]
        self.conditioning = list(self.proxy_parameters)
        # For log_prob: the sampling path de-standardizes (and corrects the
        # log-prob) inside transform_post, but evaluating at a point needs the
        # forward map too. The model's own standardization (network-bound, like
        # FlowFactor's).
        std = model.metadata["train_settings"]["data"]["standardization"]
        self.standardization = Standardization(std["mean"], std["std"])

    @property
    def produces(self) -> list[str]:
        """Emitted columns: the inference block plus the recomputed detector times."""
        return super().produces + self.gnpe_parameters

    def sample_and_log_prob(self, num_samples, context, given=None):
        """Draw `num_samples` parameter sets per proxy row (the draws for a row are
        adjacent). Returns theta plus the recomputed detector times, and the network
        log-prob. In the Gibbs loop and with a drawing proxy source this is called
        with `num_samples=1`; a value greater than 1 arises only when the proxy
        source is pinned. See `Factor.sample_and_log_prob`."""
        proxies = {p: given[p] for p in self.proxy_parameters}
        n_rows = _n_rows(proxies)
        x = {"extrinsic_parameters": dict(proxies), "parameters": {}}
        d = context.prepared_data().clone()
        x["data"] = d.expand(n_rows, *d.shape)
        x = self.transform_pre(x)
        self.model.network.eval()
        with torch.no_grad():
            if "context_parameters" in x:
                y, log_prob = self.model.sample_and_log_prob(
                    x["data"], x["context_parameters"], num_samples=num_samples
                )
            else:
                y, log_prob = self.model.sample_and_log_prob(
                    x["data"], num_samples=num_samples
                )
        # The network returns (n_rows, num_samples, dim), with n_rows the number of
        # conditioning rows (one per proxy sample). Flatten so that each row is one
        # posterior sample, with the draws for a proxy row adjacent, and repeat the
        # per-row extrinsic parameters (proxies, the preferred-proxy geocent time)
        # to match before the post-network corrections.
        x["parameters"] = y.reshape(n_rows * num_samples, y.shape[-1])
        x["log_prob"] = log_prob.reshape(n_rows * num_samples)
        x["extrinsic_parameters"] = {
            k: v.repeat_interleave(num_samples, 0)
            for k, v in x["extrinsic_parameters"].items()
        }
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
        Applies the same per-row view of the data as sampling (the shared
        representation time-shifted by the proxies, the conditioning standardized),
        then scores the standardized parameters under the network."""
        proxies = {p: given[p] for p in self.proxy_parameters}
        n_rows = _n_rows(proxies)
        x = {"extrinsic_parameters": dict(proxies), "parameters": {}}
        d = context.prepared_data().clone()
        x["data"] = d.expand(n_rows, *d.shape)
        x = self.transform_pre(x)
        theta_net = {
            net: theta_i[self.aliases.get(net, net)] for net in self._net_parameters
        }
        # Mirror transform_post: sampling subtracts the preferred-proxy geocent time
        # after the network (PostCorrectGeocentTime), so score the network in its own
        # output frame by applying the inverse correction first.
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


class RAToEventFrame(Reparametrization):
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
        self.inputs = ["ra@t_ref"]
        self.parameters = ["ra"]
        self.conditioning = []

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

    def inverse(self, params, context, given=None):
        correction = self._correction(context)
        if correction == 0.0:
            return {"ra@t_ref": params["ra"]}
        ra_tref = (params["ra"].double() - correction) % (2 * np.pi)
        return {"ra@t_ref": ra_tref.float()}


class RAToTrainingFrame(RAToEventFrame):
    """
    Rotate a pinned event-frame right ascension (`ra`) into the network's training
    frame (`ra@t_ref`): the input-side mirror of `RAToEventFrame`.

    A sky position pinned at the event time must be presented to the network in
    the frame it was trained in. A trailing `RAToEventFrame` then restores the
    event-frame value in the samples.
    """

    def __init__(self):
        self.inputs = ["ra"]
        self.parameters = ["ra@t_ref"]
        self.conditioning = []

    def forward(self, given, context):
        return super().inverse({"ra": given["ra"]}, context)

    def inverse(self, params, context, given=None):
        return super().forward({"ra@t_ref": params["ra@t_ref"]}, context)


class SpinConventionReparam(Reparametrization):
    """
    Relabel the precessing-spin angles between Dingo's internal spin convention
    and the physical (Bilby) one.

    Dingo fixes the spin-conversion phase (usually to 0) so that the Cartesian
    spins decouple from the coalescence phase. Sampling, likelihood, and
    synthetic phase all work in that convention, and stored samples keep the
    plain names `theta_jn` / `phi_jl` in it. The physical convention (spin
    conversion at the sample's own phase) is what Bilby and PESummary mean by the
    same names, so the relabel happens when samples are exported. Only
    `theta_jn` and `phi_jl` change; the conversion phase and reference frequency
    are read from the model metadata, and a model trained without a fixed
    conversion phase relabels to the identity.

    Exporting a finished weighted sample set needs no Jacobian, since proposal,
    prior, and likelihood transform together; that is `to_physical`. As a chain
    step the map is not measure-preserving in the flat `(theta_jn, phi_jl)`
    coordinates: it rotates the line of sight rigidly about the orbital angular
    momentum, preserving the spherical measure `sin(theta_jn) dtheta dphi`, so
    `log_det = log sin(theta_jn) - log sin(theta_jn')`.
    """

    def __init__(self, num_processes: int = 1):
        """
        Parameters
        ----------
        num_processes : int, default 1
            Parallel processes for the per-sample LAL spin conversion.
        """
        self.parameters = ["theta_jn", "phi_jl"]
        # The bijection overwrites theta_jn / phi_jl in place; the remaining
        # columns (phase, masses, tilts, ...) are read-only conditioning.
        self.inputs = ["theta_jn", "phi_jl"]
        self.conditioning = [
            "phase",
            "chirp_mass",
            "mass_ratio",
            "a_1",
            "a_2",
            "tilt_1",
            "tilt_2",
            "phi_12",
        ]
        self.num_processes = num_processes

    def log_det(self, given, context):
        """`log|det J|` of `forward`, per row. The map preserves the spherical
        measure, so the flat-coordinate Jacobian is
        `sin(theta_jn) / sin(theta_jn')` -- verified numerically against finite
        differences through the LAL conversion in
        `test_jacobian_matches_sin_ratio` (agreement ~1e-9)."""
        converted = self.forward(given, context)
        return self._log_det(given["theta_jn"], converted["theta_jn"])

    @staticmethod
    def _log_det(theta_jn_in, theta_jn_out):
        # Compute in double (the LAL conversion is double precision anyway),
        # return in the input dtype: a reparametrization preserves the chain's
        # dtype rather than promoting the summed log_prob.
        log_det = torch.log(torch.sin(theta_jn_in.double())) - torch.log(
            torch.sin(theta_jn_out.double())
        )
        return log_det.to(theta_jn_in.dtype)

    def sample_and_log_prob(self, num_samples, context, given=None):
        """Apply `forward`; contribute `-log|det J|`. Overridden to share the
        single LAL conversion between the transform and its Jacobian (the base
        implementation would convert twice)."""
        if num_samples != 1:
            raise ValueError("A reparametrization is 1:1; num_samples must be 1.")
        out = self.forward(given, context)
        return out, -self._log_det(given["theta_jn"], out["theta_jn"])

    @staticmethod
    def _model_convention(model_metadata: dict) -> tuple[float, Optional[float]]:
        """The reference frequency and spin-conversion phase the model trained with."""
        wfg_settings = model_metadata["dataset_settings"]["waveform_generator"]
        return wfg_settings["f_ref"], wfg_settings.get("spin_conversion_phase")

    def to_physical(self, samples: pd.DataFrame, model_metadata: dict) -> pd.DataFrame:
        """Relabel samples from the model's convention to the physical (Bilby) one."""
        f_ref, sc_phase = self._model_convention(model_metadata)
        return change_spin_conversion_phase(
            samples, f_ref, sc_phase, None, num_processes=self.num_processes
        )

    def to_network(self, samples: pd.DataFrame, model_metadata: dict) -> pd.DataFrame:
        """Relabel samples from the physical (Bilby) convention to the model's,
        e.g. to ingest external posteriors for comparison."""
        f_ref, sc_phase = self._model_convention(model_metadata)
        return change_spin_conversion_phase(
            samples, f_ref, None, sc_phase, num_processes=self.num_processes
        )

    def forward(self, given, context):
        # The conversion runs in double; the outputs return in the input dtype
        # and device.
        reference = given["theta_jn"]
        theta = pd.DataFrame({k: _to_numpy(v) for k, v in given.items()})
        converted = self.to_physical(theta, context.model_metadata)
        return {
            k: torch.as_tensor(converted[k].to_numpy()).to(
                dtype=reference.dtype, device=reference.device
            )
            for k in self.parameters
        }

    def inverse(self, params, context, given=None):
        # The physical -> network direction also needs the invariant
        # conditioning (phase, masses, tilts), which the reverse fold supplies
        # as `given`.
        if given is None:
            raise ValueError(
                "The spin-convention inverse needs the conditioning block "
                "(phase, masses, tilts); pass it as `given`, or convert "
                "DataFrames with to_network()."
            )
        reference = params["theta_jn"]
        rows = {**given, **params}
        theta = pd.DataFrame({k: _to_numpy(v) for k, v in rows.items()})
        converted = self.to_network(theta, context.model_metadata)
        return {
            k: torch.as_tensor(converted[k].to_numpy()).to(
                dtype=reference.dtype, device=reference.device
            )
            for k in self.parameters
        }


class GNPEKernelCorrection(TargetCorrection):
    """
    The single-step GNPE kernel correction, as a target-side chain step.

    Single-step GNPE samples from the joint proposal `q(theta, theta_hat | d)`
    over parameters and proxies, so the matching importance-sampling target
    acquires the kernel term `p(theta_hat | theta)`. This step evaluates that
    term at the proxies and at the detector times the main network recomputed
    from theta, and emits it as the `delta_log_prob_target` column. It
    contributes zero to the proposal density. The recomputed detector times it
    reads are a side channel of the main network, not part of the output.
    """

    def __init__(self, kernel_factor: GNPEKernelFactor):
        """
        Parameters
        ----------
        kernel_factor : GNPEKernelFactor
            The kernel whose density is evaluated; also names the proxy and
            detector-time columns.
        """
        self.kernel_factor = kernel_factor
        self.produces = ["delta_log_prob_target"]
        self.conditioning = list(kernel_factor.parameters) + list(
            kernel_factor.gnpe_parameters
        )

    def correction(self, given, context):
        proxies = {p: given[p] for p in self.kernel_factor.parameters}  # theta_hat
        gnpe_params = {k: given[k] for k in self.kernel_factor.gnpe_parameters}  # theta
        # log p(theta_hat | theta)
        correction = self.kernel_factor.log_prob(proxies, context, gnpe_params)
        return {"delta_log_prob_target": correction}
