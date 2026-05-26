"""
Phase- and psi-marginalized GW likelihood, plus synthetic-(phase, psi) sampling.

This is a standalone extension of dingo's phase-only machinery
(``StationaryGaussianGWLikelihood._log_likelihood_phase_grid_mode_decomposed`` and
``Result.sample_synthetic_phase``) to *jointly* marginalize the binary phase ``phi``
and the polarization angle ``psi``. It does not modify dingo: it operates on the
public components of an already-built ``StationaryGaussianGWLikelihood`` /
``Result``.

Why this is cheap
-----------------
For one parameter sample, both ``phi`` and ``psi`` enter the detector signal
through closed-form factors, so the expensive waveform generation is done *once*:

    mu_d(phi, psi) = sum_m exp(-i m phi) [cos(2 psi) A_m^d + sin(2 psi) B_m^d]

where ``A_m^d`` / ``B_m^d`` are the fully-projected, whitened per-mode strains at
the two reference polarizations ``psi = 0`` and ``psi = pi/4`` (the LAL call is
shared; only the antenna-pattern projection differs between them). This holds
because ``phi`` rotates each spherical-harmonic mode by ``exp(-i m phi)``, and
``psi`` enters only through the antenna patterns ``F+, Fx`` while every remaining
projection step (distance rescaling, time shifting, calibration, whitening) is
linear in the strain, hence
``h_d(psi) = cos(2 psi) h_d(0) + sin(2 psi) h_d(pi/4)``.

The whitened log-likelihood ``log L = log Zn + kappa2 - 1/2 rho2opt`` then reduces
to precomputed scalars times analytic ``phi``/``psi`` factors:

    kappa2(phi, psi)  = cos(2 psi) Re[sum_m alpha_m e^{-i m phi}]
                      + sin(2 psi) Re[sum_m beta_m  e^{-i m phi}]
    rho2opt(phi, psi) = cos^2(2 psi) S_AA(phi)
                      + cos(2 psi) sin(2 psi) S_AB+BA(phi)
                      + sin^2(2 psi) S_BB(phi)

with ``alpha_m = sum_d <d, A_m>``, ``beta_m = sum_d <d, B_m>`` and the per-mode-pair
cross terms ``C_XY[m, n] = sum_d <X_m, Y_n>`` summed over the phase grid. The
``(phi, psi)`` grid is then pure broadcasting -- no loops over grid points, no
extra waveform calls.

Assumptions
-----------
* ``waveform_generator.spin_conversion_phase == 0`` (so phase is purely extrinsic
  and modes rotate as ``exp(-i m phi)``); enforced below.
* Uniform priors ``phi ~ U[0, 2 pi)`` and ``psi ~ U[0, pi)`` for marginalization.
  ``psi`` has period ``pi`` (the antenna patterns depend on ``2 psi``), so the
  grid spans ``[0, pi)``.
* The base likelihood is *not* itself marginalized (it must expose the plain
  per-mode signal); build it with no marginalization kwargs.
"""

from functools import partial
from multiprocessing import Pool

import numpy as np
from bilby.core.prior import Constraint, Interped, Uniform
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from dingo.core.likelihood import Likelihood
from dingo.gw.likelihood import inner_product_complex
from dingo.gw.prior import split_off_extrinsic_parameters

# Reference polarizations whose projections span the full psi dependence.
PSI_BASIS = (0.0, np.pi / 4)


# ---------------------------------------------------------------------------
# Per-mode signal at the two reference polarizations (one waveform call)
# ---------------------------------------------------------------------------
def signal_m_psi_basis(likelihood, theta, psi_basis=PSI_BASIS):
    """
    Per-mode detector strains projected at each polarization angle in ``psi_basis``,
    generating the waveform only once.

    Mirrors ``GWSignal.signal_m`` but, instead of projecting at a single ``psi``,
    reuses the cached modes to project at every ``psi`` in ``psi_basis``. Each
    returned per-mode contribution transforms as ``exp(-i m phi)`` under a phase
    shift; the strain at any ``psi`` is
    ``cos(2 psi) * out[0] + sin(2 psi) * out[1]`` for the default basis.

    Parameters
    ----------
    likelihood : StationaryGaussianGWLikelihood
        Built likelihood (a ``GWSignal``), supplying the waveform generator, data
        domain, ASDs and projection transforms.
    theta : dict
        Signal parameters. Any ``psi`` entry is ignored/overridden; ``phase``
        should be (and is here treated as) the reference 0.
    psi_basis : sequence of float
        Polarization angles at which to project the cached modes.

    Returns
    -------
    list of dict
        One ``{m: {ifo: strain}}`` dict per entry of ``psi_basis`` (same order).
    """
    theta_intrinsic, theta_extrinsic = split_off_extrinsic_parameters(theta)
    theta_intrinsic = {k: float(v) for k, v in theta_intrinsic.items()}

    # Single (expensive) waveform call: the m-decomposed polarizations at phase 0.
    pol_m = likelihood.waveform_generator.generate_hplus_hcross_m(theta_intrinsic)
    pol_m = {  # truncate to data domain, as in GWSignal.signal_m
        k_m: {
            k_pol: likelihood.data_domain.update_data(v_pol)
            for k_pol, v_pol in v_m.items()
        }
        for k_m, v_m in pol_m.items()
    }

    out = []
    for psi in psi_basis:
        extrinsic = {**theta_extrinsic, "psi": psi}
        strain_m = {}
        for m, pol in pol_m.items():
            sample = {
                "parameters": theta_intrinsic,
                "extrinsic_parameters": extrinsic,
                "waveform": pol,
            }
            if likelihood.asd is not None:
                sample["asds"] = likelihood.asd
            strain_m[m] = likelihood.projection_transforms(sample)["waveform"]
        out.append(strain_m)
    return out


# ---------------------------------------------------------------------------
# Phase-psi grid likelihood (the core computation)
# ---------------------------------------------------------------------------
def phase_psi_grid_log_likelihood(likelihood, theta, phases, psis):
    """
    Log-likelihood on the full outer ``(phases, psis)`` grid for one sample.

    Generates the waveform once, projects at ``psi = 0`` and ``psi = pi/4``,
    precomputes the per-mode inner products, and evaluates ``log L`` analytically
    across the grid.

    Parameters
    ----------
    likelihood : StationaryGaussianGWLikelihood
    theta : dict
        Signal parameters (``phase`` and ``psi`` are set/swept internally).
    phases : np.ndarray, shape (n_phase,)
    psis : np.ndarray, shape (n_psi,)

    Returns
    -------
    np.ndarray, shape (n_phase, n_psi)
        ``log L`` at every grid point.
    """
    if likelihood.waveform_generator.spin_conversion_phase != 0:
        raise ValueError(
            "Phase-psi grid likelihood assumes WaveformGenerator."
            f"spin_conversion_phase = 0, got "
            f"{likelihood.waveform_generator.spin_conversion_phase}."
        )

    # Project the cached modes at the two reference polarizations (one wf call).
    A, B = signal_m_psi_basis(likelihood, {**theta, "phase": 0.0})

    return phase_psi_grid_from_per_mode_strains(
        A,
        B,
        likelihood.whitened_strains,
        likelihood.log_Zn,
        likelihood.data_domain.min_idx,
        phases,
        psis,
    )


def phase_psi_grid_from_per_mode_strains(
    A, B, whitened_strains, log_Zn, min_idx, phases, psis
):
    """
    Core analytic ``(phase, psi)`` grid likelihood from precomputed per-mode strains.

    Separated from waveform generation so the algebra can be unit-tested directly.

    Parameters
    ----------
    A, B : dict
        ``{m: {ifo: strain}}`` per-mode whitened detector strains at ``psi = 0``
        and ``psi = pi/4`` respectively (both at ``phase = 0``).
    whitened_strains : dict
        ``{ifo: strain}`` whitened data.
    log_Zn : float
        Noise log-evidence ``-1/2 <d, d>``.
    min_idx : int
        Lowest frequency bin in the likelihood integral.
    phases, psis : np.ndarray

    Returns
    -------
    np.ndarray, shape (n_phase, n_psi)
    """
    d = whitened_strains
    ifos = list(d.keys())
    m_vals = sorted(A.keys())

    # kappa2 = Re <d, mu>. Per mode: alpha_m = sum_d <d, A_m>, beta_m = sum_d <d, B_m>.
    alpha = {
        m: sum(inner_product_complex(d[i], A[m][i], min_idx) for i in ifos)
        for m in m_vals
    }
    beta = {
        m: sum(inner_product_complex(d[i], B[m][i], min_idx) for i in ifos)
        for m in m_vals
    }

    # rho2opt = <mu, mu>. Per mode pair: C_AA, C_BB, and (C_AB + C_BA).
    # All three matrices are Hermitian in (m, n), so the phase sums below are real.
    c_aa, c_bb, c_ab_ba = {}, {}, {}
    for m in m_vals:
        for n in m_vals:
            c_aa[(m, n)] = sum(
                inner_product_complex(A[m][i], A[n][i], min_idx) for i in ifos
            )
            c_bb[(m, n)] = sum(
                inner_product_complex(B[m][i], B[n][i], min_idx) for i in ifos
            )
            c_ab_ba[(m, n)] = sum(
                inner_product_complex(A[m][i], B[n][i], min_idx)
                + inner_product_complex(B[m][i], A[n][i], min_idx)
                for i in ifos
            )

    phases = np.asarray(phases)
    psis = np.asarray(psis)
    cos2psi = np.cos(2 * psis)
    sin2psi = np.sin(2 * psis)

    # Phase-dependent reductions (1D over phases).
    k_a = np.real(sum(alpha[m] * np.exp(-1j * m * phases) for m in m_vals))
    k_b = np.real(sum(beta[m] * np.exp(-1j * m * phases) for m in m_vals))

    def _phase_sum(coeffs):
        total = np.zeros(len(phases), dtype=complex)
        for (m, n), c in coeffs.items():
            total += c * np.exp(-1j * (n - m) * phases)
        return total.real

    s_aa = _phase_sum(c_aa)
    s_bb = _phase_sum(c_bb)
    s_d = _phase_sum(c_ab_ba)

    # Broadcast over the (phase, psi) grid.
    kappa2 = k_a[:, None] * cos2psi[None, :] + k_b[:, None] * sin2psi[None, :]
    rho2opt = (
        s_aa[:, None] * (cos2psi**2)[None, :]
        + s_d[:, None] * (cos2psi * sin2psi)[None, :]
        + s_bb[:, None] * (sin2psi**2)[None, :]
    )
    return log_Zn + kappa2 - 0.5 * rho2opt


def phase_psi_marginalized_log_likelihood(likelihood, theta, phases, psis):
    """
    Phase- and psi-marginalized log-likelihood for one sample.

    Marginalizes over uniform ``phi ~ U[0, 2 pi)`` and ``psi ~ U[0, pi)`` by
    averaging ``exp(log L)`` over the grid (so ``phases`` / ``psis`` should be
    uniform with ``endpoint=False``).
    """
    log_l = phase_psi_grid_log_likelihood(likelihood, theta, phases, psis)
    return logsumexp(log_l) - np.log(log_l.size)


class PhasePsiMarginalizedLikelihood(Likelihood):
    """
    Drop-in likelihood wrapper marginalizing phase and psi on a fixed grid.

    Wraps a plain (non-marginalized) ``StationaryGaussianGWLikelihood`` and exposes
    ``log_likelihood`` / ``log_likelihood_multi`` (inherited), so it can replace
    ``result.likelihood`` for importance sampling of a model trained without phase
    and psi.

    Parameters
    ----------
    base_likelihood : StationaryGaussianGWLikelihood
        Built with no marginalization kwargs.
    n_grid_phase : int
    n_grid_psi : int
    """

    def __init__(self, base_likelihood, n_grid_phase=512, n_grid_psi=128):
        if base_likelihood.waveform_generator.spin_conversion_phase != 0:
            raise ValueError(
                "PhasePsiMarginalizedLikelihood requires spin_conversion_phase = 0."
            )
        if getattr(base_likelihood, "phase_marginalization", False):
            raise ValueError(
                "base_likelihood must not be phase-marginalized; it needs to expose "
                "the per-mode signal. Build it without phase_marginalization_kwargs."
            )
        self.likelihood = base_likelihood
        # endpoint=False: phase 0/2pi and psi 0/pi are equivalent, so the mean over
        # the grid is the uniform-prior integral.
        self.phases = np.linspace(0, 2 * np.pi, n_grid_phase, endpoint=False)
        self.psis = np.linspace(0, np.pi, n_grid_psi, endpoint=False)

    def log_likelihood(self, theta):
        return phase_psi_marginalized_log_likelihood(
            self.likelihood, theta, self.phases, self.psis
        )


# ---------------------------------------------------------------------------
# 2D interpolated sampling / log-prob (factorized phi then psi|phi)
# ---------------------------------------------------------------------------
def _interp_slice_along_phase(phases, values, phase):
    """Linearly interpolate ``values`` (n_phase, n_psi) at scalar ``phase`` -> (n_psi,)."""
    i = int(np.clip(np.searchsorted(phases, phase) - 1, 0, len(phases) - 2))
    p0, p1 = phases[i], phases[i + 1]
    w = 0.0 if p1 == p0 else (phase - p0) / (p1 - p0)
    return (1 - w) * values[i] + w * values[i + 1]


def interpolated_2d_sample_and_log_prob(phases, psis, values):
    """
    Sample ``(phase, psi)`` from an (un-normalized) 2D grid distribution and return
    its log-prob, using the factorization ``p(phi) p(psi | phi)`` with bilby
    ``Interped`` on the phase marginal and the conditional psi slice.

    Parameters
    ----------
    phases : np.ndarray, shape (n_phase,)
    psis : np.ndarray, shape (n_psi,)
    values : np.ndarray, shape (n_phase, n_psi)
        Non-negative grid density (need not be normalized).

    Returns
    -------
    (float, float, float) : phase, psi, log_prob
    """
    marginal_phase = values.sum(axis=1)
    interp_phase = Interped(phases, marginal_phase)
    phase = interp_phase.sample()
    log_prob_phase = interp_phase.ln_prob(phase)

    cond_psi = _interp_slice_along_phase(phases, values, phase)
    interp_psi = Interped(psis, cond_psi)
    psi = interp_psi.sample()
    log_prob_psi = interp_psi.ln_prob(psi)

    return phase, psi, log_prob_phase + log_prob_psi


def interpolated_2d_log_prob(phases, psis, values, phase_eval, psi_eval):
    """Log-prob at ``(phase_eval, psi_eval)`` for the same 2D distribution."""
    marginal_phase = values.sum(axis=1)
    log_prob_phase = Interped(phases, marginal_phase).ln_prob(phase_eval)
    cond_psi = _interp_slice_along_phase(phases, values, phase_eval)
    log_prob_psi = Interped(psis, cond_psi).ln_prob(psi_eval)
    return log_prob_phase + log_prob_psi


# ---------------------------------------------------------------------------
# Synthetic (phase, psi) sampling for a Result
# ---------------------------------------------------------------------------
def _build_grid_density(likelihood, theta, phases, psis, uniform_weight):
    """Grid posterior over (phase, psi) for one sample, with a mass-covering floor."""
    log_l = phase_psi_grid_log_likelihood(likelihood, theta, phases, psis)
    density = np.exp(log_l - np.max(log_l))
    density += density.mean() * uniform_weight
    return density


def _sample_for_theta(theta, likelihood, phases, psis, uniform_weight):
    density = _build_grid_density(likelihood, theta, phases, psis, uniform_weight)
    return interpolated_2d_sample_and_log_prob(phases, psis, density)


def _log_prob_for_theta(theta, likelihood, phases, psis, uniform_weight):
    phase_eval = theta["phase"]
    psi_eval = theta["psi"]
    density = _build_grid_density(likelihood, theta, phases, psis, uniform_weight)
    return interpolated_2d_log_prob(phases, psis, density, phase_eval, psi_eval)


def _map_over_theta(func, theta, num_processes):
    """Like apply_func_with_multiprocessing but tolerates tuple-valued outputs."""
    with threadpool_limits(limits=1, user_api="blas"):
        rows = (row.to_dict() for _, row in theta.iterrows())
        if num_processes > 1:
            with Pool(processes=num_processes) as pool:
                result = pool.map(func, rows)
        else:
            result = list(map(func, rows))
    return result


def sample_synthetic_phase_psi(result, synthetic_kwargs, inverse=False):
    """
    Sample a synthetic ``(phase, psi)`` for samples that lack both, by likelihood
    evaluation on a ``(phase, psi)`` grid. The 2D analogue of
    ``Result.sample_synthetic_phase``.

    Forward (``inverse=False``): draws ``phase`` and ``psi`` for each sample from the
    grid posterior and adds the corresponding ``log_prob`` to ``result.samples
    ['log_prob']``. Modifies ``result.samples`` in place (adds ``phase``, ``psi``)
    and restores the phase prior into ``result.prior``.

    Inverse (``inverse=True``): evaluates the synthetic ``(phase, psi)`` log-prob at
    the existing sample values and stores it in ``result.samples['log_prob']``.

    Parameters
    ----------
    result : dingo.gw.result.Result
        Result whose samples lack ``phase`` and ``psi`` (the model did not infer
        them). ``result.phase_prior`` must be ``U[0, 2 pi)`` and ``result.prior
        ['psi']`` must be ``U[0, pi)``.
    synthetic_kwargs : dict
        ``n_grid_phase`` (required), ``n_grid_psi`` (required), and optional
        ``num_processes`` (default 1), ``uniform_weight`` (default 0.01).
    inverse : bool, default False
    """
    n_grid_phase = synthetic_kwargs["n_grid_phase"]
    n_grid_psi = synthetic_kwargs["n_grid_psi"]
    num_processes = synthetic_kwargs.get("num_processes", 1)
    uniform_weight = synthetic_kwargs.get("uniform_weight", 0.01)

    # --- validate priors -----------------------------------------------------
    phase_prior = result.phase_prior
    if not (
        isinstance(phase_prior, Uniform)
        and (phase_prior._minimum, phase_prior._maximum) == (0, 2 * np.pi)
    ):
        raise ValueError(
            f"Phase prior should be uniform [0, 2pi); got {phase_prior}. (Is the "
            "model phase-marginalized?)"
        )
    if "psi" not in result.prior:
        raise ValueError("psi prior not found in result.prior.")
    psi_prior = result.prior["psi"]
    if not (
        isinstance(psi_prior, Uniform)
        and np.isclose(psi_prior._minimum, 0)
        and np.isclose(psi_prior._maximum, np.pi)
    ):
        raise ValueError(f"psi prior should be uniform [0, pi); got {psi_prior}.")
    if not inverse and "psi" in result.samples:
        raise ValueError("Samples already contain 'psi'; nothing to synthesize.")

    # --- restrict to in-prior samples ----------------------------------------
    # Parameters present in both the (non-constraint) prior and the samples. psi is
    # in the prior but not the samples, so it is naturally excluded here.
    param_keys = [
        k
        for k, v in result.prior.items()
        if not isinstance(v, Constraint) and k in result.samples
    ]
    theta = result.samples[param_keys]
    log_prior = result.prior.ln_prob(theta, axis=0)
    constraints = result.prior.evaluate_constraints(theta)
    np.putmask(log_prior, constraints == 0, -np.inf)
    within_prior = log_prior != -np.inf
    num_valid = int(np.sum(within_prior))

    num_processes = min(num_processes, max(num_valid // 10, 1))

    if not inverse:
        result._build_likelihood()
    likelihood = result.likelihood

    phases = np.linspace(0, 2 * np.pi, n_grid_phase)
    psis = np.linspace(0, np.pi, n_grid_psi)

    if not inverse:
        func = partial(
            _sample_for_theta,
            likelihood=likelihood,
            phases=phases,
            psis=psis,
            uniform_weight=uniform_weight,
        )
        results = _map_over_theta(func, theta.iloc[within_prior], num_processes)
        results = np.asarray(results)  # (num_valid, 3): phase, psi, delta_log_prob
        new_phase, new_psi, delta_log_prob = results.T

        phase_array = np.full(len(theta), 0.0)
        psi_array = np.full(len(theta), 0.0)
        delta_log_prob_array = np.full(len(theta), -np.nan)
        phase_array[within_prior] = new_phase
        psi_array[within_prior] = new_psi
        delta_log_prob_array[within_prior] = delta_log_prob

        result.samples["phase"] = phase_array
        result.samples["psi"] = psi_array
        result.samples["log_prob"] += delta_log_prob_array

        # phase was split off at Result init; psi stayed in the prior. Restore phase.
        result.prior["phase"] = phase_prior
        result.phase_prior = None
        result.likelihood = None
    else:
        # theta here still includes phase/psi columns for evaluation.
        theta_eval = result.samples[param_keys + ["phase", "psi"]]
        func = partial(
            _log_prob_for_theta,
            likelihood=likelihood,
            phases=phases,
            psis=psis,
            uniform_weight=uniform_weight,
        )
        log_prob = np.asarray(
            _map_over_theta(func, theta_eval.iloc[within_prior], num_processes)
        )
        log_prob_array = np.full(len(theta), -np.nan)
        log_prob_array[within_prior] = log_prob
        result.samples["log_prob"] = log_prob_array
