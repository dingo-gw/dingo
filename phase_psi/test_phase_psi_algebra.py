"""
Unit test for the phase-psi grid algebra (no waveform model required).

Builds random per-mode strains A_m (psi=0) and B_m (psi=pi/4) and random whitened
data, then checks that ``phase_psi_grid_from_per_mode_strains`` reproduces a
brute-force evaluation of ``log Zn + Re<d, mu> - 1/2 <mu, mu>`` with
``mu(phi, psi) = sum_m exp(-i m phi) [cos(2 psi) A_m + sin(2 psi) B_m]``
at every grid point.
"""

import numpy as np

from phase_psi_marginalization import (
    phase_psi_grid_from_per_mode_strains,
    interpolated_2d_sample_and_log_prob,
    interpolated_2d_log_prob,
)

rng = np.random.default_rng(0)


def brute_force_grid(A, B, d, log_Zn, min_idx, phases, psis):
    ifos = list(d.keys())
    m_vals = sorted(A.keys())
    out = np.zeros((len(phases), len(psis)))
    for ip, phi in enumerate(phases):
        for jp, psi in enumerate(psis):
            c, s = np.cos(2 * psi), np.sin(2 * psi)
            kappa2 = 0.0
            rho2opt = 0.0
            for i in ifos:
                mu = sum(
                    np.exp(-1j * m * phi) * (c * A[m][i] + s * B[m][i])
                    for m in m_vals
                )
                kappa2 += np.sum((d[i].conj() * mu)[min_idx:]).real
                rho2opt += np.sum((mu.conj() * mu)[min_idx:]).real
            out[ip, jp] = log_Zn + kappa2 - 0.5 * rho2opt
    return out


def test_grid_matches_brute_force():
    n_freq, min_idx = 64, 5
    ifos = ["H1", "L1", "V1"]
    m_vals = [-2, -1, 0, 1, 2]

    def cplx(shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    A = {m: {i: cplx(n_freq) for i in ifos} for m in m_vals}
    B = {m: {i: cplx(n_freq) for i in ifos} for m in m_vals}
    d = {i: cplx(n_freq) for i in ifos}
    log_Zn = -0.5 * sum(np.sum((d[i].conj() * d[i])[min_idx:]).real for i in ifos)

    phases = np.linspace(0, 2 * np.pi, 17, endpoint=False)
    psis = np.linspace(0, np.pi, 11, endpoint=False)

    fast = phase_psi_grid_from_per_mode_strains(
        A, B, d, log_Zn, min_idx, phases, psis
    )
    slow = brute_force_grid(A, B, d, log_Zn, min_idx, phases, psis)

    max_abs = np.max(np.abs(fast - slow))
    print(f"max |fast - brute| = {max_abs:.3e}")
    assert max_abs < 1e-8, max_abs


def test_antenna_reconstruction_identity():
    # h_d(psi) = cos(2psi) h_d(0) + sin(2psi) h_d(pi/4) for the standard antenna
    # pattern convention. Verify against an explicit F+/Fx model.
    fp0, fc0 = 0.7, -0.3  # arbitrary antenna responses at psi=0
    hp = rng.standard_normal(8) + 1j * rng.standard_normal(8)
    hx = rng.standard_normal(8) + 1j * rng.standard_normal(8)
    for psi in rng.uniform(0, np.pi, size=5):
        c, s = np.cos(2 * psi), np.sin(2 * psi)
        fp = c * fp0 + s * fc0
        fc = -s * fp0 + c * fc0
        h_psi = fp * hp + fc * hx
        h0 = fp0 * hp + fc0 * hx
        h_quarter = fc0 * hp - fp0 * hx  # = h_d(pi/4)
        recon = c * h0 + s * h_quarter
        assert np.max(np.abs(h_psi - recon)) < 1e-12


def test_2d_interp_sample_and_log_prob_consistency():
    # Joint log_prob from sampling should equal the standalone log_prob evaluator.
    phases = np.linspace(0, 2 * np.pi, 40)
    psis = np.linspace(0, np.pi, 30)
    values = rng.uniform(0.1, 1.0, size=(len(phases), len(psis)))
    for _ in range(20):
        ph, ps, lp = interpolated_2d_sample_and_log_prob(phases, psis, values)
        lp2 = interpolated_2d_log_prob(phases, psis, values, ph, ps)
        assert abs(lp - lp2) < 1e-9, (lp, lp2)
        assert 0 <= ph <= 2 * np.pi and 0 <= ps <= np.pi


def test_2d_interp_normalizes():
    # The factorized density integrates to 1 over [0,2pi] x [0,pi].
    phases = np.linspace(0, 2 * np.pi, 200)
    psis = np.linspace(0, np.pi, 150)
    values = rng.uniform(0.1, 1.0, size=(len(phases), len(psis)))
    PH, PS = np.meshgrid(phases, psis, indexing="ij")
    logp = np.array(
        [
            interpolated_2d_log_prob(phases, psis, values, ph, ps)
            for ph, ps in zip(PH.ravel(), PS.ravel())
        ]
    ).reshape(PH.shape)
    integral = np.trapz(np.trapz(np.exp(logp), psis, axis=1), phases)
    print(f"integral of factorized density = {integral:.4f}")
    assert abs(integral - 1.0) < 5e-3, integral


if __name__ == "__main__":
    test_grid_matches_brute_force()
    test_antenna_reconstruction_identity()
    test_2d_interp_sample_and_log_prob_consistency()
    test_2d_interp_normalizes()
    print("All algebra tests passed.")
