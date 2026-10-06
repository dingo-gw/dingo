"""These functions are used for transforming between J and L0 frames."""
import numpy as np
import lal

def rotate_z(angle, vx, vy, vz):
    vx_new = vx * np.cos(angle) - vy * np.sin(angle)
    vy_new = vx * np.sin(angle) + vy * np.cos(angle)
    return vx_new, vy_new, vz


def rotate_y(angle, vx, vy, vz):
    vx_new = vx * np.cos(angle) + vz * np.sin(angle)
    vz_new = -vx * np.sin(angle) + vz * np.cos(angle)
    return vx_new, vy, vz_new


def get_JL0_euler_angles(p, wfg, spin_conversion_phase=None):
    p = p.copy()
    p["f_ref"] = wfg.f_ref
    phase = p["phase"]

    if spin_conversion_phase is not None:
        p["phase"] = 0.0
    p_lal = wfg._convert_parameters(p)
    p_lal = list(p_lal)

    m1, m2 = p_lal[0:2]
    chi1x, chi1y, chi1z, chi2x, chi2y, chi2z = p_lal[2:8]
    iota = p_lal[9]

    m1 = m1 / lal.MSUN_SI
    m2 = m2 / lal.MSUN_SI

    m = m1 + m2
    eta = m1 * m2 / m ** 2
    v0 = (m * lal.MTSUN_SI * np.pi * wfg.f_ref) ** (1 / 3)

    # All quantities in L0 frame, unless indicated.
    s1x = m1 * m1 * chi1x
    s1y = m1 * m1 * chi1y
    s1z = m1 * m1 * chi1z
    s2x = m2 * m2 * chi2x
    s2y = m2 * m2 * chi2y
    s2z = m2 * m2 * chi2z

    delta = np.sqrt(1 - 4 * eta)
    m1_prime = (1 + delta) / 2
    m2_prime = (1 - delta) / 2
    Sl = m1_prime ** 2 * chi1z + m2_prime ** 2 * chi2z
    Sigmal = chi2z * m2_prime - chi1z * m1_prime

    # This calculation of the orbital angular momentum is taken from Appendix G.2 of PRD 103, 104056 (2021).
    # It may not align exactly with the various XPHM PrecVersions, but the error should not be too big.
    Lmag = (m * m * eta / v0) * (
        1
        + v0 * v0 * (1.5 + eta / 6)
        + (27 / 8 - 19 * eta / 8 + eta ** 2 / 24) * v0 ** 4
        + (
            7 * eta ** 3 / 1296
            + 31 * eta ** 2 / 24
            + (41 * np.pi ** 2 / 24 - 6889 / 144) * eta
            + 135 / 16
        )
        * v0 ** 6
        + (
            -55 * eta ** 4 / 31104
            - 215 * eta ** 3 / 1728
            + (356035 / 3456 - 2255 * np.pi ** 2 / 576) * eta ** 2
            + eta
            * (
                -64 * np.log(16 * v0 ** 2) / 3
                - 6455 * np.pi ** 2 / 1536
                - 128 * lal.GAMMA / 3
                + 98869 / 5760
            )
            + 2835 / 128
        )
        * v0 ** 8
        + (-35 * Sl / 6 - 5 * delta * Sigmal / 2) * v0 ** 3
        + ((-77 / 8 + 427 * eta / 72) * Sl + delta * (-21 / 8 + 35 * eta / 12) * Sigmal)
        * v0 ** 5
    )

    Jx = s1x + s2x
    Jy = s1y + s2y
    Jz = Lmag + s1z + s2z

    Jnorm = np.sqrt(Jx * Jx + Jy * Jy + Jz * Jz)
    Jhatx = Jx / Jnorm
    Jhaty = Jy / Jnorm
    Jhatz = Jz / Jnorm

    # The calculation of the Euler angles is described in Appendix C of PRD 103, 104056 (2021).
    theta_JL0 = np.arccos(Jhatz)
    phi_JL0 = np.arctan2(Jhaty, Jhatx)

    Nx = np.sin(iota) * np.cos(np.pi / 2 - phase)
    Ny = np.sin(iota) * np.sin(np.pi / 2 - phase)
    Nz = np.cos(iota)

    # Rotate N into J' frame.
    Nx_Jp, Ny_Jp, Nz_Jp = rotate_y(-theta_JL0, *rotate_z(-phi_JL0, Nx, Ny, Nz))

    kappa = np.arctan2(Ny_Jp, Nx_Jp)

    alpha_0 = np.pi - kappa
    beta_0 = theta_JL0
    gamma_0 = np.pi - phi_JL0

    return alpha_0, beta_0, gamma_0


def convert_J_to_L0_frame(hlm_J, p, wfg, spin_conversion_phase=None):
    alpha_0, beta_0, gamma_0 = get_JL0_euler_angles(
        p, wfg, spin_conversion_phase=spin_conversion_phase
    )

    hlm_L0 = {}
    for (l, m), hlm in hlm_J.items():
        for mp in range(-l, l + 1):
            wigner_D = (
                np.exp(1j * m * alpha_0)
                * np.exp(1j * mp * gamma_0)
                * lal.WignerdMatrix(l, m, mp, beta_0)
            )
            if (l, mp) not in hlm_L0:
                hlm_L0[(l, mp)] = wigner_D * hlm
            else:
                hlm_L0[(l, mp)] += wigner_D * hlm

    return hlm_L0
