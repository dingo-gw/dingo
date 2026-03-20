from dataclasses import dataclass
from typing import Dict, Tuple

import lal
import lalsimulation as LS
import numpy as np

from dingo.gw.logs import TableStr
from dingo.gw.types import FrequencySeries, Mode, Modes


def rotate_z(
    angle: float, vx: float, vy: float, vz: float
) -> Tuple[float, float, float]:
    """Rotate a vector around the z-axis."""
    vx_new = vx * np.cos(angle) - vy * np.sin(angle)
    vy_new = vx * np.sin(angle) + vy * np.cos(angle)
    return vx_new, vy_new, vz


def rotate_y(
    angle: float, vx: float, vy: float, vz: float
) -> Tuple[float, float, float]:
    """Rotate a vector around the y-axis."""
    vx_new = vx * np.cos(angle) + vz * np.sin(angle)
    vz_new = -vx * np.sin(angle) + vz * np.cos(angle)
    return vx_new, vy, vz_new


@dataclass
class Spins(TableStr):
    """
    Container class for spin parameters of a binary system.

    Attributes
    ----------
    iota : float
        Inclination angle between the binary's orbital angular momentum
        and the line of sight
    s1x, s1y, s1z : float
        Dimensionless spin components of the first object
    s2x, s2y, s2z : float
        Dimensionless spin components of the second object
    """

    iota: float
    s1x: float
    s1y: float
    s1z: float
    s2x: float
    s2y: float
    s2z: float

    def get_JL0_euler_angles(
        self, m1: float, m2: float, f_ref: float, phase: float
    ) -> Tuple[float, float, float]:
        """
        Calculate the Euler angles for the transformation from the J frame
        to the L0 frame.

        Parameters
        ----------
        m1 : Mass of the first object (solar masses).
        m2 : Mass of the second object (solar masses).
        f_ref : Reference frequency.
        phase : Phase of the waveform.

        Returns
        -------
        The Euler angles (alpha_0, beta_0, gamma_0).
        """
        m = m1 + m2
        eta = m1 * m2 / m**2
        v0 = (m * lal.MTSUN_SI * np.pi * f_ref) ** (1 / 3)

        m1sq = m1 * m1
        m2sq = m2 * m2

        s1x = m1sq * self.s1x
        s1y = m1sq * self.s1y
        s1z = m1sq * self.s1z
        s2x = m2sq * self.s2x
        s2y = m2sq * self.s2y
        s2z = m2sq * self.s2z

        delta = np.sqrt(1 - 4 * eta)
        m1_prime = (1 + delta) / 2
        m2_prime = (1 - delta) / 2
        Sl = m1_prime**2 * self.s1z + m2_prime**2 * self.s2z
        Sigmal = self.s2z * m2_prime - self.s1z * m1_prime

        # Orbital angular momentum from Appendix G.2 of PRD 103, 104056 (2021).
        Lmag = (m * m * eta / v0) * (
            1
            + v0 * v0 * (1.5 + eta / 6)
            + (27 / 8 - 19 * eta / 8 + eta**2 / 24) * v0**4
            + (
                7 * eta**3 / 1296
                + 31 * eta**2 / 24
                + (41 * np.pi**2 / 24 - 6889 / 144) * eta
                + 135 / 16
            )
            * v0**6
            + (
                -55 * eta**4 / 31104
                - 215 * eta**3 / 1728
                + (356035 / 3456 - 2255 * np.pi**2 / 576) * eta**2
                + eta
                * (
                    -64 * np.log(16 * v0**2) / 3
                    - 6455 * np.pi**2 / 1536
                    - 128 * lal.GAMMA / 3
                    + 98869 / 5760
                )
                + 2835 / 128
            )
            * v0**8
            + (-35 * Sl / 6 - 5 * delta * Sigmal / 2) * v0**3
            + (
                (-77 / 8 + 427 * eta / 72) * Sl
                + delta * (-21 / 8 + 35 * eta / 12) * Sigmal
            )
            * v0**5
        )

        Jx = s1x + s2x
        Jy = s1y + s2y
        Jz = Lmag + s1z + s2z

        Jnorm = np.sqrt(Jx * Jx + Jy * Jy + Jz * Jz)
        Jhatx = Jx / Jnorm
        Jhaty = Jy / Jnorm
        Jhatz = Jz / Jnorm

        # Euler angles from Appendix C of PRD 103, 104056 (2021).
        theta_JL0 = np.arccos(Jhatz)
        phi_JL0 = np.arctan2(Jhaty, Jhatx)

        Nx = np.sin(self.iota) * np.cos(np.pi / 2 - phase)
        Ny = np.sin(self.iota) * np.sin(np.pi / 2 - phase)
        Nz = np.cos(self.iota)

        # Rotate N into J' frame.
        Nx_Jp, Ny_Jp, Nz_Jp = rotate_y(
            -theta_JL0, *rotate_z(-phi_JL0, Nx, Ny, Nz)
        )

        kappa = np.arctan2(Ny_Jp, Nx_Jp)

        alpha_0 = np.pi - kappa
        beta_0 = theta_JL0
        gamma_0 = np.pi - phi_JL0

        return alpha_0, beta_0, gamma_0

    def convert_J_to_L0_frame(
        self,
        hlm_J: Dict[Modes, FrequencySeries],
        m1: float,
        m2: float,
        f_ref: float,
        phase: float,
    ) -> Dict[Modes, FrequencySeries]:
        """
        Convert waveform modes from the J frame to the L0 frame.

        Parameters
        ----------
        hlm_J : Waveform modes in the J frame.
        m1 : Mass of the first object (solar masses).
        m2 : Mass of the second object (solar masses).
        f_ref : Reference frequency.
        phase : Phase of the waveform.

        Returns
        -------
        Waveform modes in the L0 frame.
        """
        alpha_0, beta_0, gamma_0 = self.get_JL0_euler_angles(m1, m2, f_ref, phase)

        hlm_L0: Dict[Modes, FrequencySeries] = {}
        for (l, m), hlm in hlm_J.items():
            for mp_ in range(-l, l + 1):
                mp = Mode(mp_)
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
