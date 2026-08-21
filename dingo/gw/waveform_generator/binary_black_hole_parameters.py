import logging
from dataclasses import dataclass
from numbers import Number
from typing import List, Optional, Tuple, Union

import lal
import numpy as np
from bilby.gw.conversion import (
    bilby_to_lalsimulation_spins,
    convert_to_lal_binary_black_hole_parameters,
)

from dingo.gw.logs import TableStr, to_table
from .spins import Spins
from .waveform_parameters import BBHWaveformParameters

_logger = logging.getLogger(__name__)


def convert_to_float(x: Union[np.ndarray, Number, float]) -> float:
    """Convert x to float."""
    if type(x) is float:
        return x
    if isinstance(x, np.ndarray):
        if x.ndim == 0 or (x.ndim == 1 and len(x) == 1):
            return float(x.item())
        raise ValueError(
            f"Expected an array of length one, but got shape = {x.shape}"
        )
    return float(x)  # type: ignore


@dataclass
class BinaryBlackHoleParameters(TableStr):
    """Parameters required for calling lalsimulation functions."""

    luminosity_distance: float
    a_1: float
    a_2: float
    phi_jl: float
    phi_12: float
    tilt_1: float
    tilt_2: float
    phase: float
    theta_jn: float
    f_ref: float
    chirp_mass: float
    mass_ratio: float
    mass_1: float
    mass_2: float
    total_mass: Optional[float] = None
    geocent_time: Optional[float] = None
    l_max: Optional[int] = None
    chi_1: Optional[float] = None
    chi_2: Optional[float] = None
    cos_tilt_1: Optional[float] = None
    cos_tilt_2: Optional[float] = None

    @classmethod
    def from_waveform_parameters(
        cls,
        waveform_params: BBHWaveformParameters,
        f_ref: float,
    ) -> "BinaryBlackHoleParameters":
        """
        Create a BinaryBlackHoleParameters instance from WaveformParameters.

        Parameters
        ----------
        waveform_params :
            The waveform parameters to convert.
        f_ref :
            The reference frequency.

        Returns
        -------
        A BinaryBlackHoleParameters instance.
        """
        params = {
            "luminosity_distance": waveform_params.luminosity_distance,
            "redshift": waveform_params.redshift,
            "comoving_distance": waveform_params.comoving_distance,
            "chi_1": waveform_params.chi_1,
            "chi_2": waveform_params.chi_2,
            "chi_1_in_plane": waveform_params.chi_1_in_plane,
            "chi_2_in_plane": waveform_params.chi_2_in_plane,
            "a_1": waveform_params.a_1,
            "a_2": waveform_params.a_2,
            "phi_jl": waveform_params.phi_jl,
            "phi_12": waveform_params.phi_12,
            "tilt_1": waveform_params.tilt_1,
            "tilt_2": waveform_params.tilt_2,
            "dec": waveform_params.dec,
            "ra": waveform_params.ra,
            "geocent_time": waveform_params.geocent_time,
            "delta_phase": waveform_params.delta_phase,
            "phase": waveform_params.phase,
            "psi": waveform_params.psi,
            "theta_jn": waveform_params.theta_jn,
            "mass_1": waveform_params.mass_1,
            "mass_2": waveform_params.mass_2,
            "total_mass": waveform_params.total_mass,
            "chirp_mass": waveform_params.chirp_mass,
            "mass_ratio": waveform_params.mass_ratio,
            "symmetric_mass_ratio": waveform_params.symmetric_mass_ratio,
            "mass_1_source": waveform_params.mass_1_source,
            "mass_2_source": waveform_params.mass_2_source,
            "total_mass_source": waveform_params.total_mass_source,
            "chirp_mass_source": waveform_params.chirp_mass_source,
            "l_max": waveform_params.l_max,
            "postadiabatic": waveform_params.postadiabatic,
            "postadiabatic_type": waveform_params.postadiabatic_type,
            "lmax_nyquist": waveform_params.lmax_nyquist,
            "f_ref": f_ref,
        }

        # Filter out None parameters
        params = {k: v for k, v in params.items() if v is not None}

        # Convert via bilby
        converted_params, _ = convert_to_lal_binary_black_hole_parameters(params)
        for k, v in converted_params.items():
            converted_params[k] = convert_to_float(v) if v is not None else None

        # Ensure chirp_mass and mass_ratio are present
        if (
            "chirp_mass" not in converted_params
            and "mass_1" in converted_params
            and "mass_2" in converted_params
        ):
            m1 = converted_params["mass_1"]
            m2 = converted_params["mass_2"]
            converted_params["chirp_mass"] = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2

        if (
            "mass_ratio" not in converted_params
            and "mass_1" in converted_params
            and "mass_2" in converted_params
        ):
            m1 = converted_params["mass_1"]
            m2 = converted_params["mass_2"]
            converted_params["mass_ratio"] = m2 / m1

        instance = cls(**converted_params)

        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(instance.to_table("generated binary black hole parameters"))

        return instance

    def get_spins(self, spin_conversion_phase: Optional[float]) -> Spins:
        """
        Calculate the spins from the binary black hole parameters.

        Parameters
        ----------
        spin_conversion_phase :
            The phase for spin conversion. If provided, overrides the phase value.

        Returns
        -------
        The spins as a Spins instance.
        """
        keys: Tuple[str, ...] = (
            "theta_jn",
            "phi_jl",
            "tilt_1",
            "tilt_2",
            "phi_12",
            "a_1",
            "a_2",
            "mass_1",
            "mass_2",
            "f_ref",
            "phase",
        )

        kwargs = {k: getattr(self, k) for k in keys}
        kwargs["mass_1"] *= lal.MSUN_SI
        kwargs["mass_2"] *= lal.MSUN_SI
        args: List[float] = [kwargs[k] for k in keys]

        if spin_conversion_phase is not None:
            args[-1] = spin_conversion_phase

        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                "calling bilby_to_lalsimulation_spins with arguments:\n"
                f"{to_table({k: a for k, a in zip(keys, args)})}"
            )

        iota_and_cart_spins: List[float] = [
            float(convert_to_float(value))
            for value in bilby_to_lalsimulation_spins(*args)
        ]

        instance = Spins(*iota_and_cart_spins)

        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug("generated spins:\n" f"{to_table(instance)}")  # type: ignore

        return instance
