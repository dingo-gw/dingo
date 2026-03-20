"""GWSignal parameter container for lalsimulation.gwsignal functions.

Ported from dingo-waveform gw_signals_parameters.py.
"""

import logging
from dataclasses import dataclass, asdict
from typing import Any, Optional

import astropy
import astropy.units
from astropy.units import Quantity

from dingo.gw.domains import DomainParameters
from dingo.gw.logs import TableStr
from .binary_black_hole_parameters import BinaryBlackHoleParameters
from .spins import Spins
from .waveform_parameters import BBHWaveformParameters

_logger = logging.getLogger(__name__)


@dataclass
class GwSignalParameters(TableStr):
    """
    Parameters in a format suitable for calling functions from the
    lalsimulation.gwsignal.core package.

    The optional parameters condition, lmax_nyquist, postadiabatic, and
    postadiabatic_type are specifically used with the SEOBNRv5 model.
    """

    mass1: Quantity
    mass2: Quantity
    spin1x: Quantity
    spin1y: Quantity
    spin1z: Quantity
    spin2x: Quantity
    spin2y: Quantity
    spin2z: Quantity
    deltaT: Quantity
    f22_start: Quantity
    f22_ref: Quantity
    f_max: Quantity
    deltaF: Quantity
    phi_ref: Quantity
    distance: Quantity
    inclination: Quantity
    condition: int
    lmax_nyquist: Optional[int] = None
    postadiabatic: Optional[Any] = None
    postadiabatic_type: Optional[Any] = None

    @classmethod
    def from_binary_black_hole_parameters(
        cls,
        bbh_params: BinaryBlackHoleParameters,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float] = None,
        f_start: Optional[float] = None,
        condition: int = 1,
        lmax_nyquist: Optional[int] = None,
        postadiabatic: Optional[Any] = None,
        postadiabatic_type: Optional[Any] = None,
    ) -> "GwSignalParameters":
        f_min = f_start if f_start is not None else domain_params.f_min

        spins: Spins = bbh_params.get_spins(spin_conversion_phase)

        dp_dict = asdict(domain_params)
        fmax = dp_dict.get("f_max")
        if fmax is None and dp_dict.get("nodes") is not None:
            fmax = dp_dict.get("nodes")[-1]
        delf = dp_dict.get("delta_f")
        if delf is None:
            delf = dp_dict.get("delta_f_initial")
        if delf is None:
            delf = dp_dict.get("base_delta_f")
        delt = dp_dict.get("delta_t")
        if delt is None and fmax is not None:
            delt = 0.5 / fmax

        params = {
            "mass1": bbh_params.mass_1 * astropy.units.solMass,
            "mass2": bbh_params.mass_2 * astropy.units.solMass,
            "spin1x": spins.s1x * astropy.units.dimensionless_unscaled,
            "spin1y": spins.s1y * astropy.units.dimensionless_unscaled,
            "spin1z": spins.s1z * astropy.units.dimensionless_unscaled,
            "spin2x": spins.s2x * astropy.units.dimensionless_unscaled,
            "spin2y": spins.s2y * astropy.units.dimensionless_unscaled,
            "spin2z": spins.s2z * astropy.units.dimensionless_unscaled,
            "deltaT": delt * astropy.units.s if delt is not None else None,
            "f22_start": f_min * astropy.units.Hz,
            "f22_ref": bbh_params.f_ref * astropy.units.Hz,
            "f_max": fmax * astropy.units.Hz if fmax is not None else None,
            "deltaF": delf * astropy.units.Hz if delf is not None else None,
            "phi_ref": bbh_params.phase * astropy.units.rad,
            "distance": bbh_params.luminosity_distance * astropy.units.Mpc,
            "inclination": spins.iota * astropy.units.rad,
            "condition": condition,
            "postadiabatic": postadiabatic,
            "postadiabatic_type": postadiabatic_type,
            "lmax_nyquist": lmax_nyquist,
        }

        return cls(**params)

    @classmethod
    def from_waveform_parameters(
        cls,
        waveform_params: BBHWaveformParameters,
        domain_params: DomainParameters,
        f_ref: float,
        spin_conversion_phase: Optional[float] = None,
        f_start: Optional[float] = None,
    ) -> "GwSignalParameters":
        bbh = BinaryBlackHoleParameters.from_waveform_parameters(waveform_params, f_ref)
        instance = cls.from_binary_black_hole_parameters(
            bbh,
            domain_params,
            spin_conversion_phase,
            f_start,
            lmax_nyquist=waveform_params.lmax_nyquist,
            postadiabatic=waveform_params.postadiabatic,
            postadiabatic_type=waveform_params.postadiabatic_type,
        )

        _logger.debug(
            instance.to_table(
                f"created an instance of {instance.__class__.__name__} with parameters:"
            )
        )

        return instance
