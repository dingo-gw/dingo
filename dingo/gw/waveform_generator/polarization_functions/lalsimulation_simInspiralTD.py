"""Wrapper for lalsimulation.SimInspiralTD.

Ported from dingo-waveform polarization_functions/lalsimulation_simInspiralTD.py.
"""

import logging
from dataclasses import asdict, astuple, dataclass
from typing import Optional

import lal
import lalsimulation as LS

from dingo.gw.approximant import Approximant, get_approximant
from dingo.gw.domains import DomainParameters
from dingo.gw.logs import TableStr
from ..binary_black_hole_parameters import BinaryBlackHoleParameters
from ..polarizations import Polarization
from ..spins import Spins
from ..waveform_generator_parameters import WaveformGeneratorParameters
from ..waveform_parameters import BBHWaveformParameters

_logger = logging.getLogger(__name__)


@dataclass
class _LALSim_InspiralTDParameters(TableStr):

    # Order matters! These define SimInspiralTD argument order.
    mass_1: float
    mass_2: float
    s1x: float
    s1y: float
    s1z: float
    s2x: float
    s2y: float
    s2z: float
    r: float
    iota: float
    phase: float
    longAscNode: float
    eccentricity: float
    meanPerAno: float
    delta_t: float
    f_min: float
    f_ref: float
    lal_params: Optional[lal.Dict]
    approximant: int

    def get_spins(self) -> Spins:
        return Spins(
            self.iota, self.s1x, self.s1y, self.s1z, self.s2x, self.s2y, self.s2z
        )

    @classmethod
    def from_binary_black_hole_parameters(
        cls,
        bbh_parameters: BinaryBlackHoleParameters,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        lal_params: Optional[lal.Dict],
        approximant: Approximant,
        f_ref: float,
    ) -> "_LALSim_InspiralTDParameters":
        spins: Spins = bbh_parameters.get_spins(spin_conversion_phase)
        params = asdict(spins)
        for attr in ("longAscNode", "eccentricity", "meanPerAno"):
            params[attr] = 0.0
        for attr in ("delta_t", "f_min"):
            params[attr] = asdict(domain_params)[attr]
        params["f_ref"] = f_ref
        params["phase"] = bbh_parameters.phase
        params["lal_params"] = lal_params
        params["approximant"] = get_approximant(approximant)
        instance = cls(**params)
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(instance.to_table("generated inspiral td parameters"))
        return instance

    @classmethod
    def from_waveform_parameters(
        cls,
        waveform_params: BBHWaveformParameters,
        f_ref: float,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        lal_params: Optional[lal.Dict],
        approximant: Approximant,
    ) -> "_LALSim_InspiralTDParameters":
        bbh_parameters = BinaryBlackHoleParameters.from_waveform_parameters(
            waveform_params, f_ref
        )
        return cls.from_binary_black_hole_parameters(
            bbh_parameters,
            domain_params,
            spin_conversion_phase,
            lal_params,
            approximant,
            f_ref,
        )

    def apply(self) -> Polarization:
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                self.to_table("generating polarization using lalsimulation.SimInspiralTD")
            )

        parameters = list(astuple(self))
        hp, hc = LS.SimInspiralTD(*parameters)
        return Polarization(h_plus=hp.data.data, h_cross=hc.data.data)


def lalsim_inspiral_TD(
    waveform_gen_params: WaveformGeneratorParameters,
    waveform_params: BBHWaveformParameters,
) -> Polarization:
    """Wrapper over lalsimulation.SimInspiralTD."""

    inspiral_td_params = _LALSim_InspiralTDParameters.from_waveform_parameters(
        waveform_params,
        waveform_gen_params.f_ref,
        waveform_gen_params.domain.get_parameters(),
        waveform_gen_params.spin_conversion_phase,
        waveform_gen_params.lal_params,
        waveform_gen_params.approximant,
    )

    return inspiral_td_params.apply()
