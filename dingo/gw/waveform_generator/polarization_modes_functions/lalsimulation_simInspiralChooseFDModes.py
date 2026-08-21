"""Wrapper for lalsimulation.SimInspiralChooseFDModes.

Ported from dingo-waveform polarization_modes_functions/lalsimulation_simInspiralChooseFDModes.py.
"""

import logging
from copy import deepcopy
from dataclasses import astuple, dataclass
from typing import Dict, Optional, cast

import lal
import lalsimulation as LS

from dingo.gw.approximant import Approximant, get_approximant
from dingo.gw.domains import DomainParameters
from dingo.gw.logs import TableStr
from dingo.gw.types import FrequencySeries, Iota, Mode, Modes
from ..binary_black_hole_parameters import BinaryBlackHoleParameters
from ..polarizations import Polarization, get_polarizations_from_fd_modes_m
from ..spins import Spins
from ..waveform_generator_parameters import WaveformGeneratorParameters
from ..waveform_parameters import BBHWaveformParameters
from . import polarization_modes_utils

_logger = logging.getLogger(__name__)


@dataclass
class _InspiralChooseFDModesParameters(TableStr):

    # Order matters! astuple will use this order for SimInspiralChooseFDModes args.
    mass_1: float
    mass_2: float
    s1x: float
    s1y: float
    s1z: float
    s2x: float
    s2y: float
    s2z: float
    delta_f: float
    f_min: float
    f_max: float
    f_ref: float
    phase: float
    r: float
    iota: Iota
    lal_params: Optional[lal.Dict]
    approximant: int

    def get_spins(self) -> Spins:
        return Spins(
            self.iota, self.s1x, self.s1y, self.s1z, self.s2x, self.s2y, self.s2z
        )

    def convert_J_to_L0_frame(
        self, hlm_J: Dict[Modes, FrequencySeries]
    ) -> Dict[Modes, FrequencySeries]:
        return self.get_spins().convert_J_to_L0_frame(
            hlm_J, self.mass_1, self.mass_2, self.f_ref, self.phase
        )

    @classmethod
    def from_binary_black_hole_parameters(
        cls,
        bbh_parameters: BinaryBlackHoleParameters,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        lal_params: Optional[lal.Dict],
        approximant: Approximant,
        f_start: Optional[float],
    ) -> "_InspiralChooseFDModesParameters":
        spins: Spins = bbh_parameters.get_spins(spin_conversion_phase)
        parameters = {
            "iota": spins.iota,
            "s1x": spins.s1x,
            "s1y": spins.s1y,
            "s1z": spins.s1z,
            "s2x": spins.s2x,
            "s2y": spins.s2y,
            "s2z": spins.s2z,
            "mass_1": bbh_parameters.mass_1,
            "mass_2": bbh_parameters.mass_2,
            "phase": bbh_parameters.phase,
        }

        df = getattr(domain_params, "delta_f", None)
        if df is None:
            df = getattr(domain_params, "delta_f_initial", None)
        if df is None:
            df = getattr(domain_params, "base_delta_f", None)
        parameters["delta_f"] = df

        fmin = getattr(domain_params, "f_min", None)
        fmax = getattr(domain_params, "f_max", None)
        nodes = getattr(domain_params, "nodes", None)
        if nodes is not None:
            if fmin is None:
                fmin = nodes[0]
            if fmax is None:
                fmax = nodes[-1]
        parameters["f_min"] = fmin
        parameters["f_max"] = fmax
        if f_start is not None:
            parameters["f_min"] = f_start
        parameters["f_ref"] = bbh_parameters.f_ref
        parameters["r"] = bbh_parameters.luminosity_distance
        parameters["lal_params"] = lal_params
        parameters["approximant"] = get_approximant(approximant)
        instance = cls(**parameters)
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                instance.to_table("generated inspiral choose fd modes parameters")
            )
        return instance

    @classmethod
    def from_waveform_parameters(
        cls,
        waveform_parameters: BBHWaveformParameters,
        f_ref: float,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        lal_params: Optional[lal.Dict],
        approximant: Approximant,
        f_start: Optional[float],
    ) -> "_InspiralChooseFDModesParameters":
        bbh_parameters = BinaryBlackHoleParameters.from_waveform_parameters(
            waveform_parameters, f_ref
        )
        return cls.from_binary_black_hole_parameters(
            bbh_parameters,
            domain_params,
            spin_conversion_phase,
            lal_params,
            approximant,
            f_start,
        )

    def apply(self) -> Dict[Mode, Polarization]:
        params: "_InspiralChooseFDModesParameters" = deepcopy(self)
        params.mass_1 *= lal.MSUN_SI
        params.mass_2 *= lal.MSUN_SI
        params.r *= 1e6 * lal.PC_SI

        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                params.to_table("calling LS.SimInspiralChooseFDModes with arguments:")
            )

        arguments = list(astuple(params))
        hlm_fd___: LS.SphHarmFrequencySeries = LS.SimInspiralChooseFDModes(*arguments)

        hlm_fd__: Dict[Modes, lal.COMPLEX16FrequencySeries] = (
            polarization_modes_utils.linked_list_modes_to_dict_modes(hlm_fd___)
        )
        hlm_fd_: Dict[Modes, FrequencySeries] = {
            k: v.data.data for k, v in hlm_fd__.items()
        }

        hlm_fd: Dict[Modes, FrequencySeries] = self.convert_J_to_L0_frame(hlm_fd_)

        return get_polarizations_from_fd_modes_m(hlm_fd, self.iota, self.phase)


def lalsim_inspiral_choose_FD_modes(
    waveform_gen_params: WaveformGeneratorParameters,
    waveform_params: BBHWaveformParameters,
) -> Dict[Mode, Polarization]:
    """Wrapper over lalsimulation.SimInspiralChooseFDModes."""

    if waveform_params.phase is None:
        raise ValueError(
            "inspiral_choose_FD_modes: phase parameter should not be None"
        )

    instance = cast(
        _InspiralChooseFDModesParameters,
        _InspiralChooseFDModesParameters.from_waveform_parameters(
            waveform_params,
            waveform_gen_params.f_ref,
            waveform_gen_params.domain.get_parameters(),
            waveform_gen_params.spin_conversion_phase,
            waveform_gen_params.lal_params,
            waveform_gen_params.approximant,
            waveform_gen_params.f_start,
        ),
    )

    return instance.apply()
