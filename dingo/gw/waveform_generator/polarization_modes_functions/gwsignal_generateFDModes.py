"""Wrapper for lalsimulation.gwsignal.core.waveform.GenerateFDModes (IMRPhenomXPHM only).

Ported from dingo-waveform polarization_modes_functions/gwsignal_generateFDModes.py.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Dict, Optional, cast

import lal
import pyseobnr
from lalsimulation.gwsignal.core import waveform
from lalsimulation.gwsignal.core.gw import GravitationalWavePolarizations
from lalsimulation.gwsignal.models import gwsignal_get_waveform_generator

from dingo.gw.approximant import Approximant
from dingo.gw.domains import DomainParameters
from dingo.gw.types import FrequencySeries, Iota, Mode, Modes
from ..binary_black_hole_parameters import BinaryBlackHoleParameters
from ..gw_signal_parameters import GwSignalParameters
from ..polarizations import Polarization, get_polarizations_from_fd_modes_m
from ..spins import Spins
from ..waveform_generator_parameters import WaveformGeneratorParameters
from ..waveform_parameters import BBHWaveformParameters
from .polarization_modes_utils import linked_list_modes_to_dict_modes

_logger = logging.getLogger(__name__)

_SupportedApproximant = Approximant("IMRPhenomXPHM")


@dataclass
class _GenerateFDModesLOParameters(GwSignalParameters):

    @classmethod
    def from_binary_black_holes_parameters(
        cls,
        bbh_params: BinaryBlackHoleParameters,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        f_start: Optional[float] = None,
    ) -> "_GenerateFDModesLOParameters":
        gw_signal_params = super().from_binary_black_hole_parameters(
            bbh_params,
            domain_params,
            spin_conversion_phase,
            f_start,
        )
        return cls(**asdict(gw_signal_params))

    def apply(
        self,
        spin_conversion_phase: float,
        phase: float,
    ) -> Dict[Mode, Polarization]:
        approximant = _SupportedApproximant

        _logger.debug(
            self.to_table("generating polarization using waveform.GenerateFDModes")
        )

        generator = gwsignal_get_waveform_generator(approximant)
        params = {k: v for k, v in asdict(self).items() if v is not None}
        hlm_fd: GravitationalWavePolarizations = waveform.GenerateFDModes(
            params, generator
        )

        hlms_lal = {}
        for key, value in hlm_fd.items():
            if type(key) != str:
                hlm_lal = lal.CreateCOMPLEX16TimeSeries(
                    key,
                    value.epoch.value,
                    0,
                    value.dt.value,
                    lal.DimensionlessUnit,
                    len(value),
                )
                hlm_lal.data.data = value.value
                hlms_lal[key] = hlm_lal

        hlm_fd_: Dict[Modes, lal.COMPLEX16FrequencySeries] = (
            linked_list_modes_to_dict_modes(hlms_lal)
        )
        hlm_fd__: Dict[Modes, FrequencySeries] = {
            k: v.data.data for k, v in hlm_fd_.items()
        }

        spins = Spins(
            self.inclination,
            self.spin1x,
            self.spin1y,
            self.spin1z,
            self.spin2x,
            self.spin2y,
            self.spin2z,
        )
        hlm_fd___: Dict[Modes, FrequencySeries] = spins.convert_J_to_L0_frame(
            hlm_fd__,
            self.mass1,
            self.mass2,
            self.f22_ref,
            spin_conversion_phase,
        )

        return get_polarizations_from_fd_modes_m(
            hlm_fd___, Iota(self.inclination), phase
        )


def gwsignal_generate_FD_modes(
    waveform_gen_params: WaveformGeneratorParameters,
    waveform_params: BBHWaveformParameters,
) -> Dict[Mode, Polarization]:
    """Wrapper over lalsimulation.gwsignal.core.waveform.GenerateFDModes (IMRPhenomXPHM only)."""

    if waveform_gen_params.spin_conversion_phase is None:
        raise ValueError(
            "generate_FD_modes_LO: spin_conversion_phase parameter should not be None"
        )

    if waveform_params.phase is None:
        raise ValueError("generate_FD_modes_LO: phase parameter should not be None")

    if waveform_gen_params.approximant != _SupportedApproximant:
        raise ValueError(
            f"generate_FD_modes_LO supports only {_SupportedApproximant}. "
            f"{waveform_gen_params.approximant} is not supported."
        )

    instance = cast(
        _GenerateFDModesLOParameters,
        _GenerateFDModesLOParameters.from_waveform_parameters(
            waveform_params,
            waveform_gen_params.domain.get_parameters(),
            waveform_gen_params.f_ref,
            waveform_gen_params.f_start,
        ),
    )

    return instance.apply(
        waveform_gen_params.spin_conversion_phase, waveform_params.phase
    )
