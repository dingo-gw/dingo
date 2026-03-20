"""Wrapper for lalsimulation.gwsignal.core.waveform.GenerateTDModes.

Ported from dingo-waveform polarization_modes_functions/gwsignal_generateTDModes.py.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Union, cast

import gwpy
import gwpy.frequencyseries
import lal
import pyseobnr
from lalsimulation.gwsignal.core import waveform
from lalsimulation.gwsignal.core.gw import (
    GravitationalWaveModes,
    SpinWeightedSphericalHarmonicMode,
)
from lalsimulation.gwsignal.models import gwsignal_get_waveform_generator

from dingo.gw.approximant import Approximant
from dingo.gw.domains import BaseFrequencyDomain, DomainParameters
from dingo.gw.types import FrequencySeries, Iota, Mode, Modes
from ..binary_black_hole_parameters import BinaryBlackHoleParameters
from ..gw_signal_parameters import GwSignalParameters
from ..polarizations import Polarization, get_polarizations_from_fd_modes_m
from ..waveform_generator_parameters import WaveformGeneratorParameters
from ..waveform_parameters import BBHWaveformParameters
from .polarization_modes_utils import taper_td_modes_in_place

_logger = logging.getLogger(__name__)


@dataclass
class _GenerateTDModesLO(GwSignalParameters):

    @classmethod
    def from_binary_black_holes_parameters(
        cls,
        bbh_params: BinaryBlackHoleParameters,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        f_start: Optional[float] = None,
    ) -> "_GenerateTDModesLO":
        gw_signal_params = super().from_binary_black_hole_parameters(
            bbh_params,
            domain_params,
            spin_conversion_phase,
            f_start,
        )
        return cls(**asdict(gw_signal_params))

    def apply(
        self, approximant: Approximant, domain: BaseFrequencyDomain, phase: float
    ) -> Dict[Mode, Polarization]:
        _logger.debug(
            self.to_table("generating polarization using waveform.GenerateTDModes")
        )

        generator = gwsignal_get_waveform_generator(approximant)
        params = {k: v for k, v in asdict(self).items() if v is not None}

        hlm_td: GravitationalWaveModes = waveform.GenerateTDModes(params, generator)

        hlms_lal: Dict[Modes, lal.CreateCOMPLEX16TimeSeries] = {}

        for key, value in hlm_td.items():
            if type(key) != str:
                hlm_lal = lal.CreateCOMPLEX16TimeSeries(
                    "hplus",
                    value.epoch.value,
                    0,
                    value.dt.value,
                    lal.DimensionlessUnit,
                    len(value),
                )
                hlm_lal.data.data = value.value
                hlms_lal[key] = hlm_lal

        taper_td_modes_in_place(hlm_td)

        h: Dict[Modes, FrequencySeries] = domain.convert_td_modes_to_fd(hlms_lal)

        return get_polarizations_from_fd_modes_m(h, Iota(self.inclination), phase)


def gwsignal_generate_TD_modes(
    waveform_gen_params: WaveformGeneratorParameters,
    waveform_params: BBHWaveformParameters,
) -> Dict[Mode, Polarization]:
    """Wrapper over lalsimulation.gwsignal.core.waveform.GenerateTDModes."""

    if not isinstance(waveform_gen_params.domain, BaseFrequencyDomain):
        raise ValueError(
            "generate_TD_modes can only be applied using on a BaseFrequencyDomain "
            f"(got {type(waveform_gen_params.domain)})"
        )

    if waveform_params.phase is None:
        raise ValueError("generate_TD_modes_LO: phase parameter should not be None")

    instance = cast(
        _GenerateTDModesLO,
        _GenerateTDModesLO.from_waveform_parameters(
            waveform_params,
            waveform_gen_params.domain.get_parameters(),
            waveform_gen_params.f_ref,
            waveform_gen_params.f_start,
        ),
    )

    return instance.apply(
        waveform_gen_params.approximant,
        waveform_gen_params.domain,
        waveform_params.phase,
    )
