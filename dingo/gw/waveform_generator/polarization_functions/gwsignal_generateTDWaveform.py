"""Wrapper for lalsimulation.gwsignal.core.waveform.GenerateTDWaveform.

Ported from dingo-waveform polarization_functions/gwsignal_generateTDWaveform.py.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Optional, cast

import pyseobnr
from lalsimulation.gwsignal.core import waveform
from lalsimulation.gwsignal.models import gwsignal_get_waveform_generator

from dingo.gw.approximant import Approximant
from dingo.gw.domains import DomainParameters
from ..binary_black_hole_parameters import BinaryBlackHoleParameters
from ..gw_signal_parameters import GwSignalParameters
from ..polarizations import Polarization
from ..waveform_generator_parameters import WaveformGeneratorParameters
from ..waveform_parameters import BBHWaveformParameters

_logger = logging.getLogger(__name__)


@dataclass
class _GWSignal_GenerateTDModesParameters(GwSignalParameters):

    @classmethod
    def from_binary_black_holes_parameters(
        cls,
        bbh_params: BinaryBlackHoleParameters,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        f_start: Optional[float] = None,
    ) -> "_GWSignal_GenerateTDModesParameters":
        gw_signal_params = super().from_binary_black_hole_parameters(
            bbh_params,
            domain_params,
            spin_conversion_phase,
            f_start,
        )
        return cls(**asdict(gw_signal_params))

    def apply(self, approximant: Approximant) -> Polarization:
        _logger.debug(
            self.to_table(
                "generating polarization using "
                "lalsimulation.gwsignal.core.waveform.GenerateTDWaveform"
            )
        )

        generator = gwsignal_get_waveform_generator(approximant)
        params = {k: v for k, v in asdict(self).items() if v is not None}
        hpc = waveform.GenerateTDWaveform(params, generator)
        return Polarization(h_cross=hpc.hc.value, h_plus=hpc.hp.value)


def gwsignal_generate_TD_modes(
    waveform_gen_params: WaveformGeneratorParameters,
    waveform_params: BBHWaveformParameters,
) -> Polarization:
    """Wrapper over lalsimulation.gwsignal.core.waveform.GenerateTDWaveform."""

    instance = cast(
        _GWSignal_GenerateTDModesParameters,
        _GWSignal_GenerateTDModesParameters.from_waveform_parameters(
            waveform_params,
            waveform_gen_params.domain.get_parameters(),
            waveform_gen_params.f_ref,
            f_start=waveform_gen_params.f_start,
            spin_conversion_phase=waveform_gen_params.spin_conversion_phase,
        ),
    )

    return instance.apply(waveform_gen_params.approximant)
