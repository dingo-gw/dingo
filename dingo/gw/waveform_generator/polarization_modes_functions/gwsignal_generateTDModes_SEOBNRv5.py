"""Wrapper for GWSignal GenerateTDModes with SEOBNRv5 conditioning.

Ported from dingo-waveform polarization_modes_functions/gwsignal_generateTDModes_SEOBNRv5.py.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Union, cast

import astropy.units
import gwpy
import gwpy.frequencyseries
import lal
import lalsimulation
import numpy as np
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

_logger = logging.getLogger(__name__)


@dataclass
class _SEOBRNRv5Conditioning:
    f_min: float
    new_f_start: float
    t_extra: float
    original_f_min: float
    f_isco: float

    def taper_td_modes_for_SEOBRNRv5_extra_time(
        self,
        series: Union[gwpy.timeseries.TimeSeries, gwpy.frequencyseries.FrequencySeries],
    ) -> lal.CreateCOMPLEX16TimeSeries:
        h_tapered_re = lal.CreateREAL8TimeSeries(
            "h_tapered", series.epoch.value, 0, series.dt.value, None, len(series)
        )
        h_tapered_re.data.data = series.value.copy().real

        h_tapered_im = lal.CreateREAL8TimeSeries(
            "h_tapered_im", series.epoch.value, 0, series.dt.value, None, len(series)
        )
        h_tapered_im.data.data = series.value.copy().imag

        lalsimulation.SimInspiralTDConditionStage1(
            h_tapered_re, h_tapered_im, self.t_extra, self.original_f_min
        )
        lalsimulation.SimInspiralTDConditionStage2(
            h_tapered_re, h_tapered_im, self.f_min, self.f_isco
        )

        h_return = lal.CreateCOMPLEX16TimeSeries(
            "h_return",
            h_tapered_re.epoch,
            0,
            h_tapered_re.deltaT,
            None,
            h_tapered_re.data.length,
        )

        h_return.data.data = h_tapered_re.data.data + 1j * h_tapered_im.data.data

        return h_return


@dataclass
class _GenerateTDModesLOConditionalExtraTimeParameters(GwSignalParameters):

    @classmethod
    def from_binary_black_holes_parameters(
        cls,
        bbh_params: BinaryBlackHoleParameters,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        f_start: Optional[float] = None,
    ) -> "_GenerateTDModesLOConditionalExtraTimeParameters":
        gw_signal_params = super().from_binary_black_hole_parameters(
            bbh_params,
            domain_params,
            spin_conversion_phase,
            f_start,
        )
        return cls(**asdict(gw_signal_params))

    def _get_starting_frequency_for_SEOBRNRv5_conditioning(
        self, extra_time_fraction: float = 0.1, extra_cycles: float = 3.0
    ) -> _SEOBRNRv5Conditioning:
        f_min = self.f22_start.value
        m1 = self.mass1.value
        m2 = self.mass2.value
        S1z = self.spin1z.value
        S2z = self.spin2z.value
        original_f_min = f_min

        f_isco = 1.0 / (pow(9.0, 1.5) * np.pi * (m1 + m2) * lal.MTSUN_SI)
        if f_min > f_isco:
            f_min = f_isco

        tchirp = lalsimulation.SimInspiralChirpTimeBound(
            f_min, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, S1z, S2z
        )
        spinkerr = lalsimulation.SimInspiralFinalBlackHoleSpinBound(S1z, S2z)
        tmerge = lalsimulation.SimInspiralMergeTimeBound(
            m1 * lal.MSUN_SI, m2 * lal.MSUN_SI
        ) + lalsimulation.SimInspiralRingdownTimeBound(
            (m1 + m2) * lal.MSUN_SI, spinkerr
        )

        textra = extra_cycles / f_min
        f_start = lalsimulation.SimInspiralChirpStartFrequencyBound(
            (1.0 + extra_time_fraction) * tchirp + tmerge + textra,
            m1 * lal.MSUN_SI,
            m2 * lal.MSUN_SI,
        )

        f_isco = 1.0 / (pow(6.0, 1.5) * np.pi * (m1 + m2) * lal.MTSUN_SI)

        return _SEOBRNRv5Conditioning(
            f_min=f_min,
            new_f_start=f_start,
            t_extra=extra_time_fraction * tchirp + textra,
            original_f_min=original_f_min,
            f_isco=f_isco,
        )

    def apply(
        self, approximant: Approximant, domain: BaseFrequencyDomain, phase: float
    ) -> Dict[Mode, Polarization]:
        SEOBRNRv5_conditioning = (
            self._get_starting_frequency_for_SEOBRNRv5_conditioning()
        )

        _logger.debug(
            self.to_table("generating polarization using waveform.GenerateTDModes")
        )

        generator = gwsignal_get_waveform_generator(approximant)
        params = {k: v for k, v in asdict(self).items() if v is not None}
        params["f22_start"] = SEOBRNRv5_conditioning.new_f_start * astropy.units.Hz
        hlm_td: GravitationalWaveModes = waveform.GenerateTDModes(params, generator)

        _logger.debug("tapering TD modes for SEOBRNRv5 extra time")

        hlms_lal: Dict[Modes, lal.CreateCOMPLEX16TimeSeries] = {}
        for key, value in hlm_td.items():
            if type(key) != str:
                modes: Modes = (Mode(key[0]), Mode(key[1]))
                hlm_lal = (
                    SEOBRNRv5_conditioning.taper_td_modes_for_SEOBRNRv5_extra_time(
                        value
                    )
                )
                hlms_lal[modes] = hlm_lal

        h: Dict[Modes, FrequencySeries] = domain.convert_td_modes_to_fd(hlms_lal)
        return get_polarizations_from_fd_modes_m(h, Iota(self.inclination.value), phase)


def gwsignal_generate_TD_modes_SEOBNRv5(
    waveform_gen_params: WaveformGeneratorParameters,
    waveform_params: BBHWaveformParameters,
) -> Dict[Mode, Polarization]:
    """Wrapper for GWSignal GenerateTDModes with SEOBNRv5 conditioning."""

    approximant = waveform_gen_params.approximant

    supported_approximants = (Approximant("SEOBNRv5PHM"), Approximant("SEOBNRv5HM"))
    if approximant not in supported_approximants:
        raise ValueError(
            "generate_TD_modes_LO_cond_extra_time does not support the approximant "
            f"{approximant}. Supported: {' '.join(supported_approximants)}"
        )

    if not isinstance(waveform_gen_params.domain, BaseFrequencyDomain):
        raise ValueError(
            "generate_TD_modes_LO_cond_extra_time can only be applied using on a BaseFrequencyDomain "
            f"(got {type(waveform_gen_params.domain)})"
        )

    if waveform_params.phase is None:
        raise ValueError(
            "generate_TD_modes_LO_cond_extra_time: phase parameter should not be None"
        )

    instance = cast(
        _GenerateTDModesLOConditionalExtraTimeParameters,
        _GenerateTDModesLOConditionalExtraTimeParameters.from_waveform_parameters(
            waveform_params,
            waveform_gen_params.domain.get_parameters(),
            waveform_gen_params.f_ref,
            waveform_gen_params.spin_conversion_phase,
            waveform_gen_params.f_start,
        ),
    )

    return instance.apply(
        approximant, waveform_gen_params.domain, waveform_params.phase
    )
