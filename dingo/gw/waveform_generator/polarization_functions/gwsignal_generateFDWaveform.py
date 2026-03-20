"""Wrapper for lalsimulation.gwsignal.core.waveform.GenerateFDWaveform.

Ported from dingo-waveform polarization_functions/gwsignal_generateFDWaveform.py.
"""

import logging
from dataclasses import asdict, dataclass
from math import isclose
from typing import Optional, cast

import numpy as np
import pyseobnr
from lalsimulation.gwsignal.core import waveform
from lalsimulation.gwsignal.core.gw import GravitationalWavePolarizations
from lalsimulation.gwsignal.models import gwsignal_get_waveform_generator

from dingo.gw.approximant import Approximant
from dingo.gw.domains import BaseFrequencyDomain, DomainParameters, MultibandedFrequencyDomain, UniformFrequencyDomain
from dingo.gw.types import WaveformGenerationError
from ..binary_black_hole_parameters import BinaryBlackHoleParameters
from ..gw_signal_parameters import GwSignalParameters
from ..polarizations import Polarization
from ..waveform_generator_parameters import WaveformGeneratorParameters
from ..waveform_parameters import BBHWaveformParameters

_logger = logging.getLogger(__name__)


@dataclass
class _GWSignal_GenerateFDModesParameters(GwSignalParameters):

    @classmethod
    def from_binary_black_holes_parameters(
        cls,
        bbh_params: BinaryBlackHoleParameters,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        f_start: Optional[float] = None,
    ) -> "_GWSignal_GenerateFDModesParameters":
        gw_signal_params = super().from_binary_black_hole_parameters(
            bbh_params,
            domain_params,
            spin_conversion_phase,
            f_start,
        )
        return cls(**asdict(gw_signal_params))

    def apply(
        self, domain: BaseFrequencyDomain, approximant: Approximant, ref_tol
    ) -> Polarization:
        self.lmax_nyquist = 2

        _logger.info(
            self.to_table(
                "generating polarization using "
                "lalsimulation.gwsignal.core.waveform.GenerateFDWaveform"
            )
        )

        generator = gwsignal_get_waveform_generator(approximant)
        params = {k: v for k, v in asdict(self).items() if v is not None}
        hpc: GravitationalWavePolarizations = waveform.GenerateFDWaveform(
            params, generator
        )

        hp = hpc.hp
        hc = hpc.hc

        if not isclose(self.deltaF.value, hp.df.value, rel_tol=ref_tol):
            raise WaveformGenerationError(
                f"Waveform delta_f is inconsistent with domain: {hp.df.value} vs {self.deltaF}! "
                f"To avoid this, ensure that f_max = {self.f_max} is a power of two "
                "when you are using a native time-domain waveform model."
            )

        if isinstance(domain, MultibandedFrequencyDomain):
            base = UniformFrequencyDomain(
                f_min=0.0,
                f_max=domain.f_max,
                delta_f=domain.base_delta_f,
                window_factor=domain.window_factor,
            )
            h_plus_full = hp.value
            h_cross_full = hc.value
            n_base = len(base)
            if len(h_plus_full) < n_base:
                pad = n_base - len(h_plus_full)
                h_plus_full = np.pad(h_plus_full, (0, pad), mode="constant")
                h_cross_full = np.pad(h_cross_full, (0, pad), mode="constant")
            elif len(h_plus_full) > n_base:
                h_plus_full = h_plus_full[:n_base]
                h_cross_full = h_cross_full[:n_base]
            dt = 1 / hp.df.value + hp.epoch.value
            time_shift = np.exp(-1j * 2 * np.pi * dt * base())
            h_plus_full = h_plus_full * time_shift
            h_cross_full = h_cross_full * time_shift
            return Polarization(h_cross=h_cross_full, h_plus=h_plus_full)

        h_plus = np.zeros((len(domain),), dtype=complex)
        h_cross = np.zeros((len(domain),), dtype=complex)

        if len(hp) > len(domain):
            _logger.warning(
                "GWSignal waveform longer than domain's `frequency_array` "
                f"({len(hp)} vs {len(domain)}). Truncating gwsignal array."
            )
            h_plus = hp[: len(h_plus)].value
            h_cross = hc[: len(h_cross)].value
        else:
            h_plus = hp.value
            h_cross = hc.value

        dt = 1 / hp.df.value + hp.epoch.value
        time_shift = np.exp(-1j * 2 * np.pi * dt * domain())
        h_plus *= time_shift
        h_cross *= time_shift

        return Polarization(h_cross=h_cross, h_plus=h_plus)


def gwsignal_generate_FD_modes(
    waveform_gen_params: WaveformGeneratorParameters,
    waveform_params: BBHWaveformParameters,
    ref_tol: float = 1e-6,
) -> Polarization:
    """Wrapper over lalsimulation.gwsignal.core.waveform.GenerateFDWaveform."""

    approximant = waveform_gen_params.approximant

    if not isinstance(waveform_gen_params.domain, BaseFrequencyDomain):
        raise ValueError(
            "generate_FD_modes can only be applied using on a BaseFrequencyDomain "
            f"(got {type(waveform_gen_params.domain)})"
        )

    instance = cast(
        _GWSignal_GenerateFDModesParameters,
        _GWSignal_GenerateFDModesParameters.from_waveform_parameters(
            waveform_params,
            waveform_gen_params.domain.get_parameters(),
            waveform_gen_params.f_ref,
            spin_conversion_phase=waveform_gen_params.spin_conversion_phase,
            f_start=waveform_gen_params.f_start,
        ),
    )

    return instance.apply(waveform_gen_params.domain, approximant, ref_tol)
