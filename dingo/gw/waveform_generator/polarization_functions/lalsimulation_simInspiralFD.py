"""Wrapper for lalsimulation.SimInspiralFD.

Ported from dingo-waveform polarization_functions/lalsimulation_simInspiralFD.py.
"""

import logging
from dataclasses import dataclass, fields, replace
from math import isclose
from typing import Optional, Tuple, Union

import lal
import lalsimulation as LS
import numpy as np
from nptyping import Float32, NDArray, Shape

from dingo.gw.approximant import Approximant
from dingo.gw.domains import (
    BaseFrequencyDomain,
    DomainParameters,
    MultibandedFrequencyDomain,
    UniformFrequencyDomain,
)
from dingo.gw.logs import TableStr
from dingo.gw.types import Iota, WaveformGenerationError
from ..binary_black_hole_parameters import BinaryBlackHoleParameters
from ..polarization_modes_functions.lalsimulation_simInspiralChooseFDModes import (
    _InspiralChooseFDModesParameters,
)
from ..polarizations import Polarization
from ..spins import Spins
from ..waveform_generator_parameters import WaveformGeneratorParameters
from ..waveform_parameters import BBHWaveformParameters

_logger = logging.getLogger(__name__)


@dataclass
class _LALSim_InspiralFDParameters(TableStr):
    """Parameters for lalsimulation's SimInspiralFD function."""

    # Order matters! These define SimInspiralFD argument order.
    mass_1: float
    mass_2: float
    s1x: float
    s1y: float
    s1z: float
    s2x: float
    s2y: float
    s2z: float
    r: float
    iota: Iota
    phase: float
    longAscNode: float
    eccentricity: float
    meanPerAno: float
    delta_f: float
    f_min: float
    f_max: float
    f_ref: float
    lal_params: Optional[lal.Dict]
    approximant: int

    def get_spins(self) -> Spins:
        return Spins(
            self.iota, self.s1x, self.s1y, self.s1z, self.s2x, self.s2y, self.s2z
        )

    def to_tuple(self) -> Tuple[Union[float, Optional[lal.Dict]]]:
        return tuple(getattr(self, f.name) for f in fields(self))

    def to_lal_args(
        self,
        lal_params_override: Optional[lal.Dict] = None,
    ) -> Tuple[Union[float, Optional[lal.Dict]], ...]:
        """Convert to LAL arguments with unit conversions applied."""
        return (
            self.mass_1 * lal.MSUN_SI,
            self.mass_2 * lal.MSUN_SI,
            self.s1x,
            self.s1y,
            self.s1z,
            self.s2x,
            self.s2y,
            self.s2z,
            self.r * 1e6 * lal.PC_SI,
            self.iota,
            self.phase,
            self.longAscNode,
            self.eccentricity,
            self.meanPerAno,
            self.delta_f,
            self.f_min,
            self.f_max,
            self.f_ref,
            lal_params_override if lal_params_override is not None else self.lal_params,
            self.approximant,
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
    ) -> "_LALSim_InspiralFDParameters":
        inspiral_choose_fd_modes_parameters = (
            _InspiralChooseFDModesParameters.from_binary_black_hole_parameters(
                bbh_parameters,
                domain_params,
                spin_conversion_phase,
                lal_params,
                approximant,
                f_start,
            )
        )
        instance = cls(
            mass_1=inspiral_choose_fd_modes_parameters.mass_1,
            mass_2=inspiral_choose_fd_modes_parameters.mass_2,
            s1x=inspiral_choose_fd_modes_parameters.s1x,
            s1y=inspiral_choose_fd_modes_parameters.s1y,
            s1z=inspiral_choose_fd_modes_parameters.s1z,
            s2x=inspiral_choose_fd_modes_parameters.s2x,
            s2y=inspiral_choose_fd_modes_parameters.s2y,
            s2z=inspiral_choose_fd_modes_parameters.s2z,
            r=inspiral_choose_fd_modes_parameters.r,
            iota=inspiral_choose_fd_modes_parameters.iota,
            phase=inspiral_choose_fd_modes_parameters.phase,
            delta_f=inspiral_choose_fd_modes_parameters.delta_f,
            f_min=inspiral_choose_fd_modes_parameters.f_min,
            f_max=inspiral_choose_fd_modes_parameters.f_max,
            f_ref=inspiral_choose_fd_modes_parameters.f_ref,
            lal_params=inspiral_choose_fd_modes_parameters.lal_params,
            approximant=inspiral_choose_fd_modes_parameters.approximant,
            longAscNode=0,
            eccentricity=0,
            meanPerAno=0,
        )
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(instance.to_table("generated inspiral fd parameters"))
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
    ) -> "_LALSim_InspiralFDParameters":
        bbh_parameters = BinaryBlackHoleParameters.from_waveform_parameters(
            waveform_parameters, f_ref
        )
        df = getattr(domain_params, "delta_f", None)
        if df is None:
            df = getattr(domain_params, "delta_f_initial", None)
        if df is None:
            df = getattr(domain_params, "base_delta_f", None)
        sanitized = replace(domain_params, delta_f=df)

        return cls.from_binary_black_hole_parameters(
            bbh_parameters,
            sanitized,
            spin_conversion_phase,
            lal_params,
            approximant,
            f_start,
        )

    def _turn_off_multibanding(
        self,
        hp: lal.COMPLEX16FrequencySeries,
        hc: lal.COMPLEX16FrequencySeries,
        threshold: float,
    ) -> Tuple[lal.COMPLEX16FrequencySeries, lal.COMPLEX16FrequencySeries, bool]:
        if max(np.max(np.abs(hp.data.data)), np.max(np.abs(hc.data.data))) <= threshold:
            return hp, hc, True

        _logger.debug("instability detected, attempting to turn off multibanding")

        lal_params = (
            self.lal_params if self.lal_params is not None else lal.CreateDict()
        )
        LS.SimInspiralWaveformParamsInsertPhenomXHMThresholdMband(lal_params, 0)
        LS.SimInspiralWaveformParamsInsertPhenomXPHMThresholdMband(lal_params, 0)

        arguments = self.to_lal_args(lal_params_override=lal_params)

        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(self.to_table("calling LS.SimInspiralFD with parameters"))

        hp, hc = LS.SimInspiralFD(*arguments)
        if max(np.max(np.abs(hp.data.data)), np.max(np.abs(hc.data.data))) <= threshold:
            return hp, hc, True
        else:
            return hp, hc, False

    def apply(
        self,
        frequency_array: NDArray[Shape["*"], Float32],
        auto_turn_off_multibanding: bool = True,
        raise_error_on_numerical_unstability: bool = False,
        stability_threshold: float = 1e-20,
        delta_f_tolerance: float = 1e-6,
    ) -> Polarization:
        arguments = self.to_lal_args()

        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                self.to_table("generating polarization using lalsimulation.SimInspiralFD")
            )

        hp, hc = LS.SimInspiralFD(*arguments)

        success = True
        if auto_turn_off_multibanding:
            hp, hc, success = self._turn_off_multibanding(hp, hc, stability_threshold)

        if not success:
            error_message = (
                f"SimInspiralFD failed to reach numerical stability (threshold {stability_threshold}), "
                "attempted to apply SimInspiralWaveformParamsInsertPhenomXHMThresholdMband and "
                "SimInspiralWaveformParamsInsertPhenomXPHMThresholdMband, but no success"
            )
            if raise_error_on_numerical_unstability:
                raise RuntimeError(error_message)
            else:
                _logger.error(error_message)

        if not isclose(self.delta_f, hp.deltaF, rel_tol=delta_f_tolerance):
            raise WaveformGenerationError(
                f"Waveform delta_f is inconsistent with domain: {hp.deltaF} vs {self.delta_f}! "
                f"To avoid this, ensure that f_max = {self.f_max} is a power of two "
                "when you are using a native time-domain waveform model."
            )

        h_plus = np.zeros_like(frequency_array, dtype=complex)
        h_cross = np.zeros_like(frequency_array, dtype=complex)

        lal_delta_f = hp.deltaF
        try:
            f0 = hp.f0
        except AttributeError:
            f0 = 0.0

        diffs = np.diff(frequency_array)
        is_uniform = np.allclose(diffs, diffs[0])
        if is_uniform and np.isclose(diffs[0], lal_delta_f, rtol=0, atol=1e-12):
            start_idx = int(np.rint((float(frequency_array[0]) - float(f0)) / float(lal_delta_f)))
            end_idx = start_idx + len(frequency_array)
            if start_idx < 0 or end_idx > len(hp.data.data):
                start_idx = max(start_idx, 0)
                end_idx = min(end_idx, len(hp.data.data))
            n_copy = end_idx - start_idx
            if n_copy > 0:
                h_plus[:n_copy] = hp.data.data[start_idx:end_idx]
                h_cross[:n_copy] = hc.data.data[start_idx:end_idx]
        else:
            N = len(hp.data.data)
            lal_freqs = f0 + np.arange(N, dtype=frequency_array.dtype) * lal_delta_f
            fmin = lal_freqs[0]
            fmax = lal_freqs[-1]
            valid = (frequency_array >= fmin) & (frequency_array <= fmax)
            if not np.all(valid):
                n_out = int(np.count_nonzero(~valid))
                _logger.warning(
                    f"{n_out} frequencies are outside the LAL waveform range and will be left as zeros."
                )
            if np.any(valid):
                hp_real = np.interp(frequency_array[valid].astype(float), lal_freqs.astype(float), hp.data.data.real.astype(float))
                hp_imag = np.interp(frequency_array[valid].astype(float), lal_freqs.astype(float), hp.data.data.imag.astype(float))
                hc_real = np.interp(frequency_array[valid].astype(float), lal_freqs.astype(float), hc.data.data.real.astype(float))
                hc_imag = np.interp(frequency_array[valid].astype(float), lal_freqs.astype(float), hc.data.data.imag.astype(float))
                h_plus[valid] = hp_real + 1j * hp_imag
                h_cross[valid] = hc_real + 1j * hc_imag

        _logger.debug("undoing the time shift done in SimInspiralFD to the waveform")

        dt = 1 / hp.deltaF + (hp.epoch.gpsSeconds + hp.epoch.gpsNanoSeconds * 1e-9)
        time_shift = np.exp(-1j * 2 * np.pi * dt * frequency_array)
        h_plus *= time_shift
        h_cross *= time_shift

        return Polarization(h_plus=h_plus, h_cross=h_cross)


def lalsim_inspiral_FD(
    waveform_gen_params: WaveformGeneratorParameters,
    waveform_params: BBHWaveformParameters,
) -> Polarization:
    """Wrapper over lalsimulation.SimInspiralFD."""

    if not isinstance(waveform_gen_params.domain, BaseFrequencyDomain):
        raise ValueError(
            "inspiral_fd can only be applied using on a BaseFrequencyDomain "
            f"(got {type(waveform_gen_params.domain)})"
        )

    inspiral_fd_params = _LALSim_InspiralFDParameters.from_waveform_parameters(
        waveform_params,
        waveform_gen_params.f_ref,
        waveform_gen_params.domain.get_parameters(),
        waveform_gen_params.spin_conversion_phase,
        waveform_gen_params.lal_params,
        waveform_gen_params.approximant,
        waveform_gen_params.f_start,
    )

    domain = waveform_gen_params.domain

    if isinstance(domain, MultibandedFrequencyDomain):
        # Generate on the base uniform grid, then decimate to MFD
        base = UniformFrequencyDomain(
            f_min=0.0,
            f_max=domain.f_max,
            delta_f=domain.base_delta_f,
            window_factor=domain.window_factor,
        )
        base_sf = base.sample_frequencies
        base_freqs = base_sf() if callable(base_sf) else base_sf
        pol = inspiral_fd_params.apply(base_freqs.astype(np.float32))
        return Polarization(
            h_plus=domain.decimate(pol.h_plus),
            h_cross=domain.decimate(pol.h_cross),
        )

    sample_freqs_attr = getattr(domain, "sample_frequencies", None)
    frequency_array = sample_freqs_attr() if callable(sample_freqs_attr) else sample_freqs_attr

    return inspiral_fd_params.apply(frequency_array)
