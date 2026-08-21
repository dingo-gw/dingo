"""Wrapper for lalsimulation.SimInspiralChooseTDModes.

Ported from dingo-waveform polarization_modes_functions/lalsimulation_simInspiralChooseTDModes.py.
"""

import logging
from copy import deepcopy
from dataclasses import InitVar, asdict, astuple, dataclass
from typing import Dict, Optional, cast

import lal
import lalsimulation as LS

from dingo.gw.approximant import Approximant, get_approximant
from dingo.gw.domains import DomainParameters, BaseFrequencyDomain
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
class _InspiralChooseTDModesParameters(TableStr):
    # Warning: order matters! The arguments will be generated in this order.
    phiRef: float
    delta_t: float
    mass_1: float
    mass_2: float
    s1x: float
    s1y: float
    s1z: float
    s2x: float
    s2y: float
    s2z: float
    f_min: float
    f_ref: float
    distance: float
    lal_params: Optional[lal.Dict]
    l_max: int
    approximant: Optional[Approximant]
    iota: InitVar[Iota] = 0

    def __post_init__(self, iota: Iota) -> None:
        # iota is required but is not an argument for LS.SimInspiralChooseTDModes.
        # Defining it as InitVar excludes it from astuple.
        self.iota = iota  # type: ignore

    @classmethod
    def from_binary_black_hole_parameters(
        cls,
        bbh_parameters: BinaryBlackHoleParameters,
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        lal_params: Optional[lal.Dict],
        approximant: Approximant,
        f_start: Optional[float],
        l_max_default: int = 5,
    ) -> "_InspiralChooseTDModesParameters":
        spins: Spins = bbh_parameters.get_spins(spin_conversion_phase)
        params = asdict(spins)
        params["phiRef"] = 0.0
        for attr in ("delta_t", "f_min"):
            params[attr] = getattr(domain_params, attr)
        if f_start is not None:
            params["f_min"] = f_start
        params["f_ref"] = bbh_parameters.f_ref
        params["distance"] = bbh_parameters.luminosity_distance
        params["l_max"] = (
            bbh_parameters.l_max if bbh_parameters.l_max is not None else l_max_default
        )
        params["mass_1"] = bbh_parameters.mass_1
        params["mass_2"] = bbh_parameters.mass_2
        params["lal_params"] = lal_params
        params["approximant"] = get_approximant(approximant)
        instance = cls(**params)
        _logger.debug(
            instance.to_table("generated inspiral choose td modes parameters")
        )
        return instance

    @classmethod
    def from_waveform_parameters(
        cls,
        waveform_params: BBHWaveformParameters,
        f_ref: float,
        f_start: Optional[float],
        domain_params: DomainParameters,
        spin_conversion_phase: Optional[float],
        lal_params: Optional[lal.Dict],
        approximant: Approximant,
    ) -> "_InspiralChooseTDModesParameters":
        bbh_parameters = BinaryBlackHoleParameters.from_waveform_parameters(
            waveform_params, f_ref
        )
        return cls.from_binary_black_hole_parameters(
            bbh_parameters,
            domain_params,
            spin_conversion_phase,
            lal_params,
            approximant,
            f_start,
        )

    def apply(self, domain: BaseFrequencyDomain, phase: float) -> Dict[Mode, Polarization]:
        params: "_InspiralChooseTDModesParameters" = deepcopy(self)
        params.mass_1 *= lal.MSUN_SI
        params.mass_2 *= lal.MSUN_SI
        params.distance *= 1e6 * lal.PC_SI

        _logger.debug(
            params.to_table("calling LS.SimInspiralChooseTDModes with arguments:")
        )

        # iota is an InitVar, excluded from astuple
        hlm__: LS.SphHarmFrequencySeries = LS.SimInspiralChooseTDModes(
            *list(astuple(params))
        )

        hlm_: Dict[Modes, lal.COMPLEX16FrequencySeries] = (
            polarization_modes_utils.linked_list_modes_to_dict_modes(hlm__)
        )

        polarization_modes_utils.taper_td_modes_in_place(hlm_)

        # Convert TD modes to FD modes using domain-specific method
        hlm: Dict[Modes, FrequencySeries] = domain.convert_td_modes_to_fd(hlm_)

        pol: Dict[Mode, Polarization] = get_polarizations_from_fd_modes_m(
            hlm, self.iota, phase  # type: ignore
        )

        return pol


def lalsim_inspiral_choose_TD_modes(
    waveform_gen_params: WaveformGeneratorParameters,
    waveform_params: BBHWaveformParameters,
) -> Dict[Mode, Polarization]:
    """Wrapper over lalsimulation.SimInspiralChooseTDModes."""

    if not isinstance(waveform_gen_params.domain, BaseFrequencyDomain):
        raise ValueError(
            "inspiral_choose_TD_modes can only be applied using on a BaseFrequencyDomain "
            f"(got {type(waveform_gen_params.domain)})"
        )

    if waveform_params.phase is None:
        raise ValueError(
            "inspiral_choose_TD_modes: phase parameter should not be None"
        )

    instance = cast(
        _InspiralChooseTDModesParameters,
        _InspiralChooseTDModesParameters.from_waveform_parameters(
            waveform_params,
            waveform_gen_params.f_ref,
            waveform_gen_params.f_start,
            waveform_gen_params.domain.get_parameters(),
            waveform_gen_params.spin_conversion_phase,
            waveform_gen_params.lal_params,
            waveform_gen_params.approximant,
        ),
    )

    return instance.apply(waveform_gen_params.domain, waveform_params.phase)
