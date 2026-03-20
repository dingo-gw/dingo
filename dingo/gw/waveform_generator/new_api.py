"""
New-style WaveformGenerator ABC and subclasses, ported from dingo-waveform.

This module provides a modular, typed waveform generator hierarchy that uses
Polarization dataclasses and WaveformParameters instead of plain dicts.

The existing WaveformGenerator class in waveform_generator.py is preserved
for backward compatibility.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type, TypeAlias, Union, cast

import lal

from dingo.gw.approximant import Approximant
from dingo.gw.domains import Domain, BaseFrequencyDomain, TimeDomain, build_domain
from dingo.gw.imports import check_function_signature, import_function, read_file
from dingo.gw.types import Mode, Modes
from . import polarization_functions, polarization_modes_functions
from .lal_params import get_lal_params
from .polarizations import Polarization, polarizations_to_table
from .waveform_generator_parameters import WaveformGeneratorParameters
from .waveform_parameters import WaveformParameters

_logger = logging.getLogger(__name__)

PolarizationFunction: TypeAlias = Callable[
    [WaveformGeneratorParameters, WaveformParameters], Polarization
]

PolarizationModesFunction: TypeAlias = Callable[
    [WaveformGeneratorParameters, WaveformParameters], Dict[Mode, Polarization]
]

PolarizationFunctions: Dict[str, PolarizationFunction] = {
    "random_inspiral_FD": polarization_functions.random_inspiral_FD,
}

PolarizationModesFunctions: Dict[str, PolarizationModesFunction] = {
    "random_inspiral_FD_modes": polarization_modes_functions.random_inspiral_FD_modes,
}

# Register LAL-based functions if available
try:
    PolarizationFunctions["lalsim_inspiral_FD"] = polarization_functions.lalsim_inspiral_FD
    PolarizationFunctions["lalsim_inspiral_TD"] = polarization_functions.lalsim_inspiral_TD
    PolarizationModesFunctions["lalsim_inspiral_choose_FD_modes"] = (
        polarization_modes_functions.lalsim_inspiral_choose_FD_modes
    )
    PolarizationModesFunctions["lalsim_inspiral_choose_TD_modes"] = (
        polarization_modes_functions.lalsim_inspiral_choose_TD_modes
    )
except AttributeError:
    pass

# Register GWSignal-based functions if available
try:
    PolarizationFunctions["gwsignal_generate_FD_modes"] = (
        polarization_functions.gwsignal_generate_FD_modes
    )
    PolarizationFunctions["gwsignal_generate_TD_modes"] = (
        polarization_functions.gwsignal_generate_TD_modes
    )
    PolarizationModesFunctions["gwsignal_generate_FD_modes"] = (
        polarization_modes_functions.gwsignal_generate_FD_modes
    )
    PolarizationModesFunctions["gwsignal_generate_TD_modes"] = (
        polarization_modes_functions.gwsignal_generate_TD_modes
    )
    PolarizationModesFunctions["gwsignal_generate_TD_modes_SEOBNRv5"] = (
        polarization_modes_functions.gwsignal_generate_TD_modes_SEOBNRv5
    )
except AttributeError:
    pass

polarization_modes_approximants: Tuple[Approximant, ...] = (
    Approximant("SEOBNRv4PHM"),
    Approximant("IMRPhenomXPHM"),
    Approximant("SEOBNRv5PHM"),
    Approximant("SEOBNRv5HM"),
    Approximant("RandomApproximant"),
)


class NewWaveformGenerator(ABC):
    """
    Abstract base class for generating gravitational wave polarizations using
    various waveform approximants and domains.

    Subclasses implement generate_hplus_hcross() with the appropriate backend.
    Subclasses that support mode-separated generation also define
    generate_hplus_hcross_m().

    Use build_waveform_generator() to construct the appropriate subclass
    based on the approximant name.
    """

    def __init__(
        self,
        approximant: Approximant,
        domain: Domain,
        f_ref: float,
        f_start: Optional[float] = None,
        spin_conversion_phase: Optional[float] = None,
        mode_list: Optional[List[Modes]] = None,
        transform: Optional[Union[str, Callable[[Polarization], Polarization]]] = None,
    ):
        lal_params: Optional[lal.Dict]
        if mode_list is not None:
            lal_params = get_lal_params(mode_list)
        else:
            lal_params = None

        if transform is not None:
            if isinstance(transform, str):
                transform = import_function(transform, [Polarization], Polarization)
            else:
                transform = cast(Callable, transform)
                if not check_function_signature(
                    transform, [Polarization], Polarization
                ):
                    raise ValueError(
                        f"waveform_generator: can not use {transform} as polarization "
                        "transform function, as it does not have the required signature "
                        "(args: Polarization, return type: Polarization)"
                    )
        transform = cast(Callable[[Polarization], Polarization], transform)

        self._waveform_gen_params = WaveformGeneratorParameters(
            approximant=approximant,
            domain=domain,
            f_ref=f_ref,
            f_start=f_start,
            spin_conversion_phase=spin_conversion_phase,
            mode_list=mode_list,
            lal_params=lal_params,
            transform=transform,
        )

        if _logger.isEnabledFor(logging.INFO):
            _logger.info(
                self._waveform_gen_params.to_table(
                    "instantiated waveform generator with parameters:"
                )
            )

        # Batch transform pipeline (for compression, whitening, etc.)
        self.transform = None

    @abstractmethod
    def generate_hplus_hcross(
        self, waveform_parameters: WaveformParameters
    ) -> Polarization:
        """Generate h+ and h× polarizations for a given set of waveform parameters."""
        ...

    def _validate_domain_for_polarization(self) -> None:
        if not isinstance(
            self._waveform_gen_params.domain, BaseFrequencyDomain
        ) and not isinstance(self._waveform_gen_params.domain, TimeDomain):
            raise ValueError(
                "generate_hplus_hcross: domain must be an instance of "
                "BaseFrequencyDomain or TimeDomain, "
                f"{type(self._waveform_gen_params.domain)} not supported"
            )

    def _apply_post_generation(self, polarization: Polarization) -> Polarization:
        # Domain-specific waveform transform (e.g., decimation for MFD).
        # dingo-gw domains may not have this method, so check first.
        waveform_transform = getattr(
            self._waveform_gen_params.domain, "waveform_transform", None
        )
        if waveform_transform is not None:
            polarization = waveform_transform(polarization)
        if self._waveform_gen_params.transform is not None:
            _logger.debug(
                f"applying transform {self._waveform_gen_params.transform} to polarization"
            )
            return self._waveform_gen_params.transform(polarization)
        return polarization

    def _log_generation_start(
        self,
        waveform_parameters: WaveformParameters,
        function_name: str,
    ) -> None:
        if _logger.isEnabledFor(logging.INFO):
            _logger.info(
                waveform_parameters.to_table(
                    f"starting to generate waveforms for approximant "
                    f"{self._waveform_gen_params.approximant} "
                    f"using function {function_name} "
                    f"and waveform parameters "
                    f"(f_ref={self._waveform_gen_params.f_ref}):"
                )
            )

    def _generate_hplus_hcross_m_checks(
        self, waveform_parameters: WaveformParameters
    ) -> None:
        if not isinstance(self._waveform_gen_params.domain, BaseFrequencyDomain):
            raise ValueError(
                "generate_hplus_hcross_m: only frequency-domain types are supported "
                f"({type(self._waveform_gen_params.domain)} not supported)"
            )
        required_keys = ("phase",)
        for rq in required_keys:
            if getattr(waveform_parameters, rq, None) is None:
                raise ValueError(
                    f"generate_hplus_hcross_m: the parameters must specify a value for '{rq}'"
                )


class LALSimWaveformGenerator(NewWaveformGenerator):
    """
    Waveform generator using the LALSimulation backend.

    Supports any LALSimulation approximant via SimInspiralFD/SimInspiralTD.
    Does not support mode-separated generation (generate_hplus_hcross_m).
    """

    def generate_hplus_hcross(
        self, waveform_parameters: WaveformParameters
    ) -> Polarization:
        self._validate_domain_for_polarization()

        if isinstance(self._waveform_gen_params.domain, BaseFrequencyDomain):
            polarization_method = polarization_functions.lalsim_inspiral_FD
        elif isinstance(self._waveform_gen_params.domain, TimeDomain):
            polarization_method = polarization_functions.lalsim_inspiral_TD
        else:
            raise ValueError(
                f"LALSimWaveformGenerator: unsupported domain type "
                f"{type(self._waveform_gen_params.domain)}"
            )

        self._log_generation_start(waveform_parameters, polarization_method.__name__)

        polarization = polarization_method(
            self._waveform_gen_params, waveform_parameters
        )
        return self._apply_post_generation(polarization)


class SEOBNRv4PHMWaveformGenerator(LALSimWaveformGenerator):
    """
    Waveform generator for SEOBNRv4PHM.

    Inherits FD/TD generation from LALSimWaveformGenerator.
    Adds mode-separated generation via SimInspiralChooseTDModes.
    """

    def generate_hplus_hcross_m(
        self, waveform_parameters: WaveformParameters
    ) -> Dict[Mode, Polarization]:
        self._generate_hplus_hcross_m_checks(waveform_parameters)

        modes_function = polarization_modes_functions.lalsim_inspiral_choose_TD_modes

        self._log_generation_start(waveform_parameters, modes_function.__name__)

        polarization_modes: Dict[Mode, Polarization] = modes_function(
            self._waveform_gen_params, waveform_parameters
        )

        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                f"generated polarizations:\n{polarizations_to_table(polarization_modes)}"
            )

        return polarization_modes


class IMRPhenomXPHMWaveformGenerator(LALSimWaveformGenerator):
    """
    Waveform generator for IMRPhenomXPHM.

    Inherits FD/TD generation from LALSimWaveformGenerator.
    Adds mode-separated generation via SimInspiralChooseFDModes.
    """

    def generate_hplus_hcross_m(
        self, waveform_parameters: WaveformParameters
    ) -> Dict[Mode, Polarization]:
        self._generate_hplus_hcross_m_checks(waveform_parameters)

        modes_function = polarization_modes_functions.lalsim_inspiral_choose_FD_modes

        self._log_generation_start(waveform_parameters, modes_function.__name__)

        polarization_modes: Dict[Mode, Polarization] = modes_function(
            self._waveform_gen_params, waveform_parameters
        )

        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                f"generated polarizations:\n{polarizations_to_table(polarization_modes)}"
            )

        return polarization_modes


class GWSignalWaveformGenerator(NewWaveformGenerator):
    """
    Waveform generator using the GWSignal backend.

    Supports SEOBNRv5PHM, SEOBNRv5HM, and other GWSignal-compatible approximants.
    FD generation uses gwsignal GenerateFDWaveform.
    TD generation uses gwsignal GenerateTDWaveform.
    Mode-separated generation uses gwsignal GenerateTDModes with SEOBNRv5 conditioning.
    """

    def generate_hplus_hcross(
        self, waveform_parameters: WaveformParameters
    ) -> Polarization:
        self._validate_domain_for_polarization()

        if isinstance(self._waveform_gen_params.domain, BaseFrequencyDomain):
            polarization_method = polarization_functions.gwsignal_generate_FD_modes
        elif isinstance(self._waveform_gen_params.domain, TimeDomain):
            polarization_method = polarization_functions.gwsignal_generate_TD_modes
        else:
            raise ValueError(
                f"GWSignalWaveformGenerator: unsupported domain type "
                f"{type(self._waveform_gen_params.domain)}"
            )

        self._log_generation_start(waveform_parameters, polarization_method.__name__)

        polarization = polarization_method(
            self._waveform_gen_params, waveform_parameters
        )
        return self._apply_post_generation(polarization)

    def generate_hplus_hcross_m(
        self, waveform_parameters: WaveformParameters
    ) -> Dict[Mode, Polarization]:
        self._generate_hplus_hcross_m_checks(waveform_parameters)

        modes_function = polarization_modes_functions.gwsignal_generate_TD_modes_SEOBNRv5

        self._log_generation_start(waveform_parameters, modes_function.__name__)

        polarization_modes: Dict[Mode, Polarization] = modes_function(
            self._waveform_gen_params, waveform_parameters
        )

        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                f"generated polarizations:\n{polarizations_to_table(polarization_modes)}"
            )

        return polarization_modes


class RandomWaveformGenerator(NewWaveformGenerator):
    """
    Waveform generator for RandomApproximant.

    Generates synthetic waveforms without calling any external library.
    Supports both generate_hplus_hcross and generate_hplus_hcross_m.
    """

    def generate_hplus_hcross(
        self, waveform_parameters: WaveformParameters
    ) -> Polarization:
        self._validate_domain_for_polarization()

        polarization_method = polarization_functions.random_inspiral_FD

        self._log_generation_start(waveform_parameters, polarization_method.__name__)

        polarization = polarization_method(
            self._waveform_gen_params, waveform_parameters
        )
        return self._apply_post_generation(polarization)

    def generate_hplus_hcross_m(
        self, waveform_parameters: WaveformParameters
    ) -> Dict[Mode, Polarization]:
        self._generate_hplus_hcross_m_checks(waveform_parameters)

        modes_function = polarization_modes_functions.random_inspiral_FD_modes

        self._log_generation_start(waveform_parameters, modes_function.__name__)

        polarization_modes: Dict[Mode, Polarization] = modes_function(
            self._waveform_gen_params, waveform_parameters
        )

        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                f"generated polarizations:\n{polarizations_to_table(polarization_modes)}"
            )

        return polarization_modes


# Mapping from approximant string to NewWaveformGenerator subclass.
_APPROXIMANT_CLASS_MAP: Dict[str, Type[NewWaveformGenerator]] = {
    "RandomApproximant": RandomWaveformGenerator,
    "SEOBNRv4PHM": SEOBNRv4PHMWaveformGenerator,
    "IMRPhenomXPHM": IMRPhenomXPHMWaveformGenerator,
    "SEOBNRv5PHM": GWSignalWaveformGenerator,
    "SEOBNRv5HM": GWSignalWaveformGenerator,
    "SEOBNRv5EHM": GWSignalWaveformGenerator,
}


def _get_waveform_generator_class(
    approximant: Approximant,
) -> Type[NewWaveformGenerator]:
    """Return the appropriate NewWaveformGenerator subclass for the given approximant."""
    return _APPROXIMANT_CLASS_MAP.get(str(approximant), LALSimWaveformGenerator)


def build_waveform_generator(
    params: Union[Dict, str, Path],
    domain: Optional[Domain] = None,
) -> NewWaveformGenerator:
    """
    Factory function to build a NewWaveformGenerator from various input types.

    Parameters
    ----------
    params : Union[Dict, str, Path]
        Either:
        - A dict with keys "approximant", "f_ref" (+ optional "domain", "waveform_generator")
        - A file path (str or Path) to a JSON/TOML/YAML config file
    domain : Optional[Domain]
        If provided, used as the domain. Required when params is a simple dict
        without a "domain" key.

    Returns
    -------
    NewWaveformGenerator
        An appropriate WaveformGenerator subclass instance.
    """
    if isinstance(params, (str, Path)):
        params = read_file(Path(params))

    if not isinstance(params, dict):
        raise TypeError(f"Expected dict, str, or Path, got {type(params)}")

    # If params has nested structure with "domain" and "waveform_generator" keys
    if "domain" in params and "waveform_generator" in params and domain is None:
        domain_params = params["domain"]
        domain = build_domain(domain_params)
        wfg_params = params["waveform_generator"]
        return _build_from_dict(wfg_params, domain)

    if domain is None:
        raise ValueError(
            "Either provide domain as argument, or include 'domain' and "
            "'waveform_generator' keys in the params dict."
        )

    return _build_from_dict(params, domain)


def _build_from_dict(params: Dict, domain: Domain) -> NewWaveformGenerator:
    """Build a NewWaveformGenerator from a flat dict + domain."""
    for key in ("approximant", "f_ref"):
        if key not in params:
            raise ValueError(
                f"the key '{key}' is required to build a waveform generator from a dictionary"
            )

    approximant = Approximant(str(params["approximant"]))
    f_ref = float(params["f_ref"])

    spin_conversion_phase = params.get("spin_conversion_phase", None)
    if spin_conversion_phase is not None:
        spin_conversion_phase = float(spin_conversion_phase)

    f_start = params.get("f_start", None)
    if f_start is not None:
        f_start = float(f_start)

    mode_list = params.get("mode_list", None)

    transform = params.get("transform", None)
    if transform is not None:
        transform = str(transform)

    cls = _get_waveform_generator_class(approximant)

    return cls(
        approximant,
        domain,
        f_ref,
        f_start=f_start,
        spin_conversion_phase=spin_conversion_phase,
        mode_list=mode_list,
        transform=transform,
    )
