from __future__ import annotations

from copy import deepcopy
from dataclasses import Field, asdict, dataclass, fields, make_dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
    Any,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from bilby.gw.prior import BBHPriorDict
from bilby.gw.conversion import (
    fill_from_fixed_priors,
    convert_to_lal_binary_black_hole_parameters,
)
from bilby.core.prior import Uniform, Sine, Cosine

import numpy as np
import warnings

from dingo.gw.logs import TableStr

if TYPE_CHECKING:
    from dingo.gw.waveform_generator.waveform_parameters import BBHWaveformParameters


def _get_bbh_waveform_parameters_class():
    """Lazy import to avoid circular import."""
    from dingo.gw.waveform_generator.waveform_parameters import BBHWaveformParameters

    return BBHWaveformParameters

# Silence INFO and WARNING messages from bilby
import logging

logging.getLogger("bilby").setLevel("ERROR")


class BBHExtrinsicPriorDict(BBHPriorDict):
    """
    This class is the same as BBHPriorDict except that it does not require mass parameters.

    It also contains a method for estimating the standardization parameters.

    TODO:
        * Add support for zenith/azimuth
        * Defaults?
    """

    def default_conversion_function(self, sample):
        out_sample = fill_from_fixed_priors(sample, self)
        out_sample, _ = convert_to_lal_binary_black_hole_parameters(out_sample)

        # The previous call sometimes adds phi_jl, phi_12 parameters. These are
        # not needed so they can be deleted.
        if "phi_jl" in out_sample.keys():
            del out_sample["phi_jl"]
        if "phi_12" in out_sample.keys():
            del out_sample["phi_12"]

        return out_sample

    def mean_std(self, keys=([]), sample_size=50000, force_numerical=False):
        """
        Calculate the mean and standard deviation over the prior.

        Parameters
        ----------
        keys: list(str)
            A list of desired parameter names
        sample_size: int
            For nonanalytic priors, number of samples to use to estimate the
            result.
        force_numerical: bool (False)
            Whether to force a numerical estimation of result, even when
            analytic results are available (useful for testing)

        Returns dictionaries for the means and standard deviations.

        TODO: Fix for constrained priors. Shouldn't be an issue for extrinsic parameters.
        """
        mean = {}
        std = {}

        if not force_numerical:
            # First try to calculate analytically (works for standard priors)
            estimation_keys = []
            for key in keys:
                p = self[key]
                # A few analytic cases
                if isinstance(p, Uniform):
                    mean[key] = (p.maximum + p.minimum) / 2.0
                    std[key] = np.sqrt((p.maximum - p.minimum) ** 2.0 / 12.0).item()
                elif isinstance(p, Sine) and p.minimum == 0.0 and p.maximum == np.pi:
                    mean[key] = np.pi / 2.0
                    std[key] = np.sqrt(0.25 * (np.pi**2) - 2).item()
                elif (
                    isinstance(p, Cosine)
                    and p.minimum == -np.pi / 2
                    and p.maximum == np.pi / 2
                ):
                    mean[key] = 0.0
                    std[key] = np.sqrt(0.25 * (np.pi**2) - 2).item()
                else:
                    estimation_keys.append(key)
        else:
            estimation_keys = keys

        # For remaining parameters, estimate numerically
        if len(estimation_keys) > 0:
            samples = self.sample_subset(keys, size=sample_size)
            samples = self.default_conversion_function(samples)
            for key in estimation_keys:
                if key in samples.keys():
                    mean[key] = np.mean(samples[key]).item()
                    std[key] = np.std(samples[key]).item()

        return mean, std


default_extrinsic_dict = {
    "dec": "bilby.core.prior.Cosine(minimum=-np.pi/2, maximum=np.pi/2, name='dec')",
    "ra": 'bilby.core.prior.Uniform(minimum=0., maximum=2*np.pi, boundary="periodic", name="ra")',
    "geocent_time": "bilby.core.prior.Uniform(minimum=-0.1, maximum=0.1, name='geocent_time')",
    "psi": 'bilby.core.prior.Uniform(minimum=0.0, maximum=np.pi, boundary="periodic", name="psi")',
    "luminosity_distance": "bilby.core.prior.Uniform(minimum=100.0, maximum=6000.0, name='luminosity_distance')",
}

default_intrinsic_dict = {
    "mass_1": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0, name='mass_1')",
    "mass_2": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0, name='mass_2')",
    "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.125, maximum=1.0, name='mass_ratio')",
    "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass(minimum=25.0, maximum=100.0, name='chirp_mass')",
    "luminosity_distance": 1000.0,
    "theta_jn": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi, name='theta_jn')",
    "phase": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic", name="phase")',
    "a_1": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99, name='a_1')",
    "a_2": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99, name='a_2')",
    "tilt_1": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi, name='tilt_1')",
    "tilt_2": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi, name='tilt_2')",
    "phi_12": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic", name="phi_12")',
    "phi_jl": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic", name="phi_jl")',
    "geocent_time": 0.0,
}

default_inference_parameters = [
    "chirp_mass",
    "mass_ratio",
    "phase",
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "theta_jn",
    "luminosity_distance",
    "geocent_time",
    "ra",
    "dec",
    "psi",
]


def build_prior_with_defaults(prior_settings: Dict[str, str]):
    """
    Generate BBHPriorDict based on dictionary of prior settings,
    allowing for default values.

    Parameters
    ----------
    prior_settings: Dict
        A dictionary containing prior definitions for intrinsic parameters
        Allowed values for each parameter are:
            * 'default' to use a default prior
            * a string for a custom prior, e.g.,
               "Uniform(minimum=10.0, maximum=80.0, name=None, latex_label=None, unit=None, boundary=None)"

    Depending on the particular prior choices the dimensionality of a
    parameter sample obtained from the returned GWPriorDict will vary.
    """

    full_prior_settings = deepcopy(prior_settings)
    for k, v in prior_settings.items():
        if v == "default":
            full_prior_settings[k] = default_intrinsic_dict[k]

    return BBHPriorDict(full_prior_settings)


def split_off_extrinsic_parameters(theta):
    """
    Split theta into intrinsic and extrinsic parameters.

    Parameters
    ----------
    theta: dict
        BBH parameters. Includes intrinsic parameters to be passed to waveform
        generator, and extrinsic parameters for detector projection.

    Returns
    -------
    theta_intrinsic: dict
        BBH intrinsic parameters.
    theta_extrinsic: dict
        BBH extrinsic parameters (includes calibration parameters).
    """
    extrinsic_parameters = ["geocent_time", "luminosity_distance", "ra", "dec", "psi"]
    theta_intrinsic = {}
    theta_extrinsic = {}
    for k, v in theta.items():
        if k in extrinsic_parameters or "recalib" in k:
            theta_extrinsic[k] = v
        else:
            theta_intrinsic[k] = v
    # set fiducial values for time and distance
    theta_intrinsic["geocent_time"] = 0
    theta_intrinsic["luminosity_distance"] = 100
    return theta_intrinsic, theta_extrinsic


# ---------------------------------------------------------------------------
# New-style dataclass-based priors (ported from dingo-waveform)
# ---------------------------------------------------------------------------


def _get_prior_dict(
    prior_instance: Union["ExtrinsicPriors", "IntrinsicPriors"],
) -> Dict[str, Union[str, float]]:
    d = {}
    default_priors = prior_instance.default_priors()
    for k, v in asdict(prior_instance).items():
        if v is not None:
            v_ = v if v != "default" else default_priors[k]
            d[k] = v_
    return d


@dataclass
class ExtrinsicPriors(TableStr):
    dec: Optional[Union[str, float]] = None
    ra: Optional[Union[str, float]] = None
    geocent_time: Optional[Union[str, float]] = None
    psi: Optional[Union[str, float]] = None
    luminosity_distance: Optional[Union[str, float]] = None

    @staticmethod
    def default_priors() -> Dict[str, Union[str, float]]:
        return {
            "dec": "bilby.core.prior.Cosine(minimum=-np.pi/2, maximum=np.pi/2)",
            "ra": 'bilby.core.prior.Uniform(minimum=0., maximum=2*np.pi, boundary="periodic")',
            "geocent_time": "bilby.core.prior.Uniform(minimum=-0.1, maximum=0.1)",
            "psi": 'bilby.core.prior.Uniform(minimum=0.0, maximum=np.pi, boundary="periodic")',
            "luminosity_distance": "bilby.core.prior.Uniform(minimum=100.0, maximum=6000.0)",
        }

    def mean_std(
        self, keys: List[str], sample_size=50000, force_numerical=False
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        bbh_prior_dict = BBHExtrinsicPriorDict(asdict(self))
        return bbh_prior_dict.mean_std(
            keys, sample_size=sample_size, force_numerical=force_numerical
        )

    def sample(self) -> BBHWaveformParameters:
        return self.samples(1)[0]

    def samples(self, nb_samples) -> List[BBHWaveformParameters]:
        _BBHWfP = _get_bbh_waveform_parameters_class()
        bbh_prior_dict = BBHExtrinsicPriorDict(_get_prior_dict(self))
        return [
            _BBHWfP(**bbh_prior_dict.sample()) for _ in range(nb_samples)
        ]


@dataclass
class IntrinsicPriors(TableStr):
    mass_1: Optional[Union[str, float]] = None
    mass_2: Optional[Union[str, float]] = None
    mass_ratio: Optional[Union[str, float]] = None
    chirp_mass: Optional[Union[str, float]] = None
    luminosity_distance: Optional[Union[str, float]] = None
    theta_jn: Optional[Union[str, float]] = None
    phase: Optional[Union[str, float]] = None
    a_1: Optional[Union[str, float]] = None
    a_2: Optional[Union[str, float]] = None
    tilt_1: Optional[Union[str, float]] = None
    tilt_2: Optional[Union[str, float]] = None
    phi_12: Optional[Union[str, float]] = None
    phi_jl: Optional[Union[str, float]] = None
    chi_1: Optional[Union[str, float]] = None
    chi_2: Optional[Union[str, float]] = None
    geocent_time: Union[str, float] = 0.0

    @staticmethod
    def default_priors() -> Dict[str, Union[str, float]]:
        return {
            "mass_1": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0)",
            "mass_2": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0)",
            "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.125, maximum=1.0)",
            "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass(minimum=25.0, maximum=100.0)",
            "luminosity_distance": 1000.0,
            "theta_jn": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
            "phase": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
            "a_1": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99)",
            "a_2": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99)",
            "tilt_1": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
            "tilt_2": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
            "phi_12": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
            "phi_jl": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
            "chi_1": 'bilby.gw.prior.AlignedSpin(name="chi_1", a_prior=Uniform(minimum=0, maximum=0.99))',
            "chi_2": 'bilby.gw.prior.AlignedSpin(name="chi_2", a_prior=Uniform(minimum=0, maximum=0.99))',
            "geocent_time": 0.0,
        }

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "IntrinsicPriors":
        from dingo.gw.imports import read_file

        parameters = read_file(file_path)
        return cls(**parameters)

    def sample_as_dict(self) -> Dict[str, float]:
        d = _get_prior_dict(self)
        bbh_prior_dict = BBHPriorDict(d)
        return bbh_prior_dict.sample()

    def sample(self) -> BBHWaveformParameters:
        return self.samples(1)[0]

    def samples(self, nb_samples: int) -> List[BBHWaveformParameters]:
        _BBHWfP = _get_bbh_waveform_parameters_class()
        bbh_prior_dict = BBHPriorDict(_get_prior_dict(self))
        return [
            _BBHWfP(**bbh_prior_dict.sample()) for _ in range(nb_samples)
        ]


def _create_priors_dataclass() -> Type:
    extrinsic_fields = list(fields(ExtrinsicPriors))
    intrinsic_fields = list(fields(IntrinsicPriors))

    all_fields_ = extrinsic_fields + [
        f
        for f in intrinsic_fields
        if f.name not in [f_.name for f_ in extrinsic_fields]
    ]
    all_fields = [(f.name, f.type, f.default) for f in all_fields_]

    _PriorType = TypeVar("_PriorType", IntrinsicPriors, ExtrinsicPriors)

    class _PriorSampling:

        def _get_prior(self, target_type: Type[_PriorType]) -> _PriorType:
            d_ = asdict(self)
            d = {
                k: v
                for k, v in d_.items()
                if k in [f.name for f in fields(target_type)]
            }
            return target_type(**d)

        def get_intrinsic_priors(self) -> IntrinsicPriors:
            return self._get_prior(IntrinsicPriors)

        def get_extrinsic_priors(self) -> ExtrinsicPriors:
            return self._get_prior(ExtrinsicPriors)

        def sample(self) -> BBHWaveformParameters:
            return self.samples(1)[0]

        def samples(self, nb_samples: int) -> List[BBHWaveformParameters]:
            ip = self.get_intrinsic_priors()
            ep = self.get_extrinsic_priors()
            intrinsic_wfs: List[BBHWaveformParameters] = ip.samples(nb_samples)
            extrinsic_wfs: List[BBHWaveformParameters] = ep.samples(nb_samples)

            def _get_wf(
                intrinsic_wf: BBHWaveformParameters, extrinsic_wf: BBHWaveformParameters
            ):
                intrinsic_dict = asdict(intrinsic_wf)
                extrinsic_dict = asdict(extrinsic_wf)
                priors_dict = intrinsic_dict
                for k, v in extrinsic_dict.items():
                    if v is not None:
                        priors_dict[k] = v
                return _get_bbh_waveform_parameters_class()(**priors_dict)

            return [_get_wf(iw, ew) for iw, ew in zip(intrinsic_wfs, extrinsic_wfs)]

    return make_dataclass("Priors", all_fields, bases=(TableStr, _PriorSampling))


Priors = _create_priors_dataclass()
"""Dataclass combining IntrinsicPriors and ExtrinsicPriors fields."""


def prior_split(
    waveform_parameters: BBHWaveformParameters,
    intrinsic_luminosity_distance: Optional[float] = 100.0,
    intrinsic_geocent_time: Optional[float] = 0.0,
) -> Tuple[BBHWaveformParameters, BBHWaveformParameters]:
    """Split waveform parameters into intrinsic and extrinsic components."""
    intrinsic_keys = [f.name for f in fields(IntrinsicPriors)]
    extrinsic_keys = [f.name for f in fields(ExtrinsicPriors)]
    waveform_dict = asdict(waveform_parameters)
    intrinsic_dict = {k: v for k, v in waveform_dict.items() if k in intrinsic_keys}
    extrinsic_dict = {k: v for k, v in waveform_dict.items() if k in extrinsic_keys}
    if intrinsic_luminosity_distance is not None:
        intrinsic_dict["luminosity_distance"] = intrinsic_luminosity_distance
    if intrinsic_geocent_time is not None:
        intrinsic_dict["geocent_time"] = intrinsic_geocent_time
    _BBHWfP = _get_bbh_waveform_parameters_class()
    return _BBHWfP(**intrinsic_dict), _BBHWfP(**extrinsic_dict)


def new_build_prior_with_defaults(
    prior_settings: Union[IntrinsicPriors, Mapping[str, Union[str, float]]],
) -> BBHPriorDict:
    """
    Generate BBHPriorDict based on prior settings, allowing for default values.

    This is the new-style version that accepts IntrinsicPriors dataclasses
    as well as plain dicts.
    """
    prior_settings_: IntrinsicPriors
    if isinstance(prior_settings, dict):
        prior_settings_ = IntrinsicPriors(**prior_settings)
    else:
        prior_settings_ = cast(IntrinsicPriors, prior_settings)
    return BBHPriorDict(_get_prior_dict(prior_settings_))
