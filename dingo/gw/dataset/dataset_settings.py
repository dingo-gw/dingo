"""Settings dataclass for waveform dataset generation."""

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, Optional

from dingo.gw.domains import DomainParameters, build_domain
from dingo.gw.prior import IntrinsicPriors
from .compression_settings import CompressionSettings, SVDSettings
from .waveform_generator_settings import WaveformGeneratorSettings


@dataclass
class DatasetSettings:
    """
    Configuration for waveform dataset generation.

    Attributes
    ----------
    domain
        Domain parameters as DomainParameters dataclass.
    waveform_generator
        Waveform generator configuration.
    intrinsic_prior
        Prior configuration for intrinsic parameters.
    num_samples
        Number of waveforms to generate.
    compression
        Optional compression settings (None for no compression).
    """

    domain: DomainParameters
    waveform_generator: WaveformGeneratorSettings
    intrinsic_prior: IntrinsicPriors
    num_samples: int
    compression: Optional[CompressionSettings] = None

    def __post_init__(self):
        if isinstance(self.waveform_generator, dict):
            self.waveform_generator = WaveformGeneratorSettings(
                approximant=self.waveform_generator.get("approximant"),
                f_ref=self.waveform_generator.get("f_ref"),
                spin_conversion_phase=self.waveform_generator.get("spin_conversion_phase"),
                f_start=self.waveform_generator.get("f_start"),
            )
        if isinstance(self.intrinsic_prior, dict):
            self.intrinsic_prior = IntrinsicPriors(**self.intrinsic_prior)
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")

    def validate(self):
        """No-op; validation happens in __post_init__."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        if is_dataclass(self.domain):
            domain_dict = asdict(self.domain)
        else:
            domain_dict = self.domain

        result = {
            "domain": domain_dict,
            "waveform_generator": self.waveform_generator.to_dict(),
            "intrinsic_prior": asdict(self.intrinsic_prior),
            "num_samples": self.num_samples,
        }
        if self.compression is not None:
            result["compression"] = asdict(self.compression)
        return result

    @classmethod
    def from_dict(cls, settings_dict: Dict[str, Any]) -> "DatasetSettings":
        domain = build_domain(settings_dict["domain"])
        domain_params = domain.get_parameters()

        wfg_dict = settings_dict["waveform_generator"]
        wfg_settings = WaveformGeneratorSettings(
            approximant=wfg_dict["approximant"],
            f_ref=wfg_dict["f_ref"],
            spin_conversion_phase=wfg_dict.get("spin_conversion_phase"),
            f_start=wfg_dict.get("f_start"),
        )

        intrinsic_prior = IntrinsicPriors(**settings_dict["intrinsic_prior"])

        compression = None
        if "compression" in settings_dict and settings_dict["compression"] is not None:
            comp_dict = settings_dict["compression"]
            svd_settings = None
            if "svd" in comp_dict:
                svd_dict = comp_dict["svd"]
                svd_settings = SVDSettings(
                    size=svd_dict["size"],
                    num_training_samples=svd_dict["num_training_samples"],
                    num_validation_samples=svd_dict.get("num_validation_samples", 0),
                    file=svd_dict.get("file"),
                )
            compression = CompressionSettings(
                svd=svd_settings,
                whitening=comp_dict.get("whitening"),
            )

        return cls(
            domain=domain_params,
            waveform_generator=wfg_settings,
            intrinsic_prior=intrinsic_prior,
            num_samples=settings_dict["num_samples"],
            compression=compression,
        )
