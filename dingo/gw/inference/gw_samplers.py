from typing import Union, Protocol

import warnings

import numpy as np
import pandas as pd
from astropy.time import Time
from bilby.core.prior import PriorDict, DeltaFunction, Constraint
from bilby.gw.detector import InterferometerList
from torchvision.transforms import Compose

from dingo.core.samplers import Sampler, GNPESampler
from dingo.core.transforms import GetItem, RenameKey
from dingo.gw.domains import (
    MultibandedFrequencyDomain,
    build_domain_from_model_metadata,
    UniformFrequencyDomain,
    Domain,
)
from dingo.gw.domains import build_domain
from dingo.gw.gwutils import get_extrinsic_prior_dict
from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.result import Result
from dingo.gw.transforms import (
    WhitenAndScaleStrain,
    RepackageStrainsAndASDS,
    ToTorch,
    SelectStandardizeRepackageParameters,
    GNPECoalescenceTimes,
    TimeShiftStrain,
    GNPEBase,
    PostCorrectGeocentTime,
    CopyToExtrinsicParameters,
    GetDetectorTimes,
    DecimateWaveformsAndASDS,
    MaskDataForFrequencyRangeUpdate,
    StrainTokenization,
    MaskTokensForFrequencyRangeUpdate,
    UnpackDict,
)


class SamplerProtocol(Protocol):
    base_model_metadata: dict

    def _initialize_transforms(self) -> None: ...


class _GWMixinProtocol(SamplerProtocol):
    detectors: list[str]
    domain: Domain
    random_strain_cropping: dict


class GWSamplerMixin(object):
    """
    Mixin class designed to add gravitational wave functionality to Sampler classes:
        * builder for data domain
        * correction for fixed detector locations during training (t_ref)
    """

    def __init__(self: SamplerProtocol, **kwargs):
        """
        Parameters
        ----------
        kwargs
            Keyword arguments that are forwarded to the superclass.
        """
        # Has to be specified before init, because the information is required in _initialize_transforms()
        self._minimum_frequency = None
        self._maximum_frequency = None
        self._detectors = None
        self._psd_notch_dict = None
        super().__init__(**kwargs)
        self.t_ref = self.base_model_metadata["train_settings"]["data"]["ref_time"]
        self._pesummary_package = "gw"
        self._result_class = Result

    @property
    def detectors(self: SamplerProtocol):
        if self._detectors is None:
            self._detectors = self.base_model_metadata["train_settings"]["data"][
                "detectors"
            ]
        return self._detectors

    @detectors.setter
    def detectors(self: _GWMixinProtocol, value: list[str]):
        check_detector_update(self.base_model_metadata, value)
        self._detectors = value
        self._initialize_transforms()

    @property
    def random_strain_cropping(self: SamplerProtocol):
        return self.base_model_metadata["train_settings"]["data"].get(
            "random_strain_cropping"
        )

    @property
    def minimum_frequency(self) -> float | dict[str, float]:
        if self._minimum_frequency is not None:
            return self._minimum_frequency
        else:
            return self.domain.f_min

    @minimum_frequency.setter
    def minimum_frequency(self: _GWMixinProtocol, value: dict[str, float] | float):
        if isinstance(self.domain, MultibandedFrequencyDomain):
            domain = self.domain.base_domain
        elif isinstance(self.domain, UniformFrequencyDomain):
            domain = self.domain
        else:
            raise ValueError("Frequency updates only possible for frequency domains.")
        _validate_frequency_bound(
            value,
            "minimum_frequency",
            domain,
            self.base_model_metadata["train_settings"]["data"],
        )
        self._minimum_frequency = value
        self._initialize_transforms()

    @property
    def maximum_frequency(self) -> float | dict[str, float]:
        if self._maximum_frequency is not None:
            return self._maximum_frequency
        else:
            return self.domain.f_max

    @maximum_frequency.setter
    def maximum_frequency(self: _GWMixinProtocol, value: dict[str, float] | float):
        if isinstance(self.domain, MultibandedFrequencyDomain):
            domain = self.domain.base_domain
        elif isinstance(self.domain, UniformFrequencyDomain):
            domain = self.domain
        else:
            raise ValueError("Frequency updates only possible for frequency domains.")
        _validate_frequency_bound(
            value,
            "maximum_frequency",
            domain,
            self.base_model_metadata["train_settings"]["data"],
        )
        self._maximum_frequency = value
        self._initialize_transforms()

    @property
    def frequency_updates(self) -> bool:
        def normalize(val):
            if isinstance(val, dict):
                return set(val.values())
            return {val}

        return normalize(self.minimum_frequency) != {self.domain.f_min} or normalize(
            self.maximum_frequency
        ) != {self.domain.f_max}

    @property
    def psd_notch_dict(self) -> dict | None:
        return getattr(self, "_psd_notch_dict", None)

    @psd_notch_dict.setter
    def psd_notch_dict(self, value: dict | None):
        if value is not None:
            if isinstance(self.domain, MultibandedFrequencyDomain):
                domain = self.domain.base_domain
            elif isinstance(self.domain, UniformFrequencyDomain):
                domain = self.domain
            else:
                raise ValueError("psd_notch_dict requires a frequency domain.")
            _validate_psd_notches(
                value, domain, self.base_model_metadata["train_settings"]["data"]
            )
        self._psd_notch_dict = value
        self._initialize_transforms()

    @property
    def event_metadata(self):
        if self._event_metadata is not None:
            metadata = self._event_metadata.copy()
        else:
            metadata = {}
        metadata["minimum_frequency"] = self.minimum_frequency
        metadata["maximum_frequency"] = self.maximum_frequency
        metadata["detectors"] = self.detectors
        if self.psd_notch_dict is not None:
            metadata["psd_notch_dict"] = self.psd_notch_dict
        return metadata

    @event_metadata.setter
    def event_metadata(self, value):
        if value is not None:
            value = value.copy()
            # Process detectors first so that frequency validation (which uses
            # self.detectors) already reflects the event's detector subset.
            if "detectors" in value and value["detectors"] is not None:
                self.detectors = value.pop("detectors")
            if "minimum_frequency" in value:
                self.minimum_frequency = value.pop("minimum_frequency")
            if "maximum_frequency" in value:
                self.maximum_frequency = value.pop("maximum_frequency")
            if "psd_notch_dict" in value:
                self.psd_notch_dict = value.pop("psd_notch_dict")
        self._event_metadata = value

    def _build_domain(self: Sampler):
        """
        Construct the domain object based on model metadata.

        Called by __init__() immediately after _build_prior().
        """
        self.domain = build_domain(
            self.base_model_metadata["dataset_settings"]["domain"]
        )

        data_settings = self.base_model_metadata["train_settings"]["data"]
        if "domain_update" in data_settings:
            self.domain.update(data_settings["domain_update"])

    def _correct_reference_time(
        self: Sampler, samples: Union[dict, pd.DataFrame], inverse: bool = False
    ):
        """
        Correct the sky position of an event based on the reference time of the model.
        This is necessary since the model was trained with with fixed detector (reference)
        positions. This transforms the right ascension based on the e difference between
        the time of the event and t_ref.

        The correction is only applied if the event time can be found in self.metadata[
        'event'].

        This method modifies the samples in place.

        Parameters
        ----------
        samples : dict or pd.DataFrame
        inverse : bool, default True
            Whether to apply instead the inverse transformation. This is used prior to
            calculating the log_prob.
        """
        if self.event_metadata is not None:
            t_event = self.event_metadata.get("time_event")
            if t_event is not None and t_event != self.t_ref and "ra" in samples:
                ra = samples["ra"]
                time_reference = Time(self.t_ref, format="gps", scale="utc")
                time_event = Time(t_event, format="gps", scale="utc")
                longitude_event = time_event.sidereal_time("apparent", "greenwich")
                longitude_reference = time_reference.sidereal_time(
                    "apparent", "greenwich"
                )
                delta_longitude = longitude_event - longitude_reference
                ra_correction = delta_longitude.rad
                if not inverse:
                    samples["ra"] = (ra + ra_correction) % (2 * np.pi)
                else:
                    samples["ra"] = (ra - ra_correction) % (2 * np.pi)

    def _post_process(self, samples: Union[dict, pd.DataFrame], inverse: bool = False):
        """
        Post-processing of parameter samples.
        * Add any fixed parameters from the prior.
        * Correct the sky position for a potentially fixed reference time.
          (see self._correct_reference_time)

        This method modifies the samples in place.

        Parameters
        ----------
        samples : dict or pd.DataFrame
        inverse : bool, default True
            Whether to apply instead the inverse transformation. This is used prior to
            calculating the log_prob.
        """
        intrinsic_prior = self.metadata["dataset_settings"]["intrinsic_prior"]
        extrinsic_prior = get_extrinsic_prior_dict(
            self.metadata["train_settings"]["data"]["extrinsic_prior"]
        )
        prior = build_prior_with_defaults({**intrinsic_prior, **extrinsic_prior})

        if not inverse:
            # Add fixed parameters from prior.
            num_samples = len(samples[list(samples.keys())[0]])
            for k, p in prior.items():
                if isinstance(p, DeltaFunction) and k not in samples:
                    v = p.peak
                    print(f"Adding fixed parameter {k} = {v} from prior.")
                    samples[k] = p.peak * np.ones(num_samples)
        else:
            # Drop non-inference parameters from samples.
            # NOTE: Important to drop "log_prob" in particular before running
            # Sampler.log_prob(), otherwise log probabilities are added.
            drop_parameters = [
                k for k in samples.keys() if k not in self.inference_parameters
            ]
            if isinstance(samples, pd.DataFrame):
                samples.drop(columns=drop_parameters, inplace=True, errors="ignore")
            elif isinstance(samples, dict):
                for k in drop_parameters:
                    samples.pop(k, None)

        if not self.unconditional_model:
            self._correct_reference_time(samples, inverse)


class GWSampler(GWSamplerMixin, Sampler):
    """
    Sampler for gravitational-wave inference using neural posterior estimation.
    Augments the base class by defining transform_pre and transform_post to prepare
    data for the inference network.

    transform_pre :
        * Decimates data (if necessary and using MultibandedFrequencyDomain).
        * Whitens strain.
        * Repackages strain data and the inverse ASDs (suitably scaled) into a torch
          tensor.

    transform_post :
        * Extract the desired inference parameters from the network output (
          array-like), de-standardize them, and repackage as a dict.

    Also mixes in GW functionality for building the domain and correcting the reference
    time.

    Allows for conditional and unconditional models, and draws samples from the model
    based on (optional) context data.

    This is intended for use either as a standalone sampler, or as a sampler producing
    initial sample points for a GNPE sampler.
    """

    def _initialize_transforms(self):
        # preprocessing transforms:
        transform_pre = []
        #   * in case of MultibandedFrequencyDomain, decimate data from base domain
        if isinstance(self.domain, MultibandedFrequencyDomain):
            transform_pre.append(
                DecimateWaveformsAndASDS(self.domain, decimation_mode="whitened")
            )

        #   * whiten and scale strain (since the inference network expects standardized
        #   data)
        transform_pre.append(WhitenAndScaleStrain(self.domain.noise_std))
        tok = self.metadata["train_settings"]["data"].get("tokenization")
        if self.frequency_updates and not tok:
            # * update frequency range
            # Needs to happen before RepackageStrainsAndASDs since we might need to
            # apply detectors specific frequency updates. For tokenized models,
            # we do not apply bin-level masking since it is inert ( every unmasked
            # token lies fully inside the requested range).
            transform_pre.append(
                MaskDataForFrequencyRangeUpdate(
                    domain=self.domain,
                    minimum_frequency=self.minimum_frequency,
                    maximum_frequency=self.maximum_frequency,
                )
            )
        #   * repackage strains and asds from dicts to an array
        #   * optionally tokenize strain (transformer embedding network only)
        #   * convert array(s) to torch tensor(s) on the correct device
        #   * extract waveform (and position, drop_token_mask for transformer)
        # Use base metadata so that unconditional samplers still know how to
        # transform data, since this transform is used by the GNPE sampler as well.
        transform_pre.append(
            RepackageStrainsAndASDS(
                ifos=self.detectors,
                first_index=self.domain.min_idx,
            )
        )

        if tok:
            # StrainTokenization operates on numpy arrays, so it must precede ToTorch.
            transform_pre.append(
                StrainTokenization(
                    domain=self.domain,
                    token_size=tok.get("token_size"),
                    num_tokens_per_block=tok.get("num_tokens_per_block"),
                    drop_last_token=tok.get("drop_last_token", False),
                )
            )
            if self.frequency_updates or self.psd_notch_dict:
                transform_pre.append(
                    MaskTokensForFrequencyRangeUpdate(
                        domain=self.domain,
                        detectors=self.detectors,
                        minimum_frequency=self.minimum_frequency,
                        maximum_frequency=self.maximum_frequency,
                        psd_notch_dict=self.psd_notch_dict,
                    )
                )

        transform_pre.append(ToTorch(device=self.model.device))

        if tok:
            transform_pre.append(UnpackDict(["waveform", "position", "token_mask"]))
        else:
            transform_pre.append(GetItem("waveform"))

        self.transform_pre = Compose(transform_pre)

        # postprocessing transforms:
        #   * de-standardize data and extract inference parameters
        self.transform_post = SelectStandardizeRepackageParameters(
            {"inference_parameters": self.inference_parameters},
            self.metadata["train_settings"]["data"]["standardization"],
            inverse=True,
            as_type="dict",
        )


class GWSamplerGNPE(GWSamplerMixin, GNPESampler):
    """
    Gravitational-wave GNPE sampler. It wraps a PosteriorModel and a standard Sampler for
    initialization. The former is used to generate initial samples for Gibbs sampling.

    Compared to the base class, this class implements the required transforms for
    preparing data and parameters for the network. This includes GNPE transforms,
    data processing transforms, and standardization/de-standardization of parameters.

    A GNPE network is conditioned on additional "proxy" context theta^, i.e.,

    p(theta | theta^, d)

    The theta^ depend on theta via a fixed kernel p(theta^ | theta). Combining these
    known distributions, this class uses Gibbs sampling to draw samples from the joint
    distribution,

    p(theta, theta^ | d)

    The advantage of this approach is that we are allowed to perform any transformation of
    d that depends on theta^. In particular, we can use this freedom to simplify the
    data, e.g., by aligning data to have merger times = 0 in each detector. The merger
    times are unknown quantities that must be inferred jointly with all other
    parameters, and GNPE provides a means to do this iteratively. See
    https://arxiv.org/abs/2111.13139 for additional details.

    Gibbs sampling breaks access to the probability density, so this must be recovered
    through other means. One way is to train an unconditional flow to represent p(theta^
    | d) for fixed d based on the samples produced through the GNPE Gibbs sampling.
    Starting from these, a single Gibbs iteration gives theta from the GNPE network,
    along with the probability density in the joint space. This is implemented in
    GNPESampler provided the init_sampler provides proxies directly and num_iterations
    = 1.

    Attributes (beyond those of Sampler)
    ------------------------------------
    init_sampler : Sampler
        Used for providing initial samples for Gibbs sampling.
    num_iterations : int
        Number of Gibbs iterations to perform.
    iteration_tracker : IterationTracker
        **not set up**
    remove_init_outliers : float
        **not set up**
    """

    @property
    def minimum_frequency(self) -> float | dict[str, float]:
        if self.init_sampler is not None:
            return self.init_sampler.minimum_frequency
        else:
            raise AttributeError(
                "init_sampler not set. Cannot access minimum frequency."
            )

    @minimum_frequency.setter
    def minimum_frequency(self, value):
        if self.init_sampler is not None:
            self.init_sampler.minimum_frequency = value
        else:
            raise AttributeError(
                "init_sampler not set. Cannot update minimum frequency."
            )

    @property
    def maximum_frequency(self) -> float | dict[str, float]:
        if self.init_sampler is not None:
            return self.init_sampler.maximum_frequency
        else:
            raise AttributeError(
                "init_sampler not set. Cannot access maximum frequency."
            )

    @maximum_frequency.setter
    def maximum_frequency(self, value):
        if self.init_sampler is not None:
            self.init_sampler.maximum_frequency = value
        else:
            raise AttributeError(
                "init_sampler not set. Cannot update maximum frequency."
            )

    def _initialize_transforms(self):
        """
        Builds the transforms that are used in the GNPE loop.
        """
        data_settings = self.metadata["train_settings"]["data"]
        ifo_list = InterferometerList(data_settings["detectors"])

        gnpe_time_settings = data_settings.get("gnpe_time_shifts")
        gnpe_chirp_settings = data_settings.get("gnpe_chirp")
        gnpe_phase_settings = data_settings.get("gnpe_phase")
        if (
            not gnpe_time_settings
            and not gnpe_chirp_settings
            and not gnpe_phase_settings
        ):
            raise KeyError(
                "GNPE inference requires network trained for either chirp mass, "
                "coalescence time, or phase GNPE."
            )

        # transforms for gnpe loop, to be applied prior to sampling step:
        #   * reset the sample (e.g., clone non-gnpe transformed waveform)
        #   * blurring detector times to obtain gnpe proxies
        #   * shifting the strain by - gnpe proxies
        #   * repackaging & standardizing proxies to sample['context_parameters']
        #     for conditioning of the inference network
        transform_pre = []
        transform_pre.append(RenameKey("data", "waveform"))
        if gnpe_time_settings:
            transform_pre.append(
                GNPECoalescenceTimes(
                    ifo_list,
                    gnpe_time_settings["kernel"],
                    gnpe_time_settings["exact_equiv"],
                    inference=True,
                )
            )
            transform_pre.append(TimeShiftStrain(ifo_list, self.domain))
        transform_pre.append(
            SelectStandardizeRepackageParameters(
                {"context_parameters": data_settings["context_parameters"]},
                data_settings["standardization"],
                device=self.model.device,
            )
        )
        transform_pre.append(RenameKey("waveform", "data"))

        # Extract GNPE information (list of parameters, dict of kernels) from the
        # transforms.
        self.gnpe_parameters = []
        self.gnpe_kernel = PriorDict()
        for transform in transform_pre:
            if isinstance(transform, GNPEBase):
                self.gnpe_parameters += transform.input_parameter_names
                for k, v in transform.kernel.items():
                    self.gnpe_kernel[k] = v
        print("GNPE parameters: ", self.gnpe_parameters)
        print("GNPE kernel: ", self.gnpe_kernel)

        self.transform_pre = Compose(transform_pre)

        # transforms for gnpe loop, to be applied after sampling step:
        #   * de-standardization of parameters
        #   * post correction for geocent time (required for gnpe with exact equivariance)
        #   * computation of detectortimes from parameters (required for next gnpe
        #       iteration)
        self.transform_post = Compose(
            [
                SelectStandardizeRepackageParameters(
                    {"inference_parameters": self.inference_parameters},
                    data_settings["standardization"],
                    inverse=True,
                    as_type="dict",
                ),
                PostCorrectGeocentTime(),
                CopyToExtrinsicParameters(
                    "ra", "dec", "geocent_time", "chirp_mass", "mass_ratio", "phase"
                ),
                GetDetectorTimes(ifo_list, data_settings["ref_time"]),
            ]
        )

    def _kernel_log_prob(self, samples):
        # TODO: Reimplement as a method of GNPEBase.
        if len({"chirp_mass", "mass_ratio", "phase"} & self.gnpe_kernel.keys()) > 0:
            raise NotImplementedError("kernel log_prob only implemented for time gnpe.")
        gnpe_proxies_diff = {
            k: np.array(samples[k] - samples[f"{k}_proxy"])
            for k in self.gnpe_kernel.keys()
        }
        return self.gnpe_kernel.ln_prob(gnpe_proxies_diff, axis=0)


# Functions for frequency cropping. Used by Sampler classes and dingo-pipe.


def _validate_frequency_bound(
    value: dict[str, float] | float,
    bound: str,
    domain: UniformFrequencyDomain | MultibandedFrequencyDomain,
    data_settings: dict,
):
    """
    Validate a requested minimum or maximum frequency against the model's training
    settings.

    ``value`` may be a float (applying to all detectors) or a per-detector dict
    constraining only the detectors it names; keys must be detectors the model was
    trained with. Values equal to the domain bound are always allowed. A changed
    value requires frequency flexibility from training: ``random_strain_cropping``
    and/or ``tokenization.mask_frequency_range`` are validated against their
    envelopes; a model with only ``tokenization.mask_random_tokens`` passes with a
    warning, since the contiguous masking pattern differs from the random training
    distribution.

    Parameters
    ----------
    value : dict[str, float] or float
        Requested frequency bound.
    bound : str
        "minimum_frequency" or "maximum_frequency".
    domain : UniformFrequencyDomain or MultibandedFrequencyDomain
        The model's base (uniform) domain.
    data_settings : dict
        ``train_settings["data"]`` of the model.

    Raises
    ------
    ValueError
        If the request is incompatible with the training settings.
    """
    minimum = bound == "minimum_frequency"
    domain_value = domain.f_min if minimum else domain.f_max
    model_detectors = data_settings["detectors"]

    if isinstance(value, dict):
        unknown = set(value) - set(model_detectors)
        if unknown:
            raise ValueError(
                f"{bound} names detectors {sorted(unknown)} the model was not "
                f"trained with (detectors: {model_detectors})."
            )
        values = dict(value)
    else:
        values = {d: value for d in model_detectors}

    # Hard domain bounds.
    for det, v in values.items():
        if minimum and v < domain.f_min:
            raise ValueError(f"f_min {values} < domain.f_min = {domain.f_min}.")
        if not minimum and v > domain.f_max:
            raise ValueError(f"f_max {values} > domain.f_max = {domain.f_max}.")

    changed = {d: v for d, v in values.items() if v != domain_value}
    if not changed:
        return

    crop_settings = data_settings.get("random_strain_cropping")
    tok = data_settings.get("tokenization") or {}
    range_settings = tok.get("mask_frequency_range")

    if crop_settings is None and range_settings is None:
        if "mask_random_tokens" in tok:
            warnings.warn(
                f"Updating {bound} relies on mask_random_tokens training only; the "
                f"contiguous masking pattern differs from the random training "
                f"distribution. Expect reduced importance-sampling efficiency and "
                f"check the effective sample size."
            )
            return
        raise ValueError(
            f"Model was not trained with variable frequency ranges "
            f"(no random_strain_cropping, mask_frequency_range, or "
            f"mask_random_tokens). Cannot update {bound}."
        )

    if crop_settings is not None:
        if crop_settings.get("cropping_probability", 0.0) == 0.0:
            raise ValueError(f"Cropping disabled; cannot update {bound} to {value}.")
        if not crop_settings.get("independent_detectors", True):
            effective = {d: values.get(d, domain_value) for d in model_detectors}
            if len(set(effective.values())) > 1:
                raise ValueError(
                    f"Independent frequencies per detector not enabled. All "
                    f"frequencies must match, got {bound} = {value}."
                )

    # Training envelopes, in shared vocabulary: f_min may be raised up to
    # f_min_upper, f_max lowered down to f_max_lower; an absent key means that
    # side was never cropped / cut in training.
    key = "f_min_upper" if minimum else "f_max_lower"
    for settings, source in (
        (crop_settings, "random_strain_cropping"),
        (range_settings, "tokenization.mask_frequency_range"),
    ):
        if settings is None:
            continue
        cap = settings.get(key, domain_value)
        caps = cap if isinstance(cap, dict) else {d: cap for d in model_detectors}
        for det, v in changed.items():
            if (minimum and v > caps[det]) or (not minimum and v < caps[det]):
                raise ValueError(
                    f"Requested {bound} for {det} ({v} Hz) is outside the "
                    f"training envelope ({key}={cap} Hz from {source})."
                )


def check_frequency_updates(
    model_metadata: dict,
    f_min: dict[str, float] | float | None = None,
    f_max: dict[str, float] | float | None = None,
):
    """
    Validate requested minimum / maximum frequencies against a model's metadata.

    Thin metadata-level wrapper around ``_validate_frequency_bound``, used by
    dingo_pipe at DAG-build time; see there for the accepted forms and semantics.
    """
    domain = build_domain_from_model_metadata(model_metadata, base=True)
    if not isinstance(domain, (UniformFrequencyDomain, MultibandedFrequencyDomain)):
        raise ValueError("Frequency updates only possible for frequency domains.")
    data_settings = model_metadata["train_settings"]["data"]
    if f_min is not None:
        _validate_frequency_bound(f_min, "minimum_frequency", domain, data_settings)
    if f_max is not None:
        _validate_frequency_bound(f_max, "maximum_frequency", domain, data_settings)


def _validate_psd_notches(
    psd_notch_dict: dict,
    domain: UniformFrequencyDomain | MultibandedFrequencyDomain,
    data_settings: dict,
):
    """
    Validate PSD notch intervals against the domain and the model's training settings.

    ``psd_notch_dict`` maps detectors to one ``[f_lo, f_hi]`` interval or a list of
    them. Configuration errors raise: a detector the model was not trained with, an
    empty interval, or an interval touching the domain bounds (at data generation a
    high-ASD run at an edge is taken for PSD padding, see ``detect_asd_notches``, so
    the frequency bound must be moved instead). A mismatch with the training
    distribution only warns, since the likelihood stays exact and the network is
    merely a worse proposal: no notch training (including non-tokenized models),
    ``mask_random_tokens`` only, or an interval outside the
    ``tokenization.mask_frequency_notches`` envelope (range and ``max_width``).

    Parameters
    ----------
    psd_notch_dict : dict
        ``{det: [f_lo, f_hi]}`` or ``{det: [[f_lo, f_hi], ...]}``.
    domain : UniformFrequencyDomain or MultibandedFrequencyDomain
        The model's base (uniform) domain.
    data_settings : dict
        ``train_settings["data"]`` of the model.

    Raises
    ------
    ValueError
        If the notches are incompatible with the model or the domain.
    """
    model_detectors = data_settings["detectors"]
    unknown = set(psd_notch_dict) - set(model_detectors)
    if unknown:
        raise ValueError(
            f"psd_notch_dict names detectors {sorted(unknown)} the model was not "
            f"trained with (detectors: {model_detectors})."
        )
    intervals = []
    for det, notch in psd_notch_dict.items():
        ranges = [notch] if not isinstance(notch[0], (list, tuple)) else notch
        for f_lo, f_hi in ranges:
            if not f_lo <= f_hi:
                raise ValueError(
                    f"psd_notch_dict interval [{f_lo}, {f_hi}] for {det} is empty."
                )
            if f_lo <= domain.f_min or f_hi >= domain.f_max:
                raise ValueError(
                    f"psd_notch_dict interval [{f_lo}, {f_hi}] for {det} touches the "
                    f"domain bounds [{domain.f_min}, {domain.f_max}]; move "
                    f"minimum_frequency / maximum_frequency instead of notching an edge."
                )
            intervals.append((det, f_lo, f_hi))

    tok = data_settings.get("tokenization") or {}
    notch_settings = tok.get("mask_frequency_notches")
    if notch_settings is None:
        if "mask_random_tokens" in tok:
            warnings.warn(
                "psd_notch_dict relies on mask_random_tokens training only; the "
                "contiguous masking pattern differs from the random training "
                "distribution. Expect reduced importance-sampling efficiency and "
                "check the effective sample size."
            )
        else:
            warnings.warn(
                "Model was not trained with mask_frequency_notches; the notched bins "
                "are out of distribution for the network. The likelihood is exact, "
                "so check the importance-sampling efficiency."
            )
        return

    # Training envelope as MaskFrequencyNotches resolves it: an explicit range is
    # clamped to the domain, and the width is capped by the range.
    f_min = notch_settings.get("f_min")
    f_max = notch_settings.get("f_max")
    notch_f_min = domain.f_min if f_min is None else max(f_min, domain.f_min)
    notch_f_max = domain.f_max if f_max is None else min(f_max, domain.f_max)
    max_width = min(notch_settings["max_width"], notch_f_max - notch_f_min)
    for det, f_lo, f_hi in intervals:
        if f_lo < notch_f_min or f_hi > notch_f_max or f_hi - f_lo > max_width + 1e-9:
            warnings.warn(
                f"psd_notch_dict interval [{f_lo}, {f_hi}] for {det} is outside the "
                f"training envelope (mask_frequency_notches: range "
                f"[{notch_f_min}, {notch_f_max}] Hz, max_width {max_width} Hz). "
                f"Expect reduced importance-sampling efficiency."
            )


def check_psd_notches(model_metadata: dict, psd_notch_dict: dict):
    """
    Validate PSD notch intervals against a model's metadata.

    Thin metadata-level wrapper around ``_validate_psd_notches``, used by dingo_pipe
    at DAG-build time; see there for the accepted forms and semantics.
    """
    domain = build_domain_from_model_metadata(model_metadata, base=True)
    if not isinstance(domain, (UniformFrequencyDomain, MultibandedFrequencyDomain)):
        raise ValueError("psd_notch_dict requires a frequency domain.")
    _validate_psd_notches(
        psd_notch_dict, domain, model_metadata["train_settings"]["data"]
    )


def _validate_detectors_transformer(
    detectors_event: list[str],
    detectors_network: list[str],
    mask_detector_settings: dict,
):
    """
    Validate that the event detectors are compatible with a transformer network
    trained with detector masking.

    The event detectors must be a subset of the training detectors, and every
    *absent* training detector must have been maskable in training. Keys missing
    from ``mask_detector_settings`` impose no constraint, since ``MaskDetectors``
    then defaulted to uniform probabilities.

    Parameters
    ----------
    detectors_event : list[str]
        Detectors present in the event data.
    detectors_network : list[str]
        Detectors the network was trained with.
    mask_detector_settings : dict
        The ``tokenization.mask_detectors`` sub-dict from the train settings.

    Raises
    ------
    ValueError
        If the detector configuration is incompatible with the network.
    """
    if not set(detectors_event).issubset(set(detectors_network)):
        raise ValueError(
            f"Event has detectors {detectors_event} but model was only trained "
            f"with detectors {detectors_network}."
        )
    absent = set(detectors_network) - set(detectors_event)

    p_mask_012 = mask_detector_settings.get("p_mask_012_detectors")
    # p_mask_012[k] = probability of masking k detectors during training.
    if p_mask_012 is not None and (
        len(absent) >= len(p_mask_012) or p_mask_012[len(absent)] == 0.0
    ):
        raise ValueError(
            f"Event has detectors {detectors_event}, but model was trained with "
            f"p_mask_012_detectors={p_mask_012}, not allowing "
            f"{len(detectors_event)} active detectors."
        )

    p_mask_hlv = mask_detector_settings.get("p_mask_hlv")
    # p_mask_hlv[det] = probability that det is masked; zero means det was always
    # present in training, so it must also be present in the event.
    if p_mask_hlv is not None:
        for det in absent:
            if p_mask_hlv.get(det, 0.0) == 0.0:
                raise ValueError(
                    f"Detector {det} was never masked in training "
                    f"(p_mask_hlv={p_mask_hlv}); cannot drop it at inference."
                )


def check_detector_update(
    model_metadata: dict,
    detectors: list[str],
):
    """
    Validate that a given set of detectors is compatible with the network.

    For transformer networks trained with ``tokenization.mask_detectors``, the event
    detectors must be a subset of the training detectors and must be allowed by the
    masking probabilities.  For networks trained with ``tokenization.mask_random_tokens``
    only the subset check is performed.  For non-tokenization networks the event detectors
    must exactly match the training detectors.

    Parameters
    ----------
    model_metadata : dict
        Dictionary containing the network's training settings and data.
    detectors : list[str]
        Detectors present in the event data.

    Raises
    ------
    ValueError
        If the detector configuration is incompatible with the model.
    """
    detectors_network = model_metadata["train_settings"]["data"]["detectors"]
    if not set(detectors).issubset(set(detectors_network)):
        raise ValueError(
            f"Event has detectors {detectors} but model was only trained with "
            f"detectors {detectors_network}."
        )
    tok = model_metadata["train_settings"]["data"].get("tokenization", {})
    if "mask_detectors" in tok:
        _validate_detectors_transformer(
            detectors_event=detectors,
            detectors_network=detectors_network,
            mask_detector_settings=tok["mask_detectors"],
        )
    elif "mask_random_tokens" in tok:
        # Token-level masking does not constrain which detectors are present.
        pass
    elif set(detectors) != set(detectors_network):
        # Without detector masking (tokenized or not), an exact match is required.
        raise ValueError(
            f"Detectors {detectors} of event do not match detectors "
            f"{detectors_network} from model."
        )
