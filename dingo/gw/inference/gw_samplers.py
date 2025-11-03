from typing import Union, Protocol

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
)


class SamplerProtocol(Protocol):
    base_model_metadata: dict

    def _initialize_transforms(self) -> None:
        ...


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
        super().__init__(**kwargs)
        self.t_ref = self.base_model_metadata["train_settings"]["data"]["ref_time"]
        self._pesummary_package = "gw"
        self._result_class = Result

    @property
    def detectors(self: SamplerProtocol):
        return self.base_model_metadata["train_settings"]["data"]["detectors"]

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
        _validate_minimum_frequency(
            value,
            self.detectors,
            domain,
            self.random_strain_cropping,
        )  # TODO: Ensure minimum frequency is a dict?
        self._minimum_frequency = value
        self._initialize_transforms()

    @property
    def maximum_frequency(self) -> float | dict[str, float]:
        if self._maximum_frequency is not None:
            return self._maximum_frequency
        else:
            return self.domain.f_max

    @maximum_frequency.setter
    def maximum_frequency(self: _GWMixinProtocol, value: Union[float, dict]):
        if isinstance(self.domain, MultibandedFrequencyDomain):
            domain = self.domain.base_domain
        elif isinstance(self.domain, UniformFrequencyDomain):
            domain = self.domain
        else:
            raise ValueError("Frequency updates only possible for frequency domains.")
        _validate_maximum_frequency(
            value,
            self.detectors,
            domain,
            self.random_strain_cropping,
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
    def event_metadata(self):
        if self._event_metadata is not None:
            metadata = self._event_metadata.copy()
        else:
            metadata = {}
        metadata["minimum_frequency"] = self.minimum_frequency
        metadata["maximum_frequency"] = self.maximum_frequency
        return metadata

    @event_metadata.setter
    def event_metadata(self, value):
        if value is not None:
            value = value.copy()
            if "minimum_frequency" in value:
                self.minimum_frequency = value.pop("minimum_frequency")
            if "maximum_frequency" in value:
                self.maximum_frequency = value.pop("maximum_frequency")
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
        if self.frequency_updates:
            # * update frequency range
            # Needs to happen before RepackageStrainsAndASDs since we might need to apply
            # detectors specific frequency updates.
            transform_pre.append(
                MaskDataForFrequencyRangeUpdate(
                    domain=self.domain,
                    minimum_frequency=self.minimum_frequency,
                    maximum_frequency=self.maximum_frequency,
                )
            )
        #   * repackage strains and asds from dicts to an array
        #   * convert array to torch tensor on the correct device
        #   * extract only strain/waveform from the sample
        transform_pre += [
            # Use base metadata so that unconditional samplers still know how to
            # transform data, since this transform is used by the GNPE sampler as
            # well.
            RepackageStrainsAndASDS(
                ifos=self.detectors,
                first_index=self.domain.min_idx,
            ),
            ToTorch(device=self.model.device),
            GetItem("waveform"),
        ]
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


def _validate_maximum_frequency(
    f_max: dict[str, float] | float,
    detectors: list[str],
    domain: UniformFrequencyDomain | MultibandedFrequencyDomain,
    crop_settings: dict | None,
):
    if isinstance(f_max, float):
        f_max = {d: f_max for d in detectors}
    if set(f_max) != set(detectors):
        raise ValueError(
            f"f_max must have exactly detectors {detectors}, got " f"{list(f_max)}."
        )
    f_max_vals = np.array([f_max[d] for d in detectors])

    # Hard upper bound
    if np.any(f_max_vals > domain.f_max):
        raise ValueError(f"f_max {f_max} > domain.f_max = {domain.f_max}.")

    # Nothing changed
    if np.all(f_max_vals == domain.f_max):
        return

    # Cropping must be on
    if not crop_settings or crop_settings.get("cropping_probability", 0.0) == 0.0:
        raise ValueError(
            f"Cropping disabled; cannot lower maximum frequency to {f_max}."
        )

    # Extract lower bounds
    floors = crop_settings.get("f_max_lower")
    if floors is None:
        floors = domain.f_max
    if not isinstance(floors, dict):
        floors = {d: floors for d in detectors}

    # Check lower bound.
    if not crop_settings.get("independent_detectors", True):
        if len(set(f_max_vals)) > 1:
            raise ValueError(
                f"Independent max frequencies per detector not enabled. "
                f"All frequencies must match, got f_max = {f_max}."
            )
        # TODO: Risk of non-constant floors with non-independent detectors.
        assert len(set(floors.values())) == 1
    for d in detectors:
        if f_max[d] < floors[d]:
            raise ValueError(
                f"Maximum frequency requested for {d} ({f_max[d]} Hz) "
                f"less than lower bound of {floors[d]} Hz."
            )


def _validate_minimum_frequency(
    f_min: dict[str, float] | float,
    detectors: list[str],
    domain: UniformFrequencyDomain | MultibandedFrequencyDomain,
    crop_settings: dict | None,
):
    if isinstance(f_min, float):
        f_min = {d: f_min for d in detectors}
    if set(f_min) != set(detectors):
        raise ValueError(
            f"f_min must have exactly detectors {detectors}, got {list(f_min)}."
        )
    f_min_vals = np.array([f_min[d] for d in detectors])

    # Hard lower bound
    if np.any(f_min_vals < domain.f_min):
        raise ValueError(f"f_min {f_min} < domain.f_min = {domain.f_min}.")

    # Nothing changed
    if np.all(f_min_vals == domain.f_min):
        return

    # Cropping must be on
    if not crop_settings or crop_settings.get("cropping_probability", 0.0) == 0.0:
        raise ValueError(
            f"Cropping disabled; cannot raise minimum frequency to {f_min}."
        )

    # Extract upper bounds
    caps = crop_settings.get("f_min_upper")
    if caps is None:
        caps = domain.f_min
    if not isinstance(caps, dict):
        caps = {d: caps for d in detectors}

    # Check upper bound.
    if not crop_settings.get("independent_detectors", True):
        if len(set(f_min_vals)) > 1:
            raise ValueError(
                f"Independent min frequencies per detector not enabled. "
                f"All frequencies must match, got f_min = {f_min}."
            )
        # TODO: Risk of non-constant caps with non-independent detectors.
        assert len(set(caps.values())) == 1
    for d in detectors:
        if f_min[d] > caps[d]:
            raise ValueError(
                f"Minimum frequency requested for {d} ({f_min[d]} Hz) "
                f"greater than upper bound of {caps[d]} Hz."
            )


def check_frequency_updates(
    model_metadata: dict,
    f_min: dict[str, float] | float | None = None,
    f_max: dict[str, float] | float | None = None,
):
    """
    Validate and apply optional minimum and maximum frequency constraints
    for a model’s frequency domain.

    This function checks that any provided per-detector minimum (`f_min`)
    or maximum (`f_max`) frequencies—either as a single float applied to
    all detectors or as a dict mapping each detector to its own value—:
      - Match exactly the set of detectors in the model metadata.
      - Respect the hard bounds defined by the domain (`domain.f_min` /
        `domain.f_max`).
      - Comply with optional random-strain-cropping settings (probability,
        independent vs. joint detectors, and per-detector caps/floors).

    Parameters
    ----------
    model_metadata : dict
        Dictionary containing the model’s training settings and data.
        Must include:
          - `["train_settings"]["data"]["detectors"]`: list of detector names.
          - `["train_settings"]["data"]["random_strain_cropping"]`: optional
            dict of cropping parameters.
    f_min : dict[str, float], float, or None, optional
        Single float or per-detector dict of minimum frequencies to enforce.
        If a float is provided, it is applied to all detectors. Each value
        must be ≥ `domain.f_min`. If `None`, no minimum-frequency
        validation is performed.
    f_max : dict[str, float], float, or None, optional
        Single float or per-detector dict of maximum frequencies to enforce.
        If a float is provided, it is applied to all detectors. Each value
        must be ≤ `domain.f_max`. If `None`, no maximum-frequency
        validation is performed.

    Raises
    ------
    ValueError
        - If `model_metadata` does not describe a `UniformFrequencyDomain`
          or `MultibandedFrequencyDomain`.
        - If `f_min`/`f_max` keys don’t exactly match the detector list.
        - If any requested frequency lies outside the hard domain bounds.
        - If cropping is disabled but a change in frequency is requested.
        - If per-detector constraints (independent vs. joint) or
          cropping caps/floors are violated.

    Returns
    -------
    None
    """
    crop_settings = model_metadata["train_settings"]["data"].get(
        "random_strain_cropping"
    )
    detectors = model_metadata["train_settings"]["data"]["detectors"]
    domain = build_domain_from_model_metadata(model_metadata, base=True)
    if not isinstance(domain, (UniformFrequencyDomain, MultibandedFrequencyDomain)):
        raise ValueError("Frequency updates only possible for frequency domains.")

    if f_min is not None:
        _validate_minimum_frequency(f_min, detectors, domain, crop_settings)
    if f_max is not None:
        _validate_maximum_frequency(f_max, detectors, domain, crop_settings)
