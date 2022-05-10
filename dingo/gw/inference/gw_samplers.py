from typing import Optional

import numpy as np
from astropy.time import Time
from bilby.core.prior import Prior, Constraint, DeltaFunction, Uniform
from bilby.gw.detector import InterferometerList
from torchvision.transforms import Compose

from dingo.core.samplers import Sampler, GNPESampler
from dingo.core.transforms import GetItem, RenameKey
from dingo.gw.domains import build_domain
from dingo.gw.gwutils import get_window_factor, get_extrinsic_prior_dict
from dingo.gw.likelihood import StationaryGaussianGWLikelihood
from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.transforms import (
    WhitenAndScaleStrain,
    RepackageStrainsAndASDS,
    ToTorch,
    SelectStandardizeRepackageParameters,
    GNPECoalescenceTimes,
    TimeShiftStrain,
    GNPEChirp,
    GNPEBase,
    PostCorrectGeocentTime,
    CopyToExtrinsicParameters,
    GetDetectorTimes,
)


class GWSamplerMixin(object):
    """
    Mixin class designed to add gravitational wave functionality to Sampler classes:
        * domain information
        * correction for fixed detector locations during training (t_ref)
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        kwargs
            Keyword arguments that are forwarded to the superclass.
        """
        super().__init__(**kwargs)

        self.inference_parameters = self.metadata["train_settings"]["data"][
            "inference_parameters"
        ]
        self.t_ref = self.base_model_metadata["train_settings"]["data"]["ref_time"]
        self._pesummary_package = "gw"

    # _build_domain amd _build_prior are called by Sampler.__init__, in that order.

    def _build_domain(self):
        """
        Constructs the domain object based on model metadata. Includes the window
        factor needed for whitening data.
        """
        self.domain = build_domain(
            self.base_model_metadata["dataset_settings"]["domain"]
        )

        data_settings = self.base_model_metadata["train_settings"]["data"]
        if "domain_update" in data_settings:
            self.domain.update(data_settings["domain_update"])

        self.domain.window_factor = get_window_factor(data_settings["window"])

    def _build_prior(self):
        """Build the prior based on model metadata."""

        intrinsic_prior = self.base_model_metadata["dataset_settings"][
            "intrinsic_prior"
        ]
        extrinsic_prior = get_extrinsic_prior_dict(
            self.base_model_metadata["train_settings"]["data"]["extrinsic_prior"]
        )
        self.prior = build_prior_with_defaults({**intrinsic_prior, **extrinsic_prior})

        # Initialize lists of parameters (from Bilby)
        for key in self.prior:
            if isinstance(self.prior[key], Prior) and self.prior[key].is_fixed is False:
                self._search_parameter_keys.append(key)
            elif isinstance(self.prior[key], Constraint):
                self._constraint_parameter_keys.append(key)
            elif isinstance(self.prior[key], DeltaFunction):
                # self.likelihood.parameters[key] = self.prior[key].sample()
                self._fixed_parameter_keys.append(key)

    # _build_likelihood is called at the beginning of Sampler.importance_sample

    def _build_likelihood(self, time_marginalization_kwargs: Optional[dict] = None):

        if time_marginalization_kwargs is not None:
            time_prior = self.prior.pop("geocent_time")
            if type(time_prior) != Uniform:
                raise NotImplementedError(
                    "Only uniform time prior is supported for time marginalization."
                )
            time_marginalization_kwargs["t_lower"] = time_prior.minimum
            time_marginalization_kwargs["t_upper"] = time_prior.maximum

        self.likelihood = StationaryGaussianGWLikelihood(
            wfg_kwargs=self.base_model_metadata["dataset_settings"][
                "waveform_generator"
            ],
            wfg_domain=build_domain(
                self.base_model_metadata["dataset_settings"]["domain"]
            ),
            data_domain=self.domain,
            event_data=self.context,
            t_ref=self.t_ref,
            time_marginalization_kwargs=time_marginalization_kwargs,
        )

    def _post_correct(self, samples: dict):
        """
        Correct the sky position of an event based on the reference time of the model.
        This is necessary since the model was trained with with fixed detector (reference)
        positions. This transforms the right ascension based on the e difference between
        the time of the event and t_ref.

        The correction is only applied if the event time can be found in self.metadata[
        'event'].

        This method modifies the samples dict in place.

        Parameters
        ----------
        samples : dict
        """
        event_metadata = self.metadata.get("event")
        if event_metadata is not None:
            t_event = event_metadata.get("time_event")
            if t_event is not None and t_event != self.t_ref:
                ra = samples["ra"]
                time_reference = Time(self.t_ref, format="gps", scale="utc")
                time_event = Time(t_event, format="gps", scale="utc")
                longitude_event = time_event.sidereal_time("apparent", "greenwich")
                longitude_reference = time_reference.sidereal_time(
                    "apparent", "greenwich"
                )
                delta_longitude = longitude_event - longitude_reference
                ra_correction = delta_longitude.rad
                samples["ra"] = (ra + ra_correction) % (2 * np.pi)


class GWSampler(GWSamplerMixin, Sampler):
    """
    Sampler for gravitational-wave inference using neural posterior estimation. Wraps a
    PosteriorModel instance.

    This is intended for use either as a standalone sampler, or as a sampler producing
    initial sample points for a GNPE sampler.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.model is not None:
            self._initialize_transforms()

    def _initialize_transforms(self):

        data_settings = self.metadata["train_settings"]["data"]

        # preprocessing transforms:
        #   * whiten and scale strain (since the inference network expects standardized
        #   data)
        #   * repackage strains and asds from dicts to an array
        #   * convert array to torch tensor on the correct device
        #   * extract only strain/waveform from the sample
        self.transforms_pre = Compose(
            [
                WhitenAndScaleStrain(self.domain.noise_std),
                RepackageStrainsAndASDS(
                    data_settings["detectors"],
                    first_index=self.domain.min_idx,
                ),
                ToTorch(device=self.model.device),
                GetItem("waveform"),
            ]
        )

        # postprocessing transforms:
        #   * de-standardize data and extract inference parameters
        self.transforms_post = SelectStandardizeRepackageParameters(
            {"inference_parameters": self.inference_parameters},
            data_settings["standardization"],
            inverse=True,
            as_type="dict",
        )


class GWSamplerGNPE(GWSamplerMixin, GNPESampler):
    """
    Sampler for graviational-wave inference using group-equivariant neural posterior
    estimation. Wraps a PosteriorModel instance.

    This sampler also contains an NPE sampler, which is used to generate initial
    samples for the GNPE loop.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.model is not None:
            self._initialize_transforms()

    def _initialize_transforms(self):

        data_settings = self.metadata["train_settings"]["data"]
        ifo_list = InterferometerList(data_settings["detectors"])

        gnpe_time_settings = data_settings.get("gnpe_time_shifts")
        gnpe_chirp_settings = data_settings.get("gnpe_chirp")
        if not gnpe_time_settings and not gnpe_chirp_settings:
            raise KeyError(
                "GNPE inference requires network trained for either chirp mass "
                "or coalescence time GNPE."
            )

        # transforms for gnpe loop, to be applied prior to sampling step:
        #   * reset the sample (e.g., clone non-gnpe transformed waveform)
        #   * blurring detector times to obtain gnpe proxies
        #   * shifting the strain by - gnpe proxies
        #   * repackaging & standardizing proxies to sample['context_parameters']
        #     for conditioning of the inference network
        transforms_pre = [RenameKey("data", "waveform")]
        if gnpe_time_settings:
            transforms_pre.append(
                GNPECoalescenceTimes(
                    ifo_list,
                    gnpe_time_settings["kernel"],
                    gnpe_time_settings["exact_equiv"],
                    inference=True,
                )
            )
            transforms_pre.append(TimeShiftStrain(ifo_list, self.domain))
        if gnpe_chirp_settings:
            transforms_pre.append(
                GNPEChirp(
                    gnpe_chirp_settings["kernel"],
                    self.domain,
                    gnpe_chirp_settings.get("order", 0),
                )
            )
        transforms_pre.append(
            SelectStandardizeRepackageParameters(
                {"context_parameters": data_settings["context_parameters"]},
                data_settings["standardization"],
                device=self.model.device,
            )
        )
        transforms_pre.append(RenameKey("waveform", "data"))

        self.gnpe_parameters = []
        for transform in transforms_pre:
            if isinstance(transform, GNPEBase):
                self.gnpe_parameters += transform.input_parameter_names
        print("GNPE parameters: ", self.gnpe_parameters)

        self.transforms_pre = Compose(transforms_pre)

        # transforms for gnpe loop, to be applied after sampling step:
        #   * de-standardization of parameters
        #   * post correction for geocent time (required for gnpe with exact equivariance)
        #   * computation of detectortimes from parameters (required for next gnpe
        #       iteration)
        self.transforms_post = Compose(
            [
                SelectStandardizeRepackageParameters(
                    {"inference_parameters": self.inference_parameters},
                    data_settings["standardization"],
                    inverse=True,
                    as_type="dict",
                ),
                PostCorrectGeocentTime(),
                CopyToExtrinsicParameters(
                    "ra", "dec", "geocent_time", "chirp_mass", "mass_ratio"
                ),
                GetDetectorTimes(ifo_list, data_settings["ref_time"]),
            ]
        )


class GWSamplerUnconditional(GWSampler):
    def _initialize_transforms(self):

        # Postprocessing transform only:
        #   * De-standardize data and extract inference parameters. Be careful to use
        #   the standardization of the correct model, not the base model.
        self.transforms_post = SelectStandardizeRepackageParameters(
            {"inference_parameters": self.inference_parameters},
            self.metadata["train_settings"]["data"]["standardization"],
            inverse=True,
            as_type="dict",
        )

    def _post_correct(self, samples: dict):
        pass
