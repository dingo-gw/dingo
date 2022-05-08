import numpy as np
from astropy.time import Time
from bilby.gw.detector import InterferometerList
from torchvision.transforms import Compose

from dingo.core.samplers import ConditionalSampler, GNPESampler
from dingo.core.transforms import GetItem, RenameKey
from dingo.gw.domains import build_domain
from dingo.gw.gwutils import get_window_factor
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
        self._build_domain()
        self.inference_parameters = self.model.metadata["train_settings"]["data"][
            "inference_parameters"
        ]
        self.t_ref = self.model.metadata["train_settings"]["data"]["ref_time"]
        self._pesummary_package = 'gw'

    def _build_domain(self):
        """
        Constructs the domain object based on model metadata. Includes the window
        factor needed for whitening data.
        """
        self.domain = build_domain(self.model.metadata["dataset_settings"]["domain"])

        data_settings = self.model.metadata["train_settings"]["data"]
        if "domain_update" in data_settings:
            self.domain.update(data_settings["domain_update"])

        self.domain.window_factor = get_window_factor(data_settings["window"])

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
        t_event = self.metadata["event"].get("time_event")
        if t_event is not None:
            ra = samples["ra"]
            time_reference = Time(self.t_ref, format="gps", scale="utc")
            time_event = Time(t_event, format="gps", scale="utc")
            longitude_event = time_event.sidereal_time("apparent", "greenwich")
            longitude_reference = time_reference.sidereal_time("apparent", "greenwich")
            delta_longitude = longitude_event - longitude_reference
            ra_correction = delta_longitude.rad
            samples["ra"] = (ra + ra_correction) % (2 * np.pi)

    def _store_metadate(self, **kwargs):
        super()._store_metadata(**kwargs)
        # TODO: Store strain data? ASD?


class GWSamplerNPE(GWSamplerMixin, ConditionalSampler):
    """
    Sampler for gravitational-wave inference using neural posterior estimation. Wraps a
    PosteriorModel instance.

    This is intended for use either as a standalone sampler, or as a sampler producing
    initial sample points for a GNPE sampler.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialize_transforms()

    def _initialize_transforms(self):

        data_settings = self.model.metadata["train_settings"]["data"]

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
        self._initialize_transforms()

    def _initialize_transforms(self):

        data_settings = self.model.metadata["train_settings"]["data"]
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
