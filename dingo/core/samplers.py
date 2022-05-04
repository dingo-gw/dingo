import numpy as np
import pandas as pd
from astropy.time import Time
from bilby.core.result import Result
from bilby.gw.detector import InterferometerList
from torchvision.transforms import Compose

from dingo.core.transforms import GetItem, RenameKey
from dingo.gw.domains import build_domain
from dingo.gw.gwutils import get_window_factor
from dingo.gw.transforms import (
    WhitenAndScaleStrain,
    RepackageStrainsAndASDS,
    SelectStandardizeRepackageParameters,
    GNPECoalescenceTimes,
    TimeShiftStrain,
    PostCorrectGeocentTime,
    GetDetectorTimes,
    CopyToExtrinsicParameters,
    ToTorch,
    GNPEChirp,
    GNPEBase,
)


class ConditionalSampler(object):
    def __init__(self, model):
        self.model = model
        self.initialize_transforms()

    def _run_sampler(self, n, context, batch_size=None):
        x = self.transforms_pre(context)
        x = x.expand(n, *x.shape)
        y = self.model.sample(x, batch_size=batch_size)
        samples = self.transforms_post({"parameters": y})["parameters"]

        return samples

    def run_sampler(self, n, context, **kwargs):
        samples = self._run_sampler(n, context)
        self._post_correct(samples, **kwargs)
        samples = {k: v.cpu() for k, v in samples.items()}
        return pd.DataFrame(samples)

    def initialize_transforms(self):
        self.transforms_pre = Compose([])
        self.transforms_post = Compose([])

    def _generate_result(self, samples):
        pass

    def _post_correct(self, samples, **kwargs):
        pass


class GNPESampler(ConditionalSampler):
    def __init__(self, model, init_sampler: ConditionalSampler, num_iterations: int):
        self.init_sampler = init_sampler
        self.num_iterations = num_iterations
        self.gnpe_parameters = None
        super().__init__(model)

    def _run_sampler(self, n, context, batch_size=None):
        if batch_size is None:
            batch_size = n

        data_ = self.init_sampler.transforms_pre(context)

        x = {
            "extrinsic_parameters": self.init_sampler._run_sampler(
                n, context, batch_size=batch_size
            ),
            "parameters": {},
        }
        for i in range(self.num_iterations):
            print(i)
            x["extrinsic_parameters"] = {
                k: x["extrinsic_parameters"][k] for k in self.gnpe_parameters
            }
            d = data_.clone()
            x["data"] = d.expand(n, *d.shape)

            x = self.transforms_pre(x)
            x["parameters"] = self.model.sample(x["data"], x["context_parameters"])
            x = self.transforms_post(x)

        samples = x["parameters"]

        return samples


class GWSamplerMixin(object):
    def __init__(self, **kwargs):
        self.model = kwargs["model"]
        self.build_domain()
        self.inference_parameters = self.model.metadata["train_settings"]["data"][
            "inference_parameters"
        ]
        self.t_ref = self.model.metadata["train_settings"]["data"]["ref_time"]
        super().__init__(**kwargs)

    def build_domain(self):
        self.domain = build_domain(self.model.metadata["dataset_settings"]["domain"])

        data_settings = self.model.metadata["train_settings"]["data"]
        if "domain_update" in data_settings:
            self.domain.update(data_settings["domain_update"])

        self.domain.window_factor = get_window_factor(data_settings["window"])

    def _post_correct(self, samples, t_event):
        ra = samples["ra"]
        time_reference = Time(self.t_ref, format="gps", scale="utc")
        time_event = Time(t_event, format="gps", scale="utc")
        longitude_event = time_event.sidereal_time("apparent", "greenwich")
        longitude_reference = time_reference.sidereal_time("apparent", "greenwich")
        delta_longitude = longitude_event - longitude_reference
        ra_correction = delta_longitude.rad
        samples["ra"] = (ra + ra_correction) % (2 * np.pi)



class GWSamplerNPE(GWSamplerMixin, ConditionalSampler):
    def initialize_transforms(self):

        data_settings = self.model.metadata["train_settings"]["data"]

        # preprocessing transforms:
        #   * whiten and scale strain (since the inference network expects standardized
        #   data)
        #   * repackage strains and asds from dicts to an array
        #   * convert array to torch tensor on the correct device
        #   * expand strain num_sample times
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
    def initialize_transforms(self):

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


class ImportanceSampler(GWSamplerMixin, ConditionalSampler):
    pass
