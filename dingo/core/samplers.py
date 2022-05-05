import numpy as np
import pandas as pd
from astropy.time import Time
from bilby.core.prior import Prior, Constraint, DeltaFunction
from bilby.core.result import Result
from bilby.gw.detector import InterferometerList
from torchvision.transforms import Compose

from dingo.core.transforms import GetItem, RenameKey
from dingo.gw.domains import build_domain
from dingo.gw.gwutils import get_window_factor, get_extrinsic_prior_dict
from dingo.gw.prior import build_prior_with_defaults
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
        self.transforms_pre = Compose([])
        self.transforms_post = Compose([])
        self._search_parameter_keys = []
        self._constraint_parameter_keys = []
        self._fixed_parameter_keys = []
        self._build_prior()
        self._reset_sampler()

    def _run_sampler(self, n, context, batch_size=None):

        x = context.copy()
        x["parameters"] = {}
        x["extrinsic_parameters"] = {}

        x = self.transforms_pre(context)
        x = x.expand(n, *x.shape)
        y = self.model.sample(x, batch_size=batch_size)
        samples = self.transforms_post({"parameters": y})["parameters"]

        return samples

    def run_sampler(
        self, n, context, label=None, as_type='result', **post_correction_kwargs
    ):
        self._reset_sampler()
        self.injection_parameters = context.pop("parameters", None)
        self.label = label

        samples = self._run_sampler(n, context)
        self._post_correct(samples, **post_correction_kwargs)
        samples = {k: v.cpu() for k, v in samples.items()}

        self._store_metadata()
        self._generate_result(samples)

        if as_type == 'result':
            return self.result
        elif as_type == 'pandas':
            samples = pd.DataFrame(samples)
            samples.attrs = self.metadata
            return samples
        elif as_type == 'dict':
            return samples

    def _post_correct(self, samples, **kwargs):
        pass

    def _build_prior(self):
        intrinsic_prior = self.model.metadata["dataset_settings"]["intrinsic_prior"]
        extrinsic_prior = get_extrinsic_prior_dict(
            self.model.metadata["train_settings"]["data"]["extrinsic_prior"]
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

    def _generate_result(self, samples):
        result_kwargs = dict(
            label=self.label,
            # outdir=self.outdir,
            sampler=self.__class__.__name__.lower(),
            search_parameter_keys=self._search_parameter_keys,
            fixed_parameter_keys=self._fixed_parameter_keys,
            constraint_parameter_keys=self._constraint_parameter_keys,
            priors=self.prior,
            meta_data=self.metadata,
            injection_parameters=self.injection_parameters,
            sampler_kwargs=None,
            use_ratio=False,
        )
        self.result = Result(**result_kwargs)
        self.result.samples = samples

        # TODO: decide whether to run this, and whether to use it to generate
        #  additional parameters.
        # self.result.samples_to_posterior()

    def _store_metadata(self):
        self.metadata = dict(
            model_metadata=self.model.metadata,
        )

    def _reset_sampler(self):
        self.result = None
        self.samples = None
        self.injection_parameters = None
        self.label = None


class GNPESampler(ConditionalSampler):
    def __init__(self, model, init_sampler: ConditionalSampler, num_iterations: int):
        super().__init__(model)
        self.init_sampler = init_sampler
        self.num_iterations = num_iterations
        self.gnpe_parameters = None

    def _run_sampler(self, n, context, batch_size=None):
        # TODO: Add batching + ability to sample several events simulataneously
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

    def _store_metadata(self):
        super()._store_metadata()
        self.init_sampler._store_metadata()
        self.metadata["init_model_metadata"] = self.init_sampler.metadata

        # TODO: Could also go in sampler_kwargs, which we don't use now.
        self.metadata["num_iterations"] = self.num_iterations


class GWSamplerMixin(object):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_domain()
        self.inference_parameters = self.model.metadata["train_settings"]["data"][
            "inference_parameters"
        ]
        self.t_ref = self.model.metadata["train_settings"]["data"]["ref_time"]

    def build_domain(self):
        self.domain = build_domain(self.model.metadata["dataset_settings"]["domain"])

        data_settings = self.model.metadata["train_settings"]["data"]
        if "domain_update" in data_settings:
            self.domain.update(data_settings["domain_update"])

        self.domain.window_factor = get_window_factor(data_settings["window"])

    def _post_correct(self, samples, t_event=None):
        self.time_event = t_event
        if t_event is not None:
            ra = samples["ra"]
            time_reference = Time(self.t_ref, format="gps", scale="utc")
            time_event = Time(t_event, format="gps", scale="utc")
            longitude_event = time_event.sidereal_time("apparent", "greenwich")
            longitude_reference = time_reference.sidereal_time("apparent", "greenwich")
            delta_longitude = longitude_event - longitude_reference
            ra_correction = delta_longitude.rad
            samples["ra"] = (ra + ra_correction) % (2 * np.pi)

    def _store_metadate(self):
        super()._store_metadata()
        self.metadata["event"] = {
            "time_event": self.time_event,
            # "time_psd": args.time_psd,
            # "time_buffer": args.time_buffer,
        }
        # TODO: Store strain data? ASD?


class GWSamplerNPE(GWSamplerMixin, ConditionalSampler):
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
