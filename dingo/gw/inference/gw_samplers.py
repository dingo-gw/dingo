from typing import Optional, Union
import time

import numpy as np
import pandas as pd
from astropy.time import Time
from bilby.core.prior import PriorDict
from bilby.gw.detector import InterferometerList
from torchvision.transforms import Compose

from dingo.core.samplers import Sampler, GNPESampler
from dingo.core.transforms import GetItem, RenameKey
from dingo.gw.domains import build_domain
from dingo.gw.gwutils import get_window_factor
from dingo.gw.result import Result
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
    GNPEPhase,
)


class GWSamplerMixin(object):
    """
    Mixin class designed to add gravitational wave functionality to Sampler classes:
        * builders for prior, domain, and likelihood
        * correction for fixed detector locations during training (t_ref)
    """

    def __init__(self, synthetic_phase_kwargs=None, **kwargs):
        """
        Parameters
        ----------
        synthetic_phase_kwargs : dict = None
            kwargs for synthetic phase generation.
        kwargs
            Keyword arguments that are forwarded to the superclass.
        """
        super().__init__(**kwargs)
        self.t_ref = self.base_model_metadata["train_settings"]["data"]["ref_time"]
        self._pesummary_package = "gw"
        self.synthetic_phase_kwargs = synthetic_phase_kwargs
        self._result_class = Result

    def _build_domain(self):
        """
        Construct the domain object based on model metadata. Includes the window factor
        needed for whitening data.

        Called by __init__() immediately after _build_prior().
        """
        self.domain = build_domain(
            self.base_model_metadata["dataset_settings"]["domain"]
        )

        data_settings = self.base_model_metadata["train_settings"]["data"]
        if "domain_update" in data_settings:
            self.domain.update(data_settings["domain_update"])

        self.domain.window_factor = get_window_factor(data_settings["window"])

    def _correct_reference_time(
        self, samples: Union[dict, pd.DataFrame], inverse: bool = False
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
            if t_event is not None and t_event != self.t_ref:
                assert self.samples_dataset is None, "t_ref correction should be needed"
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
        Post processing of parameter samples.
        * Correct the sky position for a potentially fixed reference time.
          (see self._correct_reference_time)
        * Potentially sample a synthetic phase. (see self._sample_synthetic_phase)

        This method modifies the samples in place.

        Parameters
        ----------
        samples : dict or pd.DataFrame
        inverse : bool, default True
            Whether to apply instead the inverse transformation. This is used prior to
            calculating the log_prob.
        """
        if not inverse:
            self._correct_reference_time(samples, inverse)
            if self.synthetic_phase_kwargs is not None:
                print(f"Sampling synthetic phase.")
                t0 = time.time()
                self._sample_synthetic_phase(samples, inverse)
                print(f"Done. This took {time.time() - t0:.2f} seconds.")

        # If inverting, we go in reverse order.
        else:
            if self.synthetic_phase_kwargs is not None:
                self._sample_synthetic_phase(samples, inverse)
            self._correct_reference_time(samples, inverse)


class GWSampler(GWSamplerMixin, Sampler):
    """
    Sampler for gravitational-wave inference using neural posterior estimation. Wraps a
    PosteriorModel instance.

    This is intended for use either as a standalone sampler, or as a sampler producing
    initial sample points for a GNPE sampler.
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        model : PosteriorModel
        """
        super().__init__(**kwargs)
        if self.model is not None:
            self._initialize_transforms()

    def _initialize_transforms(self):

        # preprocessing transforms:
        #   * whiten and scale strain (since the inference network expects standardized
        #   data)
        #   * repackage strains and asds from dicts to an array
        #   * convert array to torch tensor on the correct device
        #   * extract only strain/waveform from the sample
        self.transform_pre = Compose(
            [
                WhitenAndScaleStrain(self.domain.noise_std),
                # Use base metadata so that unconditional samplers still know how to
                # transform data, since this transform is used by the GNPE sampler as
                # well.
                RepackageStrainsAndASDS(
                    self.base_model_metadata["train_settings"]["data"]["detectors"],
                    first_index=self.domain.min_idx,
                ),
                ToTorch(device=self.model.device),
                GetItem("waveform"),
            ]
        )

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
    Sampler for graviational-wave inference using group-equivariant neural posterior
    estimation (GNPE). Wraps a PosteriorModel instance.

    This sampler also contains an NPE sampler, which is used to generate initial
    samples for the GNPE loop.
    """

    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        model : PosteriorModel
            GNPE model.
        init_model : PosteriodModel
            Used to produce samples for initializing the GNPE loop.
        num_iterations :
            Number of GNPE iterations to be performed by sampler.
        """
        super().__init__(**kwargs)
        self._initialize_transforms()

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
        transform_pre = [RenameKey("data", "waveform")]
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
        if gnpe_chirp_settings:
            transform_pre.append(
                GNPEChirp(
                    gnpe_chirp_settings["kernel"],
                    self.domain,
                    gnpe_chirp_settings.get("order", 0),
                )
            )
        if gnpe_phase_settings:
            transform_pre.append(
                GNPEPhase(
                    gnpe_phase_settings["kernel"],
                    gnpe_phase_settings.get("random_pi_jump", False),
                )
            )
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
            raise NotImplementedError(
                "kernel log_prob only implemented for time gnpe."
            )
        gnpe_proxies_diff = {
            k: np.array(samples[k] - samples[f"{k}_proxy"])
            for k in self.gnpe_kernel.keys()
        }
        return self.gnpe_kernel.ln_prob(gnpe_proxies_diff, axis=0)


class GWSamplerUnconditional(GWSampler):
    def _initialize_transforms(self):
        # Postprocessing transform only:
        #   * De-standardize data and extract inference parameters. Be careful to use
        #   the standardization of the correct model, not the base model.
        self.transform_post = SelectStandardizeRepackageParameters(
            {"inference_parameters": self.inference_parameters},
            self.metadata["train_settings"]["data"]["standardization"],
            inverse=True,
            as_type="dict",
        )

    def _correct_reference_time(
        self, samples: Union[dict, pd.DataFrame], inverse: bool = False
    ):
        # We do not want to correct for t_ref because we assume that the unconditional
        # model will have been trained on samples for which this correction was already
        # implemented.
        pass
