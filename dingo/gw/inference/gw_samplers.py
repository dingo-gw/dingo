from typing import Union

import numpy as np
import pandas as pd
from astropy.time import Time
from bilby.core.prior import PriorDict
from bilby.gw.detector import InterferometerList
from torchvision.transforms import Compose

from dingo.core.samplers import Sampler, GNPESampler
from dingo.core.transforms import GetItem, RenameKey
from dingo.gw.domains import build_domain, MultibandedFrequencyDomain
from dingo.gw.gwutils import get_window_factor
from dingo.gw.result import Result
from dingo.gw.gwutils import get_extrinsic_prior_dict
from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.transforms import (
    WhitenAndScaleStrain,
    DecimateWaveformsAndASDS,
    RepackageStrainsAndASDS,
    ToTorch,
    SelectStandardizeRepackageParameters,
    GNPECoalescenceTimes,
    GNPEChirp,
    TimeShiftStrain,
    GNPEBase,
    PostCorrectGeocentTime,
    CopyToExtrinsicParameters,
    GetDetectorTimes,
    HeterodynePhase,
    ApplyFrequencyMasking,
)


class GWSamplerMixin(object):
    """
    Mixin class designed to add gravitational wave functionality to Sampler classes:
        * builder for data domain
        * correction for fixed detector locations during training (t_ref)
    """

    def __init__(self, frequency_masking=None, **kwargs):
        """
        Parameters
        ----------
        kwargs
            Keyword arguments that are forwarded to the superclass.
        """
        self.frequency_masking = frequency_masking
        super().__init__(**kwargs)
        self.t_ref = self.base_model_metadata["train_settings"]["data"]["ref_time"]
        self._pesummary_package = "gw"
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
        Post processing of parameter samples.
        * Correct the sky position for a potentially fixed reference time.
          (see self._correct_reference_time)
        * Add derived parameters.
        * Add fixed parameters from the prior.

        This method modifies the samples in place.

        Parameters
        ----------
        samples : dict or pd.DataFrame
        inverse : bool, default True
            Whether to apply instead the inverse transformation. This is used prior to
            calculating the log_prob.
        """
        # Correct reference time
        if not self.unconditional_model:
            self._correct_reference_time(samples, inverse)

        # Add derived parameters
        keys = samples.keys()
        keys_new = [
            k[len("delta_") :]
            for k in keys
            if k.startswith("delta_")
            if k[len("delta_") :] not in keys and k[len("delta_") :] + "_proxy" in keys
        ]
        print(f"Adding parameters for {keys_new}.")
        for k in keys_new:
            samples[k] = samples["delta_" + k] + samples[k + "_proxy"]

        # Add fixed parameters from prior
        intrinsic_prior = self.metadata["dataset_settings"]["intrinsic_prior"]
        extrinsic_prior = get_extrinsic_prior_dict(
            self.metadata["train_settings"]["data"]["extrinsic_prior"]
        )
        priors = {**intrinsic_prior, **extrinsic_prior}
        num_samples = len(samples[list(samples.keys())[0]])
        for k, p in priors.items():
            if (
                p.startswith("bilby.core.prior.analytical.DeltaFunction")
                or p.startswith("DeltaFunction")
                and k not in samples
            ):
                v = float(p.split("(")[1].strip(")"))
                print(f"Adding fixed parameter {k} = {v} from prior.")
                samples[k] = v * np.ones(num_samples)

    def initialize_transforms_pre(self):
        """
        Initialize self.transform_pre. This is used to transform the GW data into a
        torch Tensor via:
            * in case of MultibandedFrequencyDomain: decimate data from base domain
            * whiten and scale strain (the inference network expects standardized data)
            * repackage strains and asds from dicts to an array
            * convert array to torch tensor on the correct device
            * extract only strain/waveform from the sample

        self.transform_pre is used once in the beginning of self._run_sampler. For
        GWSampler, this is the only preprocessing transform, for GWSamplerGNPE there is
        an additional transform self.transform_gnpe_loop_pre, which is applied once in
        every GNPE iteration, prior to sampling from the neural network.
        """
        # preprocessing transforms:
        transform_pre = []
        #   * in case of MultibandedFrequencyDomain: decimate data from base domain
        if isinstance(self.domain, MultibandedFrequencyDomain):
            transform_pre.append(
                DecimateWaveformsAndASDS(self.domain, decimation_mode="whitened")
            )
        #   * whiten and scale strain (the inference network expects standardized data)
        #   * repackage strains and asds from dicts to an array
        #   * convert array to torch tensor on the correct device
        #   * extract only strain/waveform from the sample
        transform_pre += [
            WhitenAndScaleStrain(self.domain.noise_std),
            # Use base metadata so that unconditional samplers still know how to
            # transform data, since this transform is used by the GNPE sampler as
            # well.
            RepackageStrainsAndASDS(
                self.base_model_metadata["train_settings"]["data"]["detectors"],
                first_index=self.domain.min_idx,
            ),
        ]
        if self.frequency_masking:
            transform_pre.append(
                ApplyFrequencyMasking(
                    domain=self.domain,
                    f_min_upper=self.frequency_masking.get("f_min", None),
                    f_max_lower=self.frequency_masking.get("f_max", None),
                    deterministic=True,
                )
            )
        transform_pre += [ToTorch(device=self.model.device), GetItem("waveform")]
        self.transform_pre = Compose(transform_pre)


class GWSampler(GWSamplerMixin, Sampler):
    """
    Sampler for gravitational-wave inference using neural posterior estimation.
    Augments the base class by defining transform_pre and transform_post to prepare
    data for the inference network.

    transform_pre :
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
        # preprocessing transforms
        self.initialize_transforms_pre()

        # postprocessing transforms
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

    def _initialize_transforms_gnpe_loop(self):
        """
        Builds the transforms that are used in the GNPE loop.
        """
        # When working with MultibandedFrequencyDomain, the data initially needs to be
        # in self.domain.base_domain, as heterodyning with gnpe-chirp (in
        # self.transform_pre, if applied) needs to be applied before decimation
        # to self.domain.
        base_domain = getattr(self.domain, "base_domain", self.domain)

        data_settings = self.metadata["train_settings"]["data"]
        ifo_list = InterferometerList(data_settings["detectors"])

        gnpe_time_settings = data_settings.get("gnpe_time_shifts")
        gnpe_chirp_settings = data_settings.get("gnpe_chirp")
        if (
            not gnpe_time_settings
            and not gnpe_chirp_settings
            and not gnpe_phase_settings
        ):
            raise KeyError(
                "GNPE inference requires network trained for either chirp mass, "
                "coalescence time, or phase GNPE."
            )

        # Set transform_gnpe_loop_pre. These transform are applied prior to each
        # sampling step in the gnpe loop.
        #   * reset the sample (e.g., clone non-gnpe transformed waveform)
        #   * blurring detector times to obtain gnpe proxies
        #   * shifting the strain by - gnpe proxies
        #   * repackaging & standardizing proxies to sample['context_parameters']
        #     for conditioning of the inference network
        transform_gnpe_loop_pre = [RenameKey("data", "waveform")]

        if gnpe_chirp_settings:
            if base_domain == self.domain:
                # If the initial data domain (base_domain) is the domain of the model
                # (self.domain) [i.e., we don't apply decimation], we can simply apply
                # the gnpe chirp transformation.
                transform_gnpe_loop_pre.append(
                    GNPEChirp(
                        gnpe_chirp_settings["kernel"],
                        base_domain,
                        gnpe_chirp_settings.get("order", 0),
                        inference=True,
                    )
                )
            else:
                # If we apply decimation (i.e., base_domain != self.domain), we need to
                # make sure to apply the gnpe chirp transformation *before* the
                # decimation, as decimation and the gnpe chirp phase transformation do
                # not commute.
                if "chirp_mass_proxy" in self.fixed_context_parameters:
                    # If we use a fixed chirp mass proxy, we can heterodyne the data in
                    # self.transform_pre (as opposed to transform_gnpe_loop_pre in the
                    # gnpe loop), which is much more efficient, as we only need to
                    # carry around the decimated strain. Fixed chirp mass proxies can
                    # e.g. be used when running single-iteration gnpe for the chirp
                    # mass as a form of prior conditioning. This is possible when the
                    # chirp mass posterior is tighter than the corresponding gnpe kernel.
                    #
                    # Specifically:
                    # (1) add the heterodyning transformation to self.transform_pre
                    # (2) omit the gnpe_chirp_transformation (i.e., no heterodyning in
                    #     the loop, but nevertheless condition the network on the proxy)
                    fixed_parameters = {
                        k[: -len("_proxy")]: v
                        for k, v in self.fixed_context_parameters.items()
                    }
                    self.transform_pre.transforms.insert(
                        0,
                        HeterodynePhase(
                            domain=base_domain,
                            order=gnpe_chirp_settings.get("order", 0),
                            inverse=False,
                            fixed_parameters=fixed_parameters,
                        ),
                    )

                else:
                    # If we don't use fixed chirp mass proxies (i.e., we gnpe-iterate
                    # the chirp mass), then we need to remove the
                    # DecimateWaveformsAndASDS and WhitenAndScaleStrain transformations
                    # from self.transform_pre, and move them *after* the gnpe chirp
                    # transform in transform_gnpe_loop_pre instead. This is currently
                    # not implemented, as this becomes extremely expensive: We need to
                    # store num_samples batches of the undecimated strain (with
                    # potentially >1e5 bins) on the GPU, which could be >20 Gb.
                    raise NotImplementedError()

        if gnpe_time_settings:
            transform_gnpe_loop_pre.append(
                GNPECoalescenceTimes(
                    ifo_list,
                    gnpe_time_settings["kernel"],
                    gnpe_time_settings["exact_equiv"],
                    inference=True,
                )
            )
            transform_gnpe_loop_pre.append(TimeShiftStrain(ifo_list, self.domain))

        transform_gnpe_loop_pre.append(
            SelectStandardizeRepackageParameters(
                {"context_parameters": data_settings["context_parameters"]},
                data_settings["standardization"],
                device=self.model.device,
            )
        )
        transform_gnpe_loop_pre.append(RenameKey("waveform", "data"))

        # Extract GNPE information (list of parameters, dict of kernels) from the
        # transforms.
        self.gnpe_parameters = []
        self.gnpe_kernel = PriorDict()
        for transform in transform_gnpe_loop_pre:
            if isinstance(transform, GNPEBase):
                self.gnpe_parameters += transform.input_parameter_names
                for k, v in transform.kernel.items():
                    self.gnpe_kernel[k] = v
        fixed_gnpe_parameters = [
            k[: -len("_proxy")]
            for k in self.fixed_context_parameters.keys()
            if k.endswith("_proxy")
        ]
        self.gnpe_parameters += fixed_gnpe_parameters
        print("GNPE parameters: ", self.gnpe_parameters)
        print("GNPE parameters fixed (not iterated): ", fixed_gnpe_parameters)
        print("GNPE kernel: ", self.gnpe_kernel)

        self.transform_gnpe_loop_pre = Compose(transform_gnpe_loop_pre)

        # transforms for gnpe loop, to be applied after sampling step:
        #   * de-standardization of parameters
        #   * post correction for geocent time (required for gnpe_time with exact
        #     equivariance)
        #   * computation of detectortimes from parameters (required for next gnpe
        #     iteration)
        transform_gnpe_loop_post = [
            SelectStandardizeRepackageParameters(
                {"inference_parameters": self.inference_parameters},
                data_settings["standardization"],
                inverse=True,
                as_type="dict",
            )
        ]
        if gnpe_time_settings and gnpe_time_settings.get("exact_equiv"):
            transform_gnpe_loop_post.append(PostCorrectGeocentTime())

        transform_gnpe_loop_post.append(
            CopyToExtrinsicParameters(
                "ra", "dec", "geocent_time", "chirp_mass", "mass_ratio", "phase"
            )
        )

        if {"geocent_time", "ra", "dec"}.issubset(self.inference_parameters):
            transform_gnpe_loop_post.append(
                GetDetectorTimes(ifo_list, data_settings["ref_time"])
            )

        self.transform_gnpe_loop_post = Compose(transform_gnpe_loop_post)

    def _initialize_transforms(self):
        """
        Initialize the transform
            * self.transfrom_pre
            * self.transform_gnpe_loop_pre
            * self.transform_gnpe_loop_post
        """
        # preprocessing transforms (self.transfrom_pre)
        self.initialize_transforms_pre()
        # transforms for gnpe loop
        # (self.transform_gnpe_loop_pre/self.transform_gnpe_loop_post)
        self._initialize_transforms_gnpe_loop()

    def _kernel_log_prob(self, samples):
        # TODO: Reimplement as a method of GNPEBase.
        if len({"chirp_mass", "mass_ratio", "phase"} & self.gnpe_kernel.keys()) > 0:
            raise NotImplementedError("kernel log_prob only implemented for time gnpe.")
        gnpe_proxies_diff = {
            k: np.array(samples[k] - samples[f"{k}_proxy"])
            for k in self.gnpe_kernel.keys()
        }
        # When using only fixed proxies (i.e., no GNPE iterations), return log probs = 0.
        if not gnpe_proxies_diff:
            return np.zeros(len(list(samples.values())[0]))
        return self.gnpe_kernel.ln_prob(gnpe_proxies_diff, axis=0)
