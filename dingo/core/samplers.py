import time
from pathlib import Path
from typing import Optional, Union, Dict
import sys

import numpy
import numpy as np
import pandas as pd
import math
import torch
from torchvision.transforms import Compose
import tempfile

from dingo.core.models import PosteriorModel
from dingo.core.result import Result
from dingo.core.density import train_unconditional_density_estimator
from dingo.core.utils import torch_detach_to_cpu, IterationTracker

# FIXME: transform below should be in core
from dingo.gw.transforms import SelectStandardizeRepackageParameters

#
# Sampler classes are motivated by the approach of Bilby.
#


class Sampler(object):
    """
    Sampler class that wraps a PosteriorModel. Allows for conditional and unconditional
    models.

    Draws samples from the model based on (optional) context data. Also implements
    importance sampling.

    Methods
    -------
        run_sampler
        log_prob
        importance_sample
        to_samples_dataset
        to_hdf5
        print_summary
    """

    def __init__(
        self,
        model: PosteriorModel = None,
        samples_dataset: Result = None,
    ):
        """
        Parameters
        ----------
        model : PosteriorModel
        """
        if (model is None) + (samples_dataset is None) != 1:
            raise ValueError(
                "Sampler should be initialized with either a model, or samples_dataset."
            )

        self.model = model
        self.samples_dataset = samples_dataset

        if self.model is not None:
            self.metadata = self.model.metadata.copy()

            if self.metadata["train_settings"]["data"].get("unconditional", False):
                self.unconditional_model = True
                # For unconditional models, the context will be stored with the model. It
                # is needed for calculating the likelihood for importance sampling.
                # However, it will not be used when sampling from the model, since it is
                # unconditional.
                self.context = self.model.context
                # TODO: Set the event metadata -> for GNPESampler or all? Here?
            else:
                self.unconditional_model = False
                self.context = None

            if "base" in self.metadata:
                self.base_model_metadata = self.metadata["base"]
            else:
                self.base_model_metadata = self.metadata

            self.inference_parameters = self.metadata["train_settings"]["data"][
                "inference_parameters"
            ]

        elif self.samples_dataset is not None:
            self.metadata = self.samples_dataset.settings.copy()
            self.unconditional_model = True
            self.context = self.samples_dataset.context
            self.base_model_metadata = self.samples_dataset.settings
            self.samples = self.samples_dataset.samples
            data_settings = self.base_model_metadata["train_settings"]["data"]
            self.inference_parameters = data_settings["inference_parameters"]

        self.transform_pre = Compose([])
        self.transform_post = Compose([])
        self._build_domain()
        self._reset_result()

        # keys with attributes to be saved by self.to_hdf5()
        self.samples_dataset_keys = [
            "settings",
            "samples",
            "context",
            "log_evidence",
            "n_eff",
            "effective_sample_size",
        ]
        self._pesummary_package = "core"
        self._result_class = Result

    def _reset_result(self):
        """Clear out all data produced by self.run_sampler(), to prepare for the next
        sampler run."""
        self.samples = None
        self.log_evidence = None
        self.effective_sample_size = None
        self.n_eff = None

    @property
    def context(self):
        """Data on which to condition the sampler. For injections, there should be a
        'parameters' key with truth values."""
        return self._context

    @context.setter
    def context(self, value):
        if value is not None:
            self._check_context(value)
            if "parameters" in value:
                self.metadata["injection_parameters"] = value.pop("parameters")
        self._context = value

    def _check_context(self, context: Optional[dict] = None):
        # TODO: Add some checks that the context is appropriate.
        pass

    @property
    def event_metadata(self):
        """Metadata for data analyzed. Can in principle influence any post-sampling
        parameter transformations (e.g., sky position correction), as well as the
        likelihood detector positions."""
        return self.base_model_metadata.get("event")

    @event_metadata.setter
    def event_metadata(self, value):
        self.base_model_metadata["event"] = value

    def prepare_log_prob(self, *args, **kwargs):
        pass

    def _run_sampler(
        self,
        num_samples: int,
        context: Optional[dict] = None,
        # **kwargs,
    ) -> dict:
        if not self.unconditional_model:
            if context is None:
                raise ValueError("Context required to run sampler.")
            x = context.copy()
            x["parameters"] = {}
            x["extrinsic_parameters"] = {}

            # transforms_pre are expected to transform the data in the same way for each
            # requested sample. We therefore expand it across the batch *after*
            # pre-processing.
            x = self.transform_pre(context)
            x = x.expand(num_samples, *x.shape)
            x = [x]
            # The number of samples is expressed via the first dimension of x,
            # so we must pass num_samples = 1 to sample_and_log_prob().
            num_samples = 1
        else:
            if context is not None:
                print("Unconditional model. Ignoring context.")
            x = []

        # For a normalizing flow, we get the log_prob for "free" when sampling,
        # so we always include this. For other architectures, it may make sense to
        # have a flag for whether to calculate the log_prob.
        self.model.model.eval()
        with torch.no_grad():
            y, log_prob = self.model.model.sample_and_log_prob(
                *x, num_samples=num_samples
            )

        samples = self.transform_post({"parameters": y, "log_prob": log_prob})
        result = samples["parameters"]
        result["log_prob"] = samples["log_prob"]
        return result

    def run_sampler(
        self,
        num_samples: int,
        batch_size: Optional[int] = None,
        **kwargs,
    ):
        """
        Generates samples and stores them as class attribute.

        Allows for batched sampling, e.g., if limited by GPU memory. Actual sampling is
        performed by _run_sampler().

        Parameters
        ----------
        num_samples : int
            Number of samples requested.
        batch_size : int, optional
            Batch size for sampler.
        get_log_prob  :  bool = False
            If set, compute log_prob for each sample
        """
        self._reset_result()

        if self.samples_dataset is not None:
            # if self.samples_dataset is set, running the
            samples = self.samples_dataset.samples.sample(
                num_samples, ignore_index=True
            )
            samples = {k: np.array(samples[k]) for k in samples.columns}
            self._post_process(samples)
            self.samples = pd.DataFrame(samples)
            return

        print(f"Running sampler to generate {num_samples} samples.")
        t0 = time.time()
        if not self.unconditional_model:
            if self.context is None:
                raise ValueError("Context must be set in order to run sampler.")
            context = self.context
        else:
            context = None

        # Carry out batched sampling by calling _run_sample() on each batch and
        # consolidating the results.
        if batch_size is None:
            batch_size = num_samples
        full_batches, remainder = divmod(num_samples, batch_size)
        samples = [
            self._run_sampler(batch_size, context, **kwargs)
            for _ in range(full_batches)
        ]
        if remainder > 0:
            samples.append(self._run_sampler(remainder, context))
        samples = {p: torch.cat([s[p] for s in samples]) for p in samples[0].keys()}
        # get_log_prob = True
        # if get_log_prob and "log_prob" not in samples.keys():
        #     log_prob = self.log_prob_mc(
        #         samples, n_mc=1000, batch_size=batch_size,
        #     )
        # # samples = {p: torch.cat([s[p] for s in samples]) for p in k_parameters}
        samples = {k: v.cpu().numpy() for k, v in samples.items()}

        # Apply any post-sampling transformation to sampled parameters (e.g.,
        # correction for t_ref or sampling of synthetic phase), and place on CPU.
        self._post_process(samples)
        self.samples = pd.DataFrame(samples)
        print(f"Done. This took {time.time() - t0:.1f} s.")
        sys.stdout.flush()

    def log_prob(self, samples: pd.DataFrame) -> np.ndarray:
        """
        Calculate the model log probability at specific sample points.

        Parameters
        ----------
        samples : pd.DataFrame
            Sample points at which to calculate the log probability.

        Returns
        -------
        np.array of log probabilities.
        """
        if self.context is None and not self.unconditional_model:
            raise ValueError("Context must be set in order to calculate log_prob.")

        # This undoes any post-correction that would have been done to the samples,
        # before evaluating the log_prob. E.g., the t_ref / sky position correction.
        samples = samples.copy()
        self._post_process(samples, inverse=True)

        # Standardize the sample parameters and place on device.
        y = samples[self.inference_parameters].to_numpy()
        standardization = self.metadata["train_settings"]["data"]["standardization"]
        mean = np.array([standardization["mean"][p] for p in self.inference_parameters])
        std = np.array([standardization["std"][p] for p in self.inference_parameters])
        y = (y - mean) / std
        y = torch.from_numpy(y).to(device=self.model.device, dtype=torch.float32)

        if not self.unconditional_model:
            x = self.context.copy()
            x["parameters"] = {}
            x["extrinsic_parameters"] = {}

            # Context is the same for each sample. Expand across batch dimension after
            # pre-processing.
            x = self.transform_pre(self.context)
            x = x.expand(len(samples), *x.shape)
            x = [x]
        else:
            x = []

        self.model.model.eval()
        with torch.no_grad():
            log_prob = self.model.model.log_prob(y, *x)

        log_prob = log_prob.cpu().numpy()
        log_prob -= np.sum(np.log(std))

        # Pre-processing step may have included a log_prob with the samples.
        if "log_prob" in samples:
            log_prob += samples["log_prob"].to_numpy()

        return log_prob

    def log_prob_mc(self, *args, **kwargs):
        pass

    def _post_process(self, samples: Union[dict, pd.DataFrame], inverse: bool = False):
        pass

    def _build_domain(self):
        self.domain = None

    def write_pesummary(self, filename):
        from pesummary.io import write
        from pesummary.utils.samples_dict import SamplesDict

        samples_dict = SamplesDict(self.samples)
        write(
            samples_dict,
            package=self._pesummary_package,
            file_format="hdf5",
            filename=filename,
        )
        # TODO: Save much more information.

    @property
    def settings(self):
        return self.metadata

    def to_samples_dataset(self) -> Result:
        data_dict = {k: getattr(self, k) for k in self.samples_dataset_keys}
        data_keys = [k for k in data_dict.keys() if k != "settings"]
        return self._result_class(dictionary=data_dict, data_keys=data_keys)

    def to_hdf5(self, label="", outdir="."):
        dataset = self.to_samples_dataset()
        file_name = "dingo_samples_" + label + ".hdf5"
        dataset.to_file(file_name=Path(outdir, file_name))

    def print_summary(self):
        print("Number of samples:", len(self.samples))
        if self.log_evidence is not None:
            print(
                f"Log(evidence): {self.log_evidence:.3f} +-{self.log_evidence_std:.3f}"
            )
            print(
                f"Effective samples {self.n_eff:.1f}: "
                f"(ESS = {100 * self.effective_sample_size:.2f}%)"
            )


class UnconditionalSampler(Sampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.unconditional_model = True
        self._initialize_transforms()

    def _initialize_transforms(self):
        # Postprocessing transform only:
        #   * De-standardize data and extract inference parameters. Be careful to use
        #     the standardization of the correct model, not the base model.
        self.transform_post = SelectStandardizeRepackageParameters(
            {"inference_parameters": self.inference_parameters},
            self.metadata["train_settings"]["data"]["standardization"],
            inverse=True,
            as_type="dict",
        )


class GNPESampler(Sampler):
    """
    Base class for GNPE sampler. It wraps a PosteriorModel, and must contain also an NPE
    sampler, which is used to generate initial samples.
    """

    def __init__(
        self,
        model: PosteriorModel,
        init_sampler: Sampler,
        num_iterations: int = 1,
    ):
        """
        Parameters
        ----------
        model : PosteriorModel
        init_sampler : Sampler
            Used for generating initial samples
        num_iterations : int
            Number of GNPE iterations to be performed by sampler.
        """
        super().__init__(model)
        self.init_sampler = init_sampler
        self.num_iterations = num_iterations
        self.gnpe_parameters = None  # Should be set in subclass.
        self.gnpe_proxy_sampler = None
        self.log_prob_correction = None  # log_prob correction, accounting for std
        self.iteration_tracker = None
        # remove self.remove_init_outliers of lowest log_prob init samples before gnpe
        self.remove_init_outliers = 0.0

    @property
    def init_sampler(self):
        return self._init_sampler

    @init_sampler.setter
    def init_sampler(self, value):
        self._init_sampler = value
        # Copy this so it persists if we delete the init model.
        self.metadata["init_model"] = self._init_sampler.model.metadata.copy()
        if self._init_sampler.unconditional_model:
            self.context = self._init_sampler.context

    @property
    def num_iterations(self):
        """The number of GNPE iterations to perform when sampling."""
        return self._num_iterations

    @num_iterations.setter
    def num_iterations(self, value):
        self._num_iterations = value
        self.metadata["num_iterations"] = self._num_iterations

    def _kernel_log_prob(self, samples):
        raise NotImplementedError("To be implemented in subclass.")

    def _run_sampler(
        self,
        num_samples: int,
        context: Optional[dict] = None,
        # use_gnpe_proxy_sampler=False,
    ) -> dict:
        if context is None:
            raise ValueError("self.context must be set to run sampler.")
        # use_gnpe_proxy_sampler = self.gnpe_proxy_sampler is not None

        data_ = self.init_sampler.transform_pre(context)
        init_samples = self.init_sampler._run_sampler(num_samples, context)

        # We could be starting with either the GNPE parameters *or* their proxies,
        # depending on the nature of the initialization network.

        start_with_proxies = False
        proxy_log_prob = None
        proxies = {}

        if {p + '_proxy' for p in self.gnpe_parameters}.issubset(init_samples.keys()):
            start_with_proxies = True
            proxy_log_prob = init_samples["log_prob"]
            proxies = {
                    k + '_proxy': init_samples[k + '_proxy'] for k in
                    self.gnpe_parameters
                }
            init_proxies = proxies.copy()

        # TODO: Possibly remove outliers from init_samples. Only do this if running
        #  several Gibbs iterations.

        x = {"extrinsic_parameters": init_samples, "parameters": {}}

        #
        # Gibbs sample.
        #

        for i in range(self.num_iterations):
            print(i)
            if start_with_proxies and i == 0:
                x["extrinsic_parameters"] = proxies.copy()
            else:
                x["extrinsic_parameters"] = {
                    k: x["extrinsic_parameters"][k] for k in self.gnpe_parameters
                }

            d = data_.clone()
            x["data"] = d.expand(num_samples, *d.shape)

            x = self.transform_pre(x)

            self.model.model.eval()
            with torch.no_grad():
                y, log_prob = self.model.model.sample_and_log_prob(
                    x["data"], x["context_parameters"]
                )

            x['parameters'] = y
            x['log_prob'] = log_prob
            x = self.transform_post(x)

        #
        # Prepare final result.
        #

        if start_with_proxies and self.num_iterations == 1:
            # In this case it makes sense to save the log_prob and the proxy parameters.

            samples = x["parameters"]
            samples["log_prob"] = x["log_prob"] + proxy_log_prob

            # The log_prob returned by gnpe is not just the log_prob over parameters
            # theta, but instead the log_prob in the *joint* space q(theta,theta^|x),
            # including the proxies theta^. For importance sampling this means,
            # that the target density is
            #
            #       p(theta,theta^|x) = p(theta^|theta) * p(theta|x).
            #
            # We compute log[p(theta^|theta)] below and store it as
            # samples["delta_log_prob_target"], such that for importance sampling we
            # only need to evaluate log[p(theta|x)] and add this correction.

            # Proxies should be sitting in extrinsic_parameters.
            all_params = {**x["extrinsic_parameters"], **samples}
            all_params = {k: torch_detach_to_cpu(v) for k, v in all_params.items()}
            kernel_log_prob = self._kernel_log_prob(all_params)
            samples["delta_log_prob_target"] = torch.Tensor(kernel_log_prob)

        else:
            # Otherwise we only save the inference parameters, and no log_prob.
            # Alternatively we could save the entire chain and the log_prob, but this
            # is not useful for our purposes.

            # Note that we do not have access to the detector_time proxy variables if
            # there is more than one iteration, since the GNPECoalescenceTimes transform
            # subtracts off the absolute time shift. Perhaps it would make sense to
            # rename the relative time shifts as H1_time_proxy_relative, etc.,
            # to preserve this information more easily.

            samples = x['parameters']

        # Extract the proxy parameters from x["extrinsic_parameters"]. These have
        # not been standardized. They are persistent from prior to inference,
        # since this is when they were placed here and their values should not have
        # changed.
        for k, v in x["extrinsic_parameters"].items():
            if k.endswith('_proxy'):
                proxies[k] = v

        samples.update(proxies)

        # Safety check for unconditional flows. Make sure the proxies haven't changed.
        if start_with_proxies and self.num_iterations == 1:
            for k in proxies:
                assert torch.equal(proxies[k], init_proxies[k])

        return samples

        # if not use_gnpe_proxy_sampler:
        #     # Run gnpe iterations to jointly infer gnpe proxies and inference parameters.
        #     self.iteration_tracker = IterationTracker(store_data=True)
        #     if self.remove_init_outliers == 0.0:
        #         init_samples = self.init_sampler._run_sampler(num_samples, context)
        #     else:
        #         init_samples = self.init_sampler._run_sampler(
        #             math.ceil(num_samples / (1 - self.remove_init_outliers)), context
        #         )
        #         thr = torch.quantile(
        #             init_samples["log_prob"], self.remove_init_outliers
        #         )
        #         inds = torch.where(init_samples["log_prob"] >= thr)[0][:num_samples]
        #         init_samples = {k: v[inds] for k, v in init_samples.items()}
        #     x = {"extrinsic_parameters": init_samples, "parameters": {}}
        #     for i in range(self.num_iterations):
        #         x["extrinsic_parameters"] = {
        #             k: x["extrinsic_parameters"][k] for k in self.gnpe_parameters
        #         }
        #         self.iteration_tracker.update(
        #             {k: v.cpu().numpy() for k, v in x["extrinsic_parameters"].items()}
        #         )
        #         d = data_.clone()
        #         x["data"] = d.expand(num_samples, *d.shape)
        #
        #         x = self.transform_pre(x)
        #         x["parameters"] = self.model.sample(x["data"], x["context_parameters"])
        #         x = self.transform_post(x)
        #         samples = x["parameters"]
        #         print(
        #             f"it {i}.\tmin pvalue: {self.iteration_tracker.pvalue_min:.3f}"
        #             f"\tproxy mean: ",
        #             *[
        #                 f"{torch.mean(v).item():.5f}"
        #                 for v in x["gnpe_proxies"].values()
        #             ],
        #             "\tproxy std:",
        #             *[f"{torch.std(v).item():.5f}" for v in x["gnpe_proxies"].values()],
        #         )
        # else:
        #     # Infer gnpe proxies based on trained model in gnpe_proxy sampler.
        #     # With this, we can infer the inference parameters with a single pass
        #     # through the gnpe network.
        #     d = data_.clone()
        #
        #     gnpe_proxies_in = self.gnpe_proxy_sampler._run_sampler(
        #         num_samples=num_samples, batch_size=num_samples
        #     )
        #     log_prob_gnpe_proxies = gnpe_proxies_in.pop("log_prob")
        #     # log_prob_gnpe_proxies is already standardized, since it is covered in
        #     # _run_sampler by self.transform_post [SelectStandardizeRepackageParameters]
        #
        #     x = {
        #         "extrinsic_parameters": {},
        #         "parameters": {},
        #         "gnpe_proxies_in": gnpe_proxies_in,
        #         "data": d.expand(num_samples, *d.shape),
        #     }
        #     x = self.transform_pre(x)
        #     with torch.no_grad():
        #         x["parameters"], x["log_prob"] = self.model.model.sample_and_log_prob(
        #             x["data"],
        #             x["context_parameters"],
        #         )
        #     x = self.transform_post(x)  # this also standardizes x["log_prob"]!
        #
        #     samples = x["parameters"]
        #     samples["log_prob"] = x["log_prob"] + log_prob_gnpe_proxies
        #     samples["log_prob_gnpe_proxies"] = log_prob_gnpe_proxies
        #
        #     # The log_prob returned by gnpe is not just the log_prob over parmeters
        #     # theta, but instead the log_prob in the *joint* space q(theta,theta^|x),
        #     # including the proxies theta^. For importance sampling this means,
        #     # that the target density is
        #     #
        #     #       p(theta,theta^|x) = p(theta^|theta) * p(theta|x).
        #     #
        #     # We compute log[p(theta^|theta)] below and store it as
        #     # samples["delta_log_prob_target"], such that for importance sampling we
        #     # only need to evaluate log[p(theta|x)] and add this correction.
        #     all_params = {**x["extrinsic_parameters"], **samples, **x["gnpe_proxies"]}
        #     all_params = {k: torch_detach_to_cpu(v) for k, v in all_params.items()}
        #     kernel_log_prob = self._kernel_log_prob(all_params)
        #     samples["delta_log_prob_target"] = torch.Tensor(kernel_log_prob)
        #
        # # add gnpe parameters:
        # for k, v in x["gnpe_proxies"].items():
        #     assert k.endswith("_proxy")
        #     samples["GNPE:" + k] = x["gnpe_proxies"][k]
        #
        # return samples
