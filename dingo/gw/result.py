import time
from typing import Optional, Union

import numpy as np
import pandas as pd
from bilby.core.prior import Uniform, Constraint

from dingo.core.density import interpolated_sample_and_log_prob_multi, \
    interpolated_log_prob_multi
from dingo.core.multiprocessing import apply_func_with_multiprocessing
from dingo.core.result import Result as CoreResult
from dingo.gw.domains import build_domain
from dingo.gw.gwutils import get_extrinsic_prior_dict, get_window_factor
from dingo.gw.likelihood import StationaryGaussianGWLikelihood
from dingo.gw.prior import build_prior_with_defaults


class Result(CoreResult):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.synthetic_phase_kwargs = None

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

    def _build_prior(self):
        """Build the prior based on model metadata. Called by __init__()."""
        intrinsic_prior = self.base_model_metadata["dataset_settings"][
            "intrinsic_prior"
        ]
        extrinsic_prior = get_extrinsic_prior_dict(
            self.base_model_metadata["train_settings"]["data"]["extrinsic_prior"]
        )
        self.prior = build_prior_with_defaults({**intrinsic_prior, **extrinsic_prior})

        # Split off prior over geocent_time if samples appear to be time-marginalized.
        # This needs to be saved to initialize the likelihood.
        if (
            "geocent_time" in self.prior.keys()
            and "geocent_time" not in self.inference_parameters
        ):
            self.geocent_time_prior = self.prior.pop("geocent_time")
        else:
            self.geocent_time_prior = None
        # Split off prior over phase if samples appear to be phase-marginalized.
        if "phase" in self.prior.keys() and "phase" not in self.inference_parameters:
            self.phase_prior = self.prior.pop("phase")
        else:
            self.phase_prior = None

    # _build_likelihood is called at the beginning of Sampler.importance_sample

    def _build_likelihood(
            self,
            time_marginalization_kwargs: Optional[dict] = None,
            phase_marginalization_kwargs: Optional[dict] = None,
            phase_grid: Optional[np.ndarray] = None,
    ):
        """
        Build the likelihood function based on model metadata. This is called at the
        beginning of importance_sample().

        Parameters
        ----------
        time_marginalization_kwargs: dict, optional
            kwargs for time marginalization. At this point the only kwarg is n_fft,
            which determines the number of FFTs used (higher n_fft means better
            accuracy, at the cost of longer computation time).
        phase_marginalization_kwargs: dict, optional
            kwargs for phase marginalization.
        """
        if time_marginalization_kwargs is not None:
            if self.geocent_time_prior is None:
                raise NotImplementedError(
                    "Time marginalization is not compatible with "
                    "non-marginalized network."
                )
            if type(self.geocent_time_prior) != Uniform:
                raise NotImplementedError(
                    "Only uniform time prior is supported for time marginalization."
                )
            time_marginalization_kwargs["t_lower"] = self.geocent_time_prior.minimum
            time_marginalization_kwargs["t_upper"] = self.geocent_time_prior.maximum

        if phase_marginalization_kwargs is not None:
            # check that phase prior is uniform [0, 2pi)
            if not (
                    isinstance(self.phase_prior, Uniform)
                    and (self.phase_prior._minimum, self.phase_prior._maximum)
                    == (0, 2 * np.pi)
            ):
                raise ValueError(
                    f"Phase prior should be uniform [0, 2pi) for phase "
                    f"marginalization, but is {self.phase_prior}."
                )

        # The detector reference positions during likelihood evaluation should be based
        # on the event time, since any post-correction to account for the training
        # reference time has already been applied to the samples.

        if self.event_metadata is not None and "time_event" in self.event_metadata:
            t_ref = self.event_metadata["time_event"]
        else:
            t_ref = self.t_ref

        self.likelihood = StationaryGaussianGWLikelihood(
            wfg_kwargs=self.base_model_metadata["dataset_settings"][
                "waveform_generator"
            ],
            wfg_domain=build_domain(
                self.base_model_metadata["dataset_settings"]["domain"]
            ),
            data_domain=self.domain,
            event_data=self.context,
            t_ref=t_ref,
            time_marginalization_kwargs=time_marginalization_kwargs,
            phase_marginalization_kwargs=phase_marginalization_kwargs,
            phase_grid=phase_grid,
        )

    def _sample_synthetic_phase(
        self, samples: Union[dict, pd.DataFrame], synthetic_phase_kwargs, inverse: bool
            = False,
    ):
        """
        Sample a synthetic phase for the waveform. This is a post-processing step
        applicable to samples theta in the full parameter space, except for the phase
        parameter (i.e., 14D samples). This step adds a phase parameter to the samples
        based on likelihood evaluations.

        A synthetic phase is sampled as follows.

            * Compute and cache the modes for the waveform mu(theta, phase=0) for
              phase 0, organize them such that each contribution m transforms as
              exp(-i * m * phase).
            * Compute the likelihood on a phase grid, by computing mu(theta, phase) from
              the cached modes. In principle this likelihood is exact, however, it can
              deviate slightly from the likelihood computed without cached modes for
              various technical reasons (e.g., slightly different windowing of cached
              modes compared to full waveform when transforming TD waveform to FD).
              These small deviations can be fully accounted for by importance sampling.
              *Note*: when approximation_22_mode=True, the entire waveform is assumed
              to transform as exp(2i*phase), in which case the likelihood is only exact
              if the waveform is fully dominated by the (2, 2) mode.
            * Build a synthetic conditional phase distribution based on this grid. We
              use an interpolated prior distribution bilby.core.prior.Interped,
              such that we can sample and also evaluate the log_prob. We add a constant
              background with weight self.synthetic_phase_kwargs to the kde to make
              sure that we keep a mass-covering property. With this, the importance
              sampling will yield exact results even when the synthetic phase conditional
              is just an approximation.

        Besides adding phase samples to samples['phase'], this method also modifies
        samples['log_prob'] by adding the log_prob of the synthetic phase conditional.

        This method modifies the samples in place.

        Parameters
        ----------
        samples : dict or pd.DataFrame
        inverse : bool, default True
            Whether to apply instead the inverse transformation. This is used prior to
            calculating the log_prob. In inverse mode, the posterior probability over
            phase is calculated for given samples. It is stored in sample['log_prob'].

        """

        # TODO: Possibly remove this class attribute. Decide where to store information.
        self.synthetic_phase_kwargs = synthetic_phase_kwargs

        approximation_22_mode = self.synthetic_phase_kwargs.get(
            "approximation_22_mode", True
        )

        if not (
            isinstance(self.phase_prior, Uniform)
            and (self.phase_prior._minimum, self.phase_prior._maximum) == (0, 2 * np.pi)
        ):
            raise ValueError(
                f"Phase prior should be uniform [0, 2pi) to work with synthetic phase."
                f" However, the prior is {self.phase_prior}."
            )

        # This builds on the Bilby approach to sampling the phase when using a
        # phase-marginalized likelihood.

        # Restrict to samples that are within the prior.
        param_keys = [k for k, v in self.prior.items() if not isinstance(v, Constraint)]
        theta = pd.DataFrame(samples)[param_keys]
        log_prior = self.prior.ln_prob(theta, axis=0)
        constraints = self.prior.evaluate_constraints(theta)
        np.putmask(log_prior, constraints == 0, -np.inf)
        within_prior = log_prior != -np.inf

        # Put a cap on the number of processes to avoid overhead:
        num_valid_samples = np.sum(within_prior)
        num_processes = min(
            self.synthetic_phase_kwargs.get("num_processes", 1), num_valid_samples // 10
        )

        if num_valid_samples > 1e4:
            print(f"Estimating synthetic phase for {num_valid_samples} samples.")
            t0 = time.time()

        if not inverse:
            # TODO: This can probably be removed.
            self._build_likelihood()

        if inverse:
            # We estimate the log_prob for given phases, so first save the evaluation
            # points.
            sample_phase = theta["phase"].to_numpy(copy=True)

        # For each sample, build the posterior over phase given the remaining parameters.

        phases = np.linspace(0, 2 * np.pi, self.synthetic_phase_kwargs["n_grid"])
        if approximation_22_mode:
            # For each sample, the un-normalized posterior depends only on (d | h(phase)):
            # The prior p(phase), and the inner products (h | h), and (d | d) only contribute
            # to the normalization. (We check above that p(phase) is constant.)
            theta["phase"] = 0.0
            d_inner_h_complex = self.likelihood.d_inner_h_complex_multi(
                theta.iloc[within_prior],
                num_processes,
            )

            # Evaluate the log posterior over the phase across the grid.
            phasor = np.exp(2j * phases)
            phase_log_posterior = np.outer(d_inner_h_complex, phasor).real
        else:
            self.likelihood.phase_grid = phases

            phase_log_posterior = apply_func_with_multiprocessing(
                self.likelihood.log_likelihood_phase_grid,
                theta.iloc[within_prior],
                num_processes=num_processes,
            )

        phase_posterior = np.exp(
            phase_log_posterior - np.amax(phase_log_posterior, axis=1, keepdims=True)
        )
        # Include a floor value to maintain mass coverage.
        phase_posterior += phase_posterior.mean(
            axis=-1, keepdims=True
        ) * self.synthetic_phase_kwargs.get("uniform_weight", 0.01)

        if not inverse:
            # Forward direction:
            #   (1) Sample a new phase according to the synthetic posterior.
            #   (2) Add the log_prob to the existing log_prob.

            new_phase, delta_log_prob = interpolated_sample_and_log_prob_multi(
                phases,
                phase_posterior,
                num_processes,
            )

            phase_array = np.full(len(theta), 0.0)
            phase_array[within_prior] = new_phase
            delta_log_prob_array = np.full(len(theta), -np.inf)
            delta_log_prob_array[within_prior] = delta_log_prob

            samples["phase"] = phase_array
            samples["log_prob"] += delta_log_prob_array

            # Insert the phase prior in the prior, since now the phase is present.
            self.prior["phase"] = self.phase_prior
            # reset likelihood for safety
            # TODO: Can this be removed?
            self.likelihood = None

        else:
            # TODO: Possibly remove.
            # Inverse direction:
            #   (1) Evaluate the synthetic log prob for given phase points, and save it.

            log_prob = interpolated_log_prob_multi(
                phases,
                phase_posterior,
                sample_phase[within_prior],
                num_processes,
            )

            # Outside of prior, set log_prob to -np.inf.
            log_prob_array = np.full(len(theta), -np.inf)
            log_prob_array[within_prior] = log_prob
            samples["log_prob"] = log_prob_array
            del samples["phase"]

        if num_valid_samples > 1e4:
            print(f"Done. This took {time.time() - t0:.2f} s.")
