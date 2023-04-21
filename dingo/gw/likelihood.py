import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.special import logsumexp
from bilby.gw.utils import ln_i0
from threadpoolctl import threadpool_limits

from dingo.core.likelihood import Likelihood
from dingo.gw.injection import GWSignal
from dingo.gw.waveform_generator import WaveformGenerator
from dingo.gw.domains import build_domain
from dingo.gw.data.data_preparation import get_event_data_and_domain


class StationaryGaussianGWLikelihood(GWSignal, Likelihood):
    """
    Implements GW likelihood for stationary, Gaussian noise.
    """

    def __init__(
        self,
        wfg_kwargs,
        wfg_domain,
        data_domain,
        event_data,
        t_ref=None,
        time_marginalization_kwargs=None,
        phase_marginalization_kwargs=None,
        calibration_marginalization_kwargs=None,
        phase_grid=None,
    ):
        # TODO: Does the phase_grid argument ever get used?
        """
        Parameters
        ----------
        wfg_kwargs: dict
            Waveform generator parameters (at least approximant and f_ref).
        wfg_domain : dingo.gw.domains.Domain
            Domain used for waveform generation. This can potentially deviate from the
            final domain, having a wider frequency range needed for waveform generation.
        data_domain: dingo.gw.domains.Domain
            Domain object for event data.
        event_data: dict
            GW data. Contains strain data in event_data["waveforms"] and asds in
            event_data["asds"].
        t_ref: float
            Reference time; true geocent time for GW is t_ref + theta["geocent_time"].
        time_marginalization_kwargs: dict
            Time marginalization parameters. If None, no time marginalization is used.
        calibration_marginalization_kwargs: dict
            Calibration marginalization parameters. If None, no calibration marginalization is used.
        phase_marginalization_kwargs: dict
            Phase marginalization parameters. If None, no phase marginalization is used.
        """
        super().__init__(
            wfg_kwargs=wfg_kwargs,
            wfg_domain=wfg_domain,
            data_domain=data_domain,
            ifo_list=list(event_data["waveform"].keys()),
            t_ref=t_ref,
        )

        self.asd = event_data["asds"]

        self.whitened_strains = {
            k: v / self.asd[k] / self.data_domain.noise_std
            for k, v in event_data["waveform"].items()
        }
        if len(list(self.whitened_strains.values())[0]) != data_domain.max_idx + 1:
            raise ValueError("Strain data does not match domain.")
        # log noise evidence, independent of theta and waveform model
        self.log_Zn = sum(
            [
                -1 / 2.0 * inner_product(d_ifo, d_ifo)
                for d_ifo in self.whitened_strains.values()
            ]
        )
        # For completeness (not used): there is a PSD-dependent contribution to the
        # likelihood,  which is typically ignored as it is constant for a given PSD
        # (e.g.,  for different waveform models) or event strains. Intuitively it is the
        # correction term that needs to be added to the log-likelihood to account
        # for the fact we compute N[0,1](strain/ASD) instead of N[0,ASD^2](strain).
        # But this contribution is typically ignored since we are only interested in
        # log-likelihood *differences*, see e.g. https://arxiv.org/pdf/1809.02293.pdf.
        # psi = - sum_i log(2pi * PSD_i) = - 2 * sum_i * log(2pi * ASD_i)
        # self.psi = -2 * sum(
        #     np.sum(np.log(2 * np.pi * asd)) for asd in self.asd.values()
        # )

        # Value in Veitch et al (2015)
        self.psi = -2 * np.sum(
            [
                np.sum(np.log(np.sqrt(2 * np.pi) * asd * self.data_domain.noise_std))
                for asd in self.asd.values()
            ]
        )
        self.whiten = True
        self.phase_grid = phase_grid

        # optionally initialize time marginalization
        self.time_marginalization = False
        if time_marginalization_kwargs is not None:
            self.initialize_time_marginalization(**time_marginalization_kwargs)

        # optionally initialize phase marginalization
        self.phase_marginalization = False
        if phase_marginalization_kwargs is not None:
            self.phase_marginalization = True
            # flag whether to use exp(2i * phi) approximation for phase transformations
            self.pm_approx_22_mode = phase_marginalization_kwargs.get(
                "approximation_22_mode", True
            )
            # if we don't use the exp(2i * phase) approximation, we need a phase grid
            if not self.pm_approx_22_mode:
                n_grid = phase_marginalization_kwargs.get("n_grid", 1_000)
                # use endpoint = False for grid, since phase = 0/2pi are equivalent
                self.phase_grid = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
            else:
                print("Using phase marginalization with (2,2) mode approximation.")

        # Initialize calibration marginalization using the setter from GWSignal.
        self.calibration_marginalization_kwargs = calibration_marginalization_kwargs

    def initialize_time_marginalization(self, t_lower, t_upper, n_fft=1):
        """
        Initialize time marginalization. Time marginalization can be performed via FFT,
        which is super fast. However, this limits the time resolution to delta_t =
        1/self.data_domain.f_max. In order to allow for a finer time resolution we
        compute the time marginalized likelihood n_fft via FFT on a grid of n_fft
        different time shifts [0, delta_t, 2*delta_t, ..., (n_fft-1)*delta_t] and
        average over the time shifts. The effective time resolution is thus

            delta_t_eff = delta_t / n_fft = 1 / (f_max * n_fft).

        Note: Time marginalization in only implemented for uniform time priors.

        Parameters
        ----------
        t_lower: float
            Lower time bound of the uniform time prior.
        t_upper: float
            Upper time bound of the uniform time prior.
        n_fft: int = 1
            Size of grid for FFT for time marginalization.
        """
        self.time_marginalization = True
        self.n_fft = n_fft
        delta_t = 1.0 / self.data_domain.f_max  # time resolution of FFT
        # time shifts for different FFTs
        self.t_FFT = np.arange(self.n_fft) * delta_t / self.n_fft

        self.shifted_strains = {}
        for idx, dt in enumerate(self.t_FFT):
            # Instead of shifting the waveform mu by + dt when computing the
            # time-marginalized likelihood, we shift the strain data by -dt. This saves
            # time for likelihood evaluations, since it can be precomputed.
            self.shifted_strains[dt] = {
                k: v * np.exp(-2j * np.pi * self.data_domain() * (-dt))
                for k, v in self.whitened_strains.items()
            }

        # Get the time prior. This will be multiplied with the result of the FFT.
        T = 1 / self.data_domain.delta_f
        time_axis = np.arange(len(self.data_domain())) / self.data_domain.f_max
        self.time_grid = time_axis[:, np.newaxis] + self.t_FFT[np.newaxis, :]
        active_indices = np.where(
            (self.time_grid >= t_lower) & (self.time_grid <= t_upper)
            | (self.time_grid - T >= t_lower) & (self.time_grid - T <= t_upper)
        )
        time_prior = np.zeros(self.time_grid.shape)
        time_prior[active_indices] = 1.0
        with np.errstate(divide="ignore"):  # ignore warnings for log(0) = -inf
            self.time_prior_log = np.log(time_prior / np.sum(time_prior))

    def log_likelihood(self, theta):
        if (
            sum(
                [
                    self.time_marginalization,
                    self.phase_marginalization,
                    bool(self.calibration_marginalization_kwargs),
                ]
            )
            > 1
        ):
            raise NotImplementedError(
                "Only one out of time-marginalization, phase-marginalization and calibration-marginalization "
                "can be used at a time."
            )

        if self.time_marginalization:
            return self._log_likelihood_time_marginalized(theta)
        elif self.phase_marginalization:
            return self._log_likelihood_phase_marginalized(theta)
        elif self.calibration_marginalization_kwargs:
            return self._log_likelihood_calibration_marginalized(theta)
        else:
            return self._log_likelihood(theta)

    def _log_likelihood(self, theta):
        """
        The likelihood is given by

                  log L(d|theta) = psi - 1/2. <d - mu(theta), d - mu(theta)>

        where psi is a waveform model independent (and thus irrelevant) constant. Here,
        we denote the strain data by d and the GW signal by mu(theta).
        [see e.g. arxiv.org/pdf/1809.02293, equation (44) for details]
        We expand this expression below to compute the log likelihood, and omit psi.
        The expaneded expression reads

                log L(d|theta) = log_Zn + kappa2(theta) - 1/2 rho2opt(theta),

                log_Zn = -1/2. <d, d>,
                kappa2(theta) = <d, mu(theta)>,
                rho2opt(theta) = <mu(theta), mu(theta)>.

        The noise-weighted inner product is defined as

                  <a, b> = 4 * delta_f * sum(a.conj() * b / PSD).real.

        Here, we work with data d and signals mu that are already whitened by
        1 / [sqrt(PSD) * domain.noise_std], where

                  noise_std = np.sqrt(window_factor) / np.sqrt(4 * delta_f).

        With this preprocessing, the inner products thus simply become

                  <a, b> = sum(a.conj() * b).real.

        ! Be careful with window factors here !


        Time marginalization:
        The above expansion of the likelihood is particularly useful for time
        marginalization, as only kappa2 depends on the time parameter.


        Parameters
        ----------
        theta: dict
            BBH parameters.

        Returns
        -------
        log_likelihood: float
        """

        # Step 1: Compute whitened GW strain mu(theta) for parameters theta.
        mu = self.signal(theta)["waveform"]
        d = self.whitened_strains

        # Step 2: Compute likelihood. log_Zn is precomputed, so we only need to
        # compute the remaining terms rho2opt and kappa2
        rho2opt = sum([inner_product(mu_ifo, mu_ifo) for mu_ifo in mu.values()])
        kappa2 = sum(
            [
                inner_product(d_ifo, mu_ifo)
                for d_ifo, mu_ifo in zip(d.values(), mu.values())
            ],
        )
        return self.log_Zn + kappa2 - 1 / 2.0 * rho2opt

    def log_likelihood_phase_grid(self, theta, phases=None):
        # TODO: Implement for time marginalization
        if self.phase_marginalization:
            raise ValueError(
                "Can't compute likelihood on a phase grid for "
                "phase-marginalized posteriors"
            )
        if self.time_marginalization:
            raise NotImplementedError(
                "log_likelihood on phase grid not yet implemented."
            )

        if self.waveform_generator.spin_conversion_phase != 0:
            raise ValueError(
                f"The log likelihood on a phase grid assumes "
                f"WaveformGenerator.spin_conversion_phase = 0, "
                f"got {self.waveform_generator.spin_conversion_phase}."
            )

        d = self.whitened_strains
        if phases is None:
            phases = self.phase_grid

        # Step 1: Compute signal for phase = 0, separated into the m-contributions from
        # the individual modes.
        pol_m = self.signal_m({**theta, "phase": 0})
        pol_m = {k: pol["waveform"] for k, pol in pol_m.items()}

        # Step 2: Precompute complex inner products (mu, mu) and (d, mu) for the
        # individual modes m.
        min_idx = self.data_domain.min_idx
        m_vals = sorted(pol_m.keys())

        # rho2opt is defined as the inner product of the waveform mu with itself.
        # Since we work with whitened data, the inner product simply reads
        #
        #   (mu, mu) = sum(mu.conj() * mu).real,
        #
        # where the sum extends over frequency bins and detectors. Applying phase
        # transformations exp(-i * m * phi) to the individual modes will lead to
        # cross terms (the ^-symbol indicates the phase shift by phi)
        #
        #   (mu^_m, mu^_n) = [(mu_m.conj() * mu_n) * exp(-i * (n - m) * phi)].real.
        #
        # Below we precompute the cross terms sum(mu_m.conj() * mu_n) and the constant
        # contribution for m = n.
        rho2opt_const = 0
        rho2opt_crossterms = {}
        for idx, m in enumerate(m_vals):
            mu_m = pol_m[m]
            # contribution to rho2opt_const
            rho2opt_const += sum(
                [inner_product(mu_ifo, mu_ifo, min_idx) for mu_ifo in mu_m.values()]
            )
            # cross terms
            for n in m_vals[idx + 1 :]:
                mu_n = pol_m[n]
                # factor 2, since (m, n) and (n, m) cross terms contribute symmetrically
                rho2opt_crossterms[(m, n)] = 2 * sum(
                    [
                        inner_product_complex(mu_m_ifo, mu_n_ifo, min_idx)
                        for mu_m_ifo, mu_n_ifo in zip(mu_m.values(), mu_n.values())
                    ]
                )
        # kappa2 is given by
        #
        #   (d, mu) = sum(d.conj() * mu).real.
        #
        # Applying phase transformations exp(-i * m * phi) to the individual modes
        # thus corresponds to
        #
        #   (d, mu^_m) = [(d.conj() * mu_m) * exp(-i * m * phi)].real.
        #
        # Below we precompute (d.conj() * mu_m) for the different modes m.
        kappa2_modes = {}
        for m in m_vals:
            mu_m = pol_m[m]
            kappa2_modes[m] = sum(
                [
                    inner_product_complex(d_ifo, mu_ifo, min_idx)
                    for d_ifo, mu_ifo in zip(d.values(), mu_m.values())
                ]
            )

        log_likelihoods = np.ones(len(phases))
        kappa2_all = []
        rho2opt_all = []
        for idx, phase in enumerate(phases):
            # get rho2opt
            rho2opt = rho2opt_const
            for (m, n), c in rho2opt_crossterms.items():
                rho2opt += (c * np.exp(-1j * (n - m) * phase)).real
            # get kappa2
            kappa2 = 0
            for m in m_vals:
                kappa2 += (kappa2_modes[m] * np.exp(-1j * m * phase)).real
            rho2opt_all.append(rho2opt)
            kappa2_all.append(kappa2)

            log_likelihoods[idx] = self.log_Zn + kappa2 - 1 / 2.0 * rho2opt

            # # comment out for cross check:
            # mu = sum_contributions_m(pol_m, phase_shift=phase)
            # rho2opt_ref = sum([inner_product(mu_ifo, mu_ifo) for mu_ifo in mu.values()])
            # kappa2_ref = sum(
            #     [
            #         inner_product(d_ifo, mu_ifo)
            #         for d_ifo, mu_ifo in zip(d.values(), mu.values())
            #     ]
            # )
            # assert rho2opt - rho2opt_ref < 1e-10
            # assert kappa2 - kappa2_ref < 1e-10

        # # Test that this works:
        # idx = len(phases) // 3
        # phase = phases[idx]
        # log_likelihood_ref = self.log_likelihood({**theta, "phase": phase})
        # print(log_likelihoods[idx] - log_likelihood_ref)

        return log_likelihoods

    def _log_likelihood_phase_marginalized(self, theta):
        """

        Parameters
        ----------
        theta: dict
            BBH parameters.

        Returns
        -------
        log_likelihood: float
        """

        # exp(2i * phase) approximation
        if self.pm_approx_22_mode:
            # Step 1: Compute whitened GW strain mu(theta) for parameters theta.
            # The phase parameter needs to be set to 0.
            theta["phase"] = 0.0
            mu = self.signal(theta)["waveform"]
            d = self.whitened_strains

            # Step 2: Compute likelihood. log_Zn is precomputed, so we only need to
            # compute the remaining terms rho2opt and kappa2
            rho2opt = sum([inner_product(mu_ifo, mu_ifo) for mu_ifo in mu.values()])
            # For the phase marginalized likelihood, we need to replace kappa2 with
            #
            #       log I0 ( |kappa2C| ),
            #
            # where kappa2C is the same as kappa2, but with the complex inner product
            # instead of the regular one, and I0 is the modified Bessel function of the
            # first kind of order 0.
            kappa2C = sum(
                [
                    inner_product_complex(d_ifo, mu_ifo)
                    for d_ifo, mu_ifo in zip(d.values(), mu.values())
                ]
            )
            return self.log_Zn + ln_i0(np.abs(kappa2C)) - 1 / 2.0 * rho2opt

        else:
            log_likelihoods_phase_grid = self.log_likelihood_phase_grid(theta)
            # Marginalize over phase. The phase-marginalized posterior is given by
            #
            #   p(theta^|d) = int d_phase p(d|theta^,phase) * p(theta^) * p(phase) / p(d),
            #
            #   where theta^ consists of all parameters except for the phase.
            #   To marginalize over the phase we evaluate it on a phase grid.
            #   With c = p(theta) / p(d) this yields
            #
            #   p(theta^|d) = c * int d_phase p(d|theta^,phase) * p(phase)
            #               = c * sum_i^N delta_phase_i p(theta^,i * delta_phase_i|d)
            #                   * p(i * delta_phase_i)
            #               = c * sum_i^N (2*pi/N) * p(theta^,i * delta_phase_i|d)
            #                   * (1/2*pi)
            #               = c * mean[p(theta^,i * delta_phase_i|d)].
            #
            # So to marginalize over the phase we simply need to take the mean of
            # np.exp(log_likelihoods_phase_grid). The constant c is accounted for
            # outside of this function.
            return logsumexp(log_likelihoods_phase_grid) - np.log(
                len(log_likelihoods_phase_grid)
            )

    def _log_likelihood_time_marginalized(self, theta):
        """
        Compute log likelihood with time marginalization.

        Parameters
        ----------
        theta

        Returns
        -------
        log_likelihood: float
        """
        # Step 1: Compute whitened GW strain mu(theta) for parameters theta.
        # The geocent_time parameter needs to be set to 0.
        theta["geocent_time"] = 0.0
        mu = self.signal(theta)["waveform"]
        # d = self.whitened_strains

        # Step 2: Compute likelihood. log_Zn is precomputed, so we only need to
        # compute the remaining terms rho2opt and kappa2.
        # rho2opt is time independent, and thus same as in the log_likelihood method.
        rho2opt = sum([inner_product(mu_ifo, mu_ifo) for mu_ifo in mu.values()])

        # kappa2 is time dependent. We use FFT to compute it for the discretized times
        # k * (delta_t/n_fft) and then sum over the time bins. The kappa2 contribution
        # is then given by
        #
        #       log sum_k exp(kappa2_k + log_prior_k),
        #
        # see Eq. (52) in https://arxiv.org/pdf/1809.02293.pdf. Here, kappa2_k is the
        # value of kappa2 and log_prior_k is the log_prior density at time
        # k * (delta_t/n_fft). The sum over k is the discretized integration of t.
        # Note: the time is discretized in two ways; for each FFT j (n_fft in total),
        # there are len(data_domain) time samples i, such that
        #
        #       t_ij = i * delta_t + j * (delta_t/n_fft).
        #
        # Summing over the time bins corresponds to a sum across both axes i and j.
        kappa2_ij = np.zeros((len(self.data_domain), self.n_fft))
        for j, dt in enumerate(self.t_FFT):
            # Get precomputed whitened strain, that is shifted by -dt.
            d = self.shifted_strains[dt]
            # Compute kappa2 contribution
            kappa2_ = [
                fft(d_ifo.conj() * mu_ifo).real
                for d_ifo, mu_ifo in zip(d.values(), mu.values())
            ]
            # sum contributions of different ifos
            kappa2_ij[:, j] = np.sum(kappa2_, axis=0)
        # Marginalize over time; this requires multiplying the likelihoods with the
        # prior (*not* in log space), summing over the time bins (both axes i and j!),
        # and then taking the log. See Eq. (52) in https://arxiv.org/pdf/1809.02293.pdf.
        # To prevent numerical issues, we use the logsumexp trick.
        assert kappa2_ij.shape == self.time_prior_log.shape
        exponent = kappa2_ij + self.time_prior_log
        alpha = np.max(exponent)
        kappa2 = alpha + np.log(np.sum(np.exp(exponent - alpha)))

        return self.log_Zn + kappa2 - 1 / 2.0 * rho2opt

    def _log_likelihood_calibration_marginalized(self, theta):
        """
        Computes log likelihood with calibration_marginalization

        Parameters
        ----------
        theta: dict
            BBH parameters.

        Returns
        -------
        log_likelihood: float
        """

        # Step 1: Compute whitened GW strain mu(theta) for parameters theta.
        mu = self.signal(theta)["waveform"]
        d = self.whitened_strains

        # In the calibration-marginalization case, the values of mu for each detector
        # have an additional leading dimension, which corresponds to the calibration
        # draws. These are averaged over in computing the likelihood.
        # TODO: Combine with _log_likelihood() into a single method.

        # Step 2: Compute likelihood. log_Zn is precomputed, so we only need to
        # compute the remaining terms rho2opt and kappa2

        rho2opt = np.sum(
            [inner_product(mu_ifo.T, mu_ifo.T) for mu_ifo in mu.values()], axis=0
        )
        kappa2 = np.sum(
            [
                inner_product(d_ifo[:, None], mu_ifo.T)
                for d_ifo, mu_ifo in zip(d.values(), mu.values())
            ],
            axis=0,
        )

        likelihoods = self.log_Zn + kappa2 - 1 / 2.0 * rho2opt
        # Return the average over calibration envelopes
        return logsumexp(likelihoods) - np.log(len(likelihoods))

    def d_inner_h_complex_multi(
        self, theta: pd.DataFrame, num_processes: int = 1
    ) -> np.ndarray:
        """
        Calculate the complex inner product (d | h(theta)) between the stored data d
        and a simulated waveform with given parameters theta. Works with multiprocessing.

        Parameters
        ----------
        theta : pd.DataFrame
            Parameters at which to evaluate h.
        num_processes : int
            Number of parallel processes to use.

        Returns
        -------
        complex : Inner product
        """
        with threadpool_limits(limits=1, user_api="blas"):
            # Generator object for theta rows. For idx this yields row idx of
            # theta dataframe, converted to dict, ready to be passed to
            # self.log_likelihood.
            theta_generator = (d[1].to_dict() for d in theta.iterrows())

            if num_processes > 1:
                with Pool(processes=num_processes) as pool:
                    d_inner_h_complex = pool.map(
                        self.d_inner_h_complex, theta_generator
                    )
            else:
                d_inner_h_complex = list(map(self.d_inner_h_complex, theta_generator))

        return np.array(d_inner_h_complex)

    def d_inner_h_complex(self, theta):
        """
        Calculate the complex inner product (d | h(theta)) between the stored data d
        and a simulated waveform with given parameters theta.

        Parameters
        ----------
        theta : dict
            Parameters at which to evaluate h.

        Returns
        -------
        complex : Inner product
        """
        # TODO: Implement for time marginalization.
        return self._d_inner_h_complex(theta)

    def _d_inner_h_complex(self, theta):
        mu = self.signal(theta)["waveform"]
        d = self.whitened_strains
        return sum(
            [
                inner_product_complex(d_ifo, mu_ifo)
                for d_ifo, mu_ifo in zip(d.values(), mu.values())
            ]
        )


def inner_product(a, b, min_idx=0, delta_f=None, psd=None):
    """
    Compute the inner product between two complex arrays. There are two modes: either,
    the data a and b are not whitened, in which case delta_f and the psd must be
    provided. Alternatively, if delta_f and psd are not provided, the data a and b are
    assumed to be whitened already (i.e., whitened as d -> d * sqrt(4 delta_f / psd)).

    Note: sum is only taken along axis 0 (which is assumed to be the frequency axis),
    while other axes are preserved. This is e.g. useful when evaluating kappa2 on a
    phase grid.

    Parameters
    ----------
    a: np.ndaarray
        First array with frequency domain data.
    b: np.ndaarray
        Second array with frequency domain data.
    min_idx: int = 0
        Truncation of likelihood integral, index of lowest frequency bin to consider.
    delta_f: float
        Frequency resolution of the data. If None, a and b are assumed to be whitened
        and the inner product is computed without further whitening.
    psd: np.ndarray = None
        PSD of the data. If None, a and b are assumed to be whitened and the inner
        product is computed without further whitening.

    Returns
    -------
    inner_product: float
    """
    #
    if psd is not None:
        if delta_f is None:
            raise ValueError(
                "If unwhitened data is provided, both delta_f and psd must be provided."
            )
        return 4 * delta_f * np.sum((a.conj() * b / psd)[min_idx:], axis=0).real
    else:
        return np.sum((a.conj() * b)[min_idx:], axis=0).real


def inner_product_complex(a, b, min_idx=0, delta_f=None, psd=None):
    """
    Same as inner product, but without taking the real part. Retaining phase
    information is useful for the phase-marginalized likelihood. For further
    documentation see inner_product function.
    """
    #
    if psd is not None:
        if delta_f is None:
            raise ValueError(
                "If unwhitened data is provided, both delta_f and psd must be provided."
            )
        return 4 * delta_f * np.sum((a.conj() * b / psd)[min_idx:], axis=0)
    else:
        return np.sum((a.conj() * b)[min_idx:], axis=0)


def build_stationary_gaussian_likelihood(
    metadata,
    event_dataset=None,
    time_marginalization_kwargs=None,
):
    """
    Build a StationaryGaussianLikelihoodBBH object from the metadata.

    Parameters
    ----------
    metadata: dict
        Metadata from stored dingo parameter samples file.
        Typially accessed via pd.read_pickle(/path/to/dingo-output.pkl).metadata.
    event_dataset: str = None
        Path to event dataset for caching. If None, don't cache.
    time_marginalization_kwargs: dict = None
        Forwarded to the likelihood.

    Returns
    -------
    likelihood: StationaryGaussianGWLikelihood
        likelihood object
    """
    # get strain data
    event_data, data_domain = get_event_data_and_domain(
        metadata["model"], event_dataset=event_dataset, **metadata["event"]
    )

    # set up likelihood
    likelihood = StationaryGaussianGWLikelihood(
        wfg_kwargs=metadata["model"]["dataset_settings"]["waveform_generator"],
        wfg_domain=build_domain(metadata["model"]["dataset_settings"]["domain"]),
        data_domain=data_domain,
        event_data=event_data,
        t_ref=metadata["event"]["time_event"],
        time_marginalization_kwargs=time_marginalization_kwargs,
    )

    return likelihood


def get_wfg(wfg_kwargs, data_domain, frequency_range=None):
    """
    Set up waveform generator from wfg_kwargs. The domain of the wfg is primarily
    determined by the data domain, but a new (larger) frequency range can be
    specified if this is necessary for the waveforms to be generated successfully
    (e.g., for EOB waveforms which require a sufficiently small f_min and sufficiently
    large f_max).

    Parameters
    ----------
    wfg_kwargs: dict
        Waveform generator parameters.
    data_domain: dingo.gw.domains.Domain
        Domain of event data, with bounds determined by likelihood integral.
    frequency_range: dict = None
        Frequency range for waveform generator. If None, that of data domain is used,
        which corresponds to the bounds of the likelihood integral.
        Possible keys:
            'f_start': float
                Frequency at which to start the waveform generation. Overrides f_start in
                metadata["model"]["dataset_settings"]["waveform_generator"].
            'f_end': float
                Frequency at which to start the waveform generation.

    Returns
    -------
    wfg: dingo.gw.waveform_generator.WaveformGenerator
        Waveform generator object.

    """
    if frequency_range is None:
        return WaveformGenerator(domain=data_domain, **wfg_kwargs)

    else:
        if "f_start" in frequency_range and frequency_range["f_start"] is not None:
            if frequency_range["f_start"] > data_domain.f_min:
                raise ValueError("f_start must be less than f_min.")
            wfg_kwargs["f_start"] = frequency_range["f_start"]
        if "f_end" in frequency_range and frequency_range["f_end"] is not None:
            if frequency_range["f_end"] < data_domain.f_max:
                raise ValueError("f_end must be greater than f_max.")
            # get wfg domain, but care to not modify the original data_domain
            data_domain = build_domain(
                {**data_domain.domain_dict, "f_max": frequency_range["f_end"]}
            )
        return WaveformGenerator(domain=data_domain, **wfg_kwargs)


def main():
    import pandas as pd

    samples = pd.read_pickle(
        "/Users/maxdax/Documents/Projects/GW-Inference/dingo/datasets/dingo_samples"
        "/02_XPHM/dingo_samples_GW150914.pkl"
    )
    event_dataset = (
        "/Users/maxdax/Documents/Projects/GW-Inference/dingo/dingo-devel"
        "/tutorials/02_gwpe/datasets/strain_data/events_dataset.hdf5"
    )

    likelihood = build_stationary_gaussian_likelihood(samples.attrs, event_dataset)

    from tqdm import tqdm

    log_likelihoods = []
    for idx in tqdm(range(1000)):
        theta = dict(samples.iloc[idx])
        try:
            l = likelihood.log_prob(theta)
        except:
            print(idx)
            l = float("nan")
        log_likelihoods.append(l)
    log_likelihoods = np.array(log_likelihoods)
    log_likelihoods = log_likelihoods[~np.isnan(log_likelihoods)]
    print(f"mean: {np.mean(log_likelihoods)}")
    print(f"std: {np.std(log_likelihoods)}")
    print(f"max: {np.max(log_likelihoods)}")
    print(f"min: {np.min(log_likelihoods)}")


if __name__ == "__main__":
    main()
