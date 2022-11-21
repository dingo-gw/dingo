import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import scipy
import scipy.optimize
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from dingo.gw.domains import build_domain
from dingo.gw.noise_dataset.utils import get_path_raw_data


def lorentzian_eval(x, f0, A, Q, delta_f=100):

    if f0 == 0 or A < 0:
        return np.zeros_like(x)

    # used to truncate tails of Lorentzian if necessary. Will have no effect, if delta_f sufficiently large
    truncate = np.where(np.abs(x - f0) <= delta_f, 1, np.exp(-np.abs(x - f0) / delta_f))

    return truncate * A * (f0**4) / ((x * f0) ** 2 + Q**2 * (f0**2 - x**2) ** 2)


def parameterize_single_psd(real_psd, domain, parameterization_settings):

    sigma = float(parameterization_settings["sigma"])

    # optional truncation parameter for Lorentzians. Set to None if non-positive value is passed
    delta_f = parameterization_settings.get("delta_f", None)
    delta_f = float(delta_f) if delta_f > 0 else None

    # transform psd to log space and scale to zero mean for numerical stability
    transformed_psd = np.log(real_psd)
    scale_factor = np.mean(transformed_psd)
    transformed_psd -= scale_factor

    # parameterize broad-band noise
    xs, ys = fit_broadband_noise(
        domain=domain,
        psd=transformed_psd,
        num_spline_positions=parameterization_settings["num_spline_positions"],
        sigma=sigma
    )
    spline = scipy.interpolate.CubicSpline(xs, ys)
    broadband_noise = spline(domain.sample_frequencies)

    lorentzians, features = fit_spectral(
        frequencies=domain.sample_frequencies,
        psd=transformed_psd,
        broadband_noise=broadband_noise,
        num_spectral_segments=parameterization_settings["num_spectral_segments"],
        sigma=sigma
    )

    p_psd = np.random.normal(broadband_noise + lorentzians, sigma)
    p_psd = np.exp(p_psd + scale_factor)
    parameter_dict = {
        "x_positions": xs,
        "y_values": ys,
        "spectral_features": features
    }
    return parameter_dict, p_psd


def fit_broadband_noise(domain, psd, num_spline_positions, sigma):
    frequencies = domain.sample_frequencies

    # standardize frequencies to the interval [0,1]
    standardized_frequencies = (frequencies - frequencies[0]) / (frequencies[-1] - frequencies[0])

    # log-distributed x-positions in [20/f_max, 1] for the interpolating base noise spline
    # TODO: minimum frequency has to be positive. We usually use 20, but we might also want to go lower at some point?
    log_xs = np.logspace(np.log10(20 / domain.f_max), 0, num_spline_positions)

    # get indices corresponding to log_xs
    def get_index_for_elem(arr, elem):
        return (np.abs(arr - elem)).argmin()
    xs_indices = np.array(
        [get_index_for_elem(standardized_frequencies, log_x) for log_x in log_xs]
    )
    xs = frequencies[xs_indices]
    ys = []

    # for the y-values take the mean over an e-interval around the given x-value. Furthermore, we remove outliers
    # (further than 3 stds away from median) before computing the means such that spectral features are not considered
    # for the interpolation
    ind_min_old = 0
    for i, ind in enumerate(xs_indices):

        if i == 0:
            ind_min, ind_max = ind, int((ind + xs_indices[i + 1]) / 2)
        elif i == len(xs_indices) - 1:
            ind_min, ind_max = int((ind + xs_indices[i - 1]) / 2), ind
        else:
            ind_min, ind_max = int((ind + xs_indices[i - 1]) / 2), int(
                (ind + xs_indices[i + 1]) / 2
            )

        # Apply filter to remove outliers, i.e. spectral lines
        assert(ind_min != ind_max)
        data = psd[ind_min:ind_max]
        mov_median = np.median(psd[ind_min_old:ind_max])

        # Any samples more than 3 noise stds away are removed as outliers
        ind_red = np.where(data - mov_median < 3 * sigma)
        data_red = data[ind_red]

        # Use empirical mean over cleaned data as an estimate for the spline value (mean of the Gaussian noise)
        ys.append(np.mean(data_red))

        ind_min_old = ind_min

    return xs, np.array(ys)


def fit_spectral(frequencies, psd, broadband_noise, num_spectral_segments, sigma):
    """
    Parameters
    ----------
    num_spectral_segments
    frequencies
    psd: PSD of which to fit the spectral features
    broadband_noise: Base noise of the PSD, i.e. non-spectral features
     num_spectral_segments
    sigma
    Returns
    -------
    """
    
    # divide frequency spectrum into equi-length sub-intervals, in each of which a single spectral line is fitted
    spectral_segments = np.array_split(
        np.arange(psd.shape[0]),
        num_spectral_segments,
    )

    spectral_features = np.array([[0.0, 0.0, 0.0]] * num_spectral_segments)
    lorentzians = np.array([])

    for i, segment in enumerate(spectral_segments):

        psd_data = psd[segment]
        frequency_data = frequencies[segment]
        broadband_noise_data = broadband_noise[segment]

        # TODO: add delta_f
        data = {
            "psd": psd_data,
            "broadband_noise": broadband_noise_data,
            "frequencies": frequency_data,
            "lower_freq": frequency_data[0],
            "upper_freq": frequency_data[-1],
        }

        with threadpool_limits(limits=1, user_api="blas"):

            try:
                popt = curve_fit(data, sigma)
            except RuntimeError:
                popt = [0.0, 0.0, 0.0]  # will be replaced by sampled version below

        f0, A, Q = popt

        lorentzian = lorentzian_eval(frequency_data, f0, A, Q)
        # if no spectral lines has been found -> peak smaller than 3 stds.
        # We don't want to fit random fluctuations as spectral lines
        if np.max(lorentzian) <= 3 * sigma:
            # sample from fixed distribution to group the event "no spectral line" together
            f0 = np.random.normal(frequency_data[0], 1.0e-3)
            A = np.random.normal(-1, 1.0e-8)
            Q = np.random.normal(500, 1)
            lorentzian = np.zeros_like(frequency_data)

        spectral_features[i] = [f0, A, Q]
        lorentzians = np.concatenate((lorentzians, lorentzian), axis=0)

    return lorentzians, spectral_features


# TODO: should the parameters here also be optionally passed via settings file?
def curve_fit(data, std):
    popt, pcov = scipy.optimize.curve_fit(
        lorentzian_eval,
        data["frequencies"],
        data["psd"] - data["broadband_noise"],
        p0=[(data["lower_freq"] + data["upper_freq"]) / 2, 5, 100],
        sigma=[std] * len(data["frequencies"]),
        bounds=[[data["lower_freq"], 0, 10], [data["upper_freq"], 12, 1000]],
        maxfev=5000000,
    )
    return popt
