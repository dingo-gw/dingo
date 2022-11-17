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


def lorentzian_eval(x, f0, A, Q):
    if f0 == 0 or A < 0:
        return np.zeros_like(x)
    delta_f = (x[-1] - x[0]) / 4
    truncate = np.where(np.abs(x - f0) <= delta_f, 1, np.exp(-np.abs(x - f0) / delta_f))
    # equivalent to a large delta_f. This helps fit adjacent spectral features without a gap in between
    # truncate = 1
    return truncate * A * (f0**4) / ((x * f0) ** 2 + Q**2 * (f0**2 - x**2) ** 2)


def apply_parameterization(
        data_dir,
        settings,
        detector,
        num_processes=1,
        verbose=True,
        job_index=None,
        num_jobs=None,
):
    f_min = 0
    f_max = settings["dataset_settings"].get("f_max", (settings["dataset_settings"]["f_s"] / 2))
    T_PSD = settings["dataset_settings"]["T_PSD"]
    T_gap = settings["dataset_settings"]["T_gap"]
    T = settings["dataset_settings"]["T"]

    delta_f = 1 / T
    domain = build_domain(
        {
            "type": "FrequencyDomain",
            "f_min": f_min,
            "f_max": f_max,
            "delta_f": delta_f,
            "window_factor": None,
        }
    )

    frequencies = domain.sample_frequencies
    psd_path = get_path_raw_data(
        data_dir, settings["dataset_settings"]["observing_run"], detector, T_PSD, T_gap
    )

    num_psds = len(
        [
            name
            for name in os.listdir(psd_path)
            if os.path.isfile(os.path.join(psd_path, name)) and name.startswith("psd")
        ]
    )
    os.makedirs(psd_path, exist_ok=True)

    task_func = partial(
        parameterize_func,
        psd_path=psd_path,
        domain=domain,
        settings=settings["parameterization_settings"],
    )
    print(f"parameterizing {num_psds} psds")
    if num_jobs is not None and job_index is not None:
        job_segment_len = int(num_psds / num_jobs)
        indices = range(job_index * job_segment_len, (job_index + 1) * job_segment_len)
    else:
        indices = range(num_psds)

    if num_processes > 1:
        with Pool(processes=num_processes) as pool:
            with tqdm(total=len(indices), disable=not verbose) as pbar:
                for _, i in enumerate(pool.imap_unordered(task_func, indices)):
                    pbar.update()

    else:
        with tqdm(total=len(indices), disable=not verbose) as pbar:
            for _, i in enumerate(map(task_func, indices)):
                pbar.update()


# TODO: load and save to the same file. Add key to dictionary
# def parameterize_func(index, psd_path, domain, settings):
#     raw_filename = join(psd_path, f"psd_{index:05d}.npy")
#     p_filename = join(psd_path, f"p_psd_{index:05d}.npy")
#
#     if not exists(p_filename):
#
#         try:
#             psd_data = np.load(raw_filename, allow_pickle=True).item()
#         except FileNotFoundError:
#             print(f"file {raw_filename} not found")
#             return
#         parameter_dict, p_psd = parameterize_single_psd(
#             psd_data["psd"], domain, settings
#         )
#
#         np.save(
#             p_filename,
#             {
#                 "detector": psd_data["detector"],
#                 "segment": psd_data["segment"],
#                 "time": psd_data["time"],
#                 "psd": p_psd,
#                 "tukey_window": psd_data["tukey_window"],
#                 "parameters": parameter_dict,
#             },
#         )


def parameterize_single_psd(real_psd, domain, parameterization_settings):
    scale_factor = float(parameterization_settings["scale_factor"])
    # make PSD compatible with domain
    real_psd = domain.update_data(real_psd)

    transformed_psd = np.log(real_psd * scale_factor)

    xs, ys = fit_base_noise(
        domain=domain,
        psd=transformed_psd,
        num_spline_positions=parameterization_settings["num_spline_positions"],
    )
    spline = scipy.interpolate.CubicSpline(xs, ys)
    base_noise = spline(domain.sample_frequencies)

    spectral_segments = np.array_split(
        np.arange(transformed_psd.shape[0]),
        parameterization_settings["num_spectral_segments"],
    )

    lorentzians, features, variance = fit_spectral(
        frequencies=domain.sample_frequencies,
        psd=transformed_psd,
        base_noise=base_noise,
        settings=parameterization_settings,
        spectral_segments=spectral_segments,
    )

    p_psd = np.random.normal(base_noise + lorentzians, variance)
    p_psd = np.exp(p_psd) / scale_factor
    parameter_dict = {
        "x_positions": xs,
        "y_values": ys,
        "spectral_features": features,
        "variance": variance,
    }
    return parameter_dict, p_psd


def fit_base_noise(domain, psd, num_spline_positions):
    frequencies = domain.sample_frequencies

    # helper function to get index of closest element of an array
    def get_index_for_elem(arr, elem):
        return (np.abs(arr - elem)).argmin()

    def trafo(x):
        return (1 - 0) / (frequencies[-1] - frequencies[0]) * (x - frequencies[0]) + 0

    # log the frequencies and set xs to uniformly distirbuted in log-space. Stadardization used to always produces the
    # same spacing between x positions independent of the frequency range of interest
    standardized_frequencies = trafo(frequencies)
    # print(normalized_frequencies)
    # log_frequencies = np.log10(normalized_frequencies)

    # log-distributed x-positions for the interpolating base noise spline
    log_xs = np.logspace(np.log10(20 / domain.f_max), 0, num_spline_positions)

    xs_indices = np.array(
        [get_index_for_elem(standardized_frequencies, log_x) for log_x in log_xs]
    )
    xs = frequencies[xs_indices]
    ys = []

    # for the y-values take the mean over an e-interval around the given x-value. Furthermore, we remove outliers
    # (further than 2 stds away from median) before computing the means such that spectral features are not considered
    # for the interpolation
    # print(xs)
    # print(xs_indices)
    ind_min_old = None
    mov_medians = []
    data_reds = []
    freqs_reds = []
    for i, ind in enumerate(xs_indices):

        if i == 0:
            ind_min, ind_max = ind, int((ind + xs_indices[i + 1]) / 2)
        elif i == len(xs_indices) - 1:
            ind_min, ind_max = int((ind + xs_indices[i - 1]) / 2), ind
        else:
            ind_min, ind_max = int((ind + xs_indices[i - 1]) / 2), int(
                (ind + xs_indices[i + 1]) / 2
            )

        if ind_min == ind_max:
            ys.append(psd[ind_min])
            # data_red = psd[ind_min:ind_max]
            # data_reds = np.concatenate((data_reds, data_red))

        else:
            data = psd[ind_min:ind_max]
            # compute moving median of the neighborhoods of the spline positions for robust outlier removal
            mov_median = np.median(psd[ind_min_old:ind_max])
            # mov_median = np.median(psd[ind_min:ind_max])
            mov_medians.append(mov_median)

            # Any samples more than 3 noise stds away are removed as outliers
            ind_red = np.where(data - mov_median < 3 * 0.14)
            data_red = data[ind_red]
            data_reds = np.concatenate((data_reds, data_red))
            # for plotting only
            freqs_red = frequencies[ind_min:ind_max]
            freqs_red = freqs_red[ind_red]
            freqs_reds = np.concatenate((freqs_reds, freqs_red))

            # Use empirical mean over cleaned data as an estimate for the spline value (mean of the Gaussian noise)
            ys.append(np.mean(data_red))

        ind_min_old = ind_min

    # print(xs)
    # spline = scipy.interpolate.CubicSpline(xs, ys)
    # base_noise = spline(frequencies)

    # fig = plt.figure(figsize=(16,9))
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.plot(frequencies, psd, label="true psd")
    # plt.plot(freqs_reds, data_reds, label="outlier removed data")
    # plt.plot(frequencies, base_noise, label="computed base psd")
    #
    # plt.scatter(xs[1:-2], mov_medians[1:-2], c='red', zorder=10, label="moving median")
    # plt.xlim(20, 2048)
    # plt.legend()
    # plt.savefig("base_noise_estimation.png", dpi=600)
    # plt.show()
    # exit(1)

    return xs, np.array(ys)


def fit_spectral(frequencies, psd, base_noise, settings, spectral_segments):
    """
    Parameters
    ----------
    psd: PSD of which to fit the spectral features
    base_noise: Base noise of the PSD, i.e. non-spectral features
    spectral_indices: Provide indices where spectral features have been detected to skip those segments where
        there are none
    Returns
    -------
    """
    # could also pass these so they do not have to be re-computed

    spectral_features = np.array([[0.0, 0.0, 0.0]] * len(spectral_segments))
    lorentzians = np.array([])

    for i, segment in enumerate(spectral_segments):

        psd_data = psd[segment]
        frequency_data = frequencies[segment]
        base_noise_data = base_noise[segment]

        data = {
            "psd": psd_data,
            "base_noise": base_noise_data,
            "frequencies": frequency_data,
            "lower_freq": frequency_data[0],
            "upper_freq": frequency_data[-1],
        }

        sigma = 0.14  # std in psd data obtained as a MAP estimate in a spectral-free segment

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

    return lorentzians, spectral_features, sigma


def curve_fit(data, std):
    popt, pcov = scipy.optimize.curve_fit(
        lorentzian_eval,
        data["frequencies"],
        data["psd"] - data["base_noise"],
        p0=[(data["lower_freq"] + data["upper_freq"]) / 2, 5, 500],
        sigma=[std] * len(data["frequencies"]),
        bounds=[[data["lower_freq"], 0, 50], [data["upper_freq"], 8, 500000]],
        maxfev=5000000,
    )
    return popt
