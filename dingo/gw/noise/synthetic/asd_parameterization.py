from functools import partial

from multiprocessing import Pool
import numpy as np
import scipy
import tqdm
import scipy.optimize
from threadpoolctl import threadpool_limits

from dingo.gw.noise.synthetic.utils import get_index_for_elem, lorentzian_eval

P0_A, P0_Q = 5, 100
MIN_A, MIN_Q = 0, 10
MAX_A, MAX_Q = 12, 1000
MAXFEV = 5000000


def parameterize_asd_dataset(real_dataset, parameterization_settings, num_processes, verbose):
    """
    Parameterize a dataset of ASDs using a spline fit to the broadband noise and Lorentzians for the spectral features.

    Parameters
    ----------
    real_dataset : ASDDataset
        Dataset containing the ASDs to be parameterized.
    parameterization_settings : dict
        Dictionary containing the settings for the parameterization.
    num_processes : int
        Number of processes to use for parallelization.
    verbose : bool
        If True, print progress bars.
    """
    real_asd_dict = real_dataset.asds
    domain = real_dataset.domain

    parameters_dict = {det: [] for det in real_asd_dict.keys()}
    for det, asds in real_asd_dict.items():
        if num_processes > 1:
            with threadpool_limits(limits=1, user_api="blas"):
                with Pool(processes=num_processes) as pool:
                    parameters_dict[det] = parameterize_asds_parallel(
                        asds, domain, parameterization_settings, pool=pool, verbose=verbose
                    )
        else:
            parameters_dict[det] = parameterize_asds_parallel(asds, domain, parameterization_settings)

    return parameters_dict


def parameterize_asds_parallel(asds, domain, parameterization_settings, pool=None, verbose=False):
    """
    Helper function to be called for parallel ASD parameterization.

    Parameters
    ----------
    asds : array_like
        Array containing the ASDs to be parameterized.
    domain : Domain
        Domain object containing the frequency grid.
    parameterization_settings : dict
        Dictionary containing the settings for the parameterization.
    pool : Pool, optional
        Pool object for parallelization. If None, the function is not parallelized.
    verbose : bool
        If True, print progress bars.

    """

    task_func = partial(parameterize_single_psd, domain=domain, parameterization_settings=parameterization_settings)
    psds = asds ** 2
    if pool is not None:
        parameters_list = list(tqdm.tqdm(pool.imap(task_func, psds), total=psds.shape[0], disable=not verbose))
    else:
        parameters_list = list(tqdm.tqdm(map(task_func, psds), total=psds.shape[0], disable=not verbose))

    parameters = {
        feature: np.stack([asd_param[feature] for asd_param in parameters_list])
        for feature in parameters_list[0].keys()
    }
    # the x-positions are the same for all asds
    parameters["x_positions"] = parameters["x_positions"][0]

    return parameters


def parameterize_single_psd(real_psd, domain, parameterization_settings):
    """
    Parameterize a single ASD using a spline fit to the broadband noise and Lorentzians for the spectral features.

    Parameters
    ----------
    real_psd : array_like
        PSD to be parameterized.
    domain : Domain
        Domain object containing the frequency grid.
    parameterization_settings : dict
        Dictionary containing the settings for the parameterization.

    """
    sigma = float(parameterization_settings["sigma"])
    if not len(real_psd) == len(domain.sample_frequencies):
        real_psd = domain.update_data(real_psd)

    # optional truncation parameter for Lorentzians. Set to None if non-positive value is passed
    delta_f = parameterization_settings.get("delta_f", -1)
    delta_f = float(delta_f) if delta_f > 0 else None

    # transform psd to log space
    transformed_psd = np.log(real_psd)

    # parameterize broad-band noise
    xs, ys = fit_broadband_noise(
        domain=domain,
        psd=transformed_psd,
        num_spline_positions=parameterization_settings["num_spline_positions"],
        sigma=sigma,
    )
    spline = scipy.interpolate.CubicSpline(xs, ys)
    broadband_noise = spline(domain.sample_frequencies)

    _, features = fit_spectral(
        frequencies=domain.sample_frequencies,
        psd=transformed_psd,
        broadband_noise=broadband_noise,
        num_spectral_segments=parameterization_settings["num_spectral_segments"],
        sigma=sigma,
        delta_f=delta_f,
    )
    # features = np.random.normal(100, 10, size=(400, 3))
    parameter_dict = {"x_positions": xs, "y_values": ys, "spectral_features": features}

    return parameter_dict


def fit_broadband_noise(domain, psd, num_spline_positions, sigma, f_min=20):
    """
    Fit a spline to the broadband noise of a PSD.

    Parameters
    ----------
    domain : Domain
        Domain object containing the frequency grid.
    psd : array_like
        PSD to be parameterized.
    num_spline_positions : int
        Number of spline positions.
    sigma : float
        Standard deviation of the Gaussian noise used for the spline fit.
    f_min : float, optional
        position of the first node for the spline fi

    """
    frequencies = domain.sample_frequencies

    # standardize frequencies to the interval [0,1]
    standardized_frequencies = (frequencies - frequencies[0]) / (
        frequencies[-1] - frequencies[0]
    )

    # log-distributed x-positions in [20/f_max, 1] for the interpolating base noise spline
    log_xs = np.logspace(np.log10(f_min / domain.f_max), 0, num_spline_positions)

    # get indices corresponding to log_xs

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
            ind_min, ind_max = int((ind + xs_indices[i - 1]) / 2), ind + 1
        else:
            ind_min, ind_max = int((ind + xs_indices[i - 1]) / 2), int(
                (ind + xs_indices[i + 1]) / 2
            )

        # Apply filter to remove outliers, i.e. spectral lines
        assert ind_min != ind_max
        data = psd[ind_min:ind_max]
        mov_median = np.median(psd[ind_min_old:ind_max])

        # Any samples more than 3 noise stds away are removed as outliers
        ind_red = np.where(data - mov_median < 3 * sigma)
        data_red = data[ind_red]

        # Use mean over cleaned data as an estimate for the spline value (mean of the Gaussian noise)
        ys.append(np.mean(data_red))

        ind_min_old = ind_min

    return xs, np.array(ys)


def fit_spectral(
    frequencies, psd, broadband_noise, num_spectral_segments, sigma, delta_f
):
    """
    Fit Lorentzians to the spectral features of a PSD.

    Parameters
    ----------
    frequencies : array_like
        Frequency grid.
    psd : array_like
        PSD to be parameterized.
    broadband_noise : array_like
        Broadband noise of the PSD.
    num_spectral_segments : int
        Number of spectral segments.
    sigma : float
        Standard deviation of the Gaussian noise used for the spline fit.
    delta_f : float
        Truncation parameter for Lorentzians. Set to None if non-positive value is passed.

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

        data = {
            "psd": psd_data,
            "broadband_noise": broadband_noise_data,
            "frequencies": frequency_data,
            "lower_freq": frequency_data[0],
            "upper_freq": frequency_data[-1],
        }

        with threadpool_limits(limits=1, user_api="blas"):
            try:
                popt = curve_fit(data, sigma, delta_f)
            except RuntimeError:
                popt = [0.0, 0.0, 0.0]  # will be replaced by sampled version below

        f0, A, Q = popt

        lorentzian = lorentzian_eval(frequency_data, f0, A, Q, delta_f=delta_f)
        # if no spectral line has been found -> peak smaller than 3 stds.
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


def curve_fit(data, std, delta_f=None):
    """
    Fit a Lorentzian to the PSD.

    Parameters
    ----------
    data : dict
        Dictionary containing the PSD, broadband noise, and frequency grid.
    std : float
        Standard deviation of the Gaussian noise.
    delta_f : float
        Truncation parameter for Lorentzians. Set to None if non-positive value is passed.

    """
    func = partial(lorentzian_eval, delta_f=delta_f)

    popt, pcov = scipy.optimize.curve_fit(
        func,
        data["frequencies"],
        data["psd"] - data["broadband_noise"],
        p0=[(data["lower_freq"] + data["upper_freq"]) / 2, P0_A, P0_Q],
        sigma=[std] * len(data["frequencies"]),
        bounds=[[data["lower_freq"], MIN_A, MIN_Q], [data["upper_freq"], MAX_A, MAX_Q]],
        maxfev=MAXFEV,
    )
    return popt
