import os
import numpy as np
import requests
from typing import Dict, List
from functools import partial
from tqdm import tqdm
import random
from multiprocessing import Pool
from dingo.gw.ASD_dataset.noise_dataset import ASDDataset
from dingo.gw.domains import build_domain
import h5py
import warnings
from os.path import join
from io import StringIO
from tqdm import trange

from dingo.core.dataset import recursive_hdf5_save
from dingo.gw.gwutils import get_window
from dingo.gw.download_strain_data import download_psd

"""
Contains links for PSD segment lists with quality label BURST_CAT2 from the Gravitational Wave Open Science Center.
Some events are split up into multiple chunks such that there are multiple URLs for one observing run
"""
URL_DIRECTORY = {
    "O1_L1": [
        "https://www.gw-openscience.org/timeline/segments/O1/L1_BURST_CAT2/1126051217/11203200/"
    ],
    "O1_H1": [
        "https://www.gw-openscience.org/timeline/segments/O1/H1_BURST_CAT2/1126051217/11203200/"
    ],
    "O2_L1": [
        "https://www.gw-openscience.org/timeline/segments/O2_16KHZ_R1/L1_BURST_CAT2/1164556817/23176801/"
    ],
    "O2_H1": [
        "https://www.gw-openscience.org/timeline/segments/O2_16KHZ_R1/H1_BURST_CAT2/1164556817/23176801/"
    ],
    "O2_V1": [
        "https://www.gw-openscience.org/timeline/segments/O2_16KHZ_R1/V1_BURST_CAT2/1164556817/23176801/"
    ],
    "O3_L1": [
        "https://www.gw-openscience.org/timeline/segments/O3a_16KHZ_R1/L1_BURST_CAT2/1238166018/15811200/",
        "https://www.gw-openscience.org/timeline/segments/O3b_16KHZ_R1/L1_BURST_CAT2/1256655618/12708000/",
    ],
    "O3_H1": [
        "https://www.gw-openscience.org/timeline/segments/O3a_16KHZ_R1/H1_BURST_CAT2/1238166018/15811200/",
        "https://www.gw-openscience.org/timeline/segments/O3b_16KHZ_R1/H1_BURST_CAT2/1256655618/12708000/",
    ],
    "O3_V1": [
        "https://www.gw-openscience.org/timeline/segments/O3a_16KHZ_R1/V1_BURST_CAT2/1238166018/15811200/",
        "https://www.gw-openscience.org/timeline/segments/O3b_16KHZ_R1/V1_BURST_CAT2/1256655618/12708000/",
    ],
}


def get_time_segments(data_dir, settings):
    """
    Given the segments `segs` and the time constraints `T_PSD`, `delta_T`, return all segments
    that can be used to estimate a PSD

    Parameters
    ----------
    settings : dict
        Contains all settings necessary

    Returns
    -------
    All segments that can be used to estimate a PSD
    """

    time_segments = dict(zip(settings["detectors"], [[] * len(settings["detectors"])]))

    run = settings["observing_run"]
    T_PSD = settings["T_PSD"]
    T_gap = settings["T_gap"]
    T = settings["T"]
    f_s = settings["f_s"]
    num_psds_max = settings.get("num_psds_max")

    for detector in settings["detectors"]:

        path_raw_psds = get_path_raw_data(data_dir, run, detector)
        os.makedirs(path_raw_psds, exist_ok=True)

        key = run + "_" + detector
        urls = URL_DIRECTORY[key]

        starts, stops, durations = [], [], []
        for url in urls:
            r = requests.get(url, allow_redirects=True)
            c = StringIO(r.content.decode("utf-8"))
            starts_seg, stops_seg, durations_seg = np.loadtxt(
                c, dtype="int", unpack=True
            )
            starts = np.hstack([starts, starts_seg])
            stops = np.hstack([stops, stops_seg])
            durations = np.hstack([durations, durations_seg])

        segs = np.array(list(zip(starts, stops, durations)), dtype=int)
        segs = segs[segs[:, 2] >= T_PSD]

        valid_segments = []
        for idx in range(segs.shape[0]):
            seg = segs[idx, :]
            start_time = seg[0]
            end_time = start_time + T_PSD

            while end_time in range(seg[0], seg[1] + 1):
                valid_segments.append((start_time, end_time))
                start_time = end_time + T_gap
                end_time = start_time + T_PSD

        # randomly sample a subset of time segments to estimate PSDs
        if num_psds_max is not None and num_psds_max > 0:
            valid_segments = valid_segments[:num_psds_max] #random.sample(valid_segments, num_psds_max)

        time_segments[detector] = valid_segments

    return time_segments


def get_path_raw_data(data_dir, run, detector):
    """
    Return the directory where the PSD data is to be stored
    Parameters
    ----------
    data_dir : str
        Path to the directory where the PSD dataset will be stored
    run : str
        Observing run that is used for the PSD dataset generation
    detector : str
        Detector that is used for the PSD dataset generation

    Returns
    -------
    the path where the data is stored
    """
    return join(data_dir, "tmp", run, detector)


def estimate_func(seg, run, domain, estimation_kwargs, psd_path, settings):

    dataset_dict = {
        "settings": {
            "dataset_settings": settings["dataset_settings"],
            "domain_dict": domain.domain_dict,
        }
    }

    start, end = seg[0], seg[1]
    filename = join(psd_path, f"asd_{start}.hdf5")

    parameterize = settings.get("parameterization_settings", False)
    # TODO: more elegant way to do this?
    if not os.path.exists(filename) or parameterize:

        if not os.path.exists(filename):
            psd = download_psd(
                time_start=start,
                **estimation_kwargs
            )

        # otherwise parameterization settings are passed
        else:
            pass

        asd = np.sqrt(psd[domain.min_idx: domain.max_idx + 1])
        gps_time = start
        dataset_dict["asds"] = {estimation_kwargs["det"]: np.array([asd])}
        dataset_dict["gps_times"] = {estimation_kwargs["det"]: np.array([gps_time])}

        dataset = ASDDataset(dictionary=dataset_dict)
        dataset.to_file(file_name=filename)


def download_and_estimate_PSDs(
    data_dir: str,
    settings: dict,
    time_segments: dict,
    verbose=False,
):

    dataset_settings = settings["dataset_settings"]
    run = dataset_settings["observing_run"]
    detectors = (
        time_segments.keys()
    )  # time_segments may only contain a subset of all detectors for parallelization

    f_min = 0
    f_max = dataset_settings.get("f_max", (dataset_settings["f_s"] / 2))
    T = dataset_settings["T"]
    T_PSD = dataset_settings["T_PSD"]
    f_s = dataset_settings["f_s"]

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

    window_kwargs = {
        "f_s": f_s,
        "roll_off": dataset_settings["window"]["roll_off"],
        "type": dataset_settings["window"]["type"],
        "T": T,
    }
    w = get_window(window_kwargs)

    for det in detectors:
        psd_path = get_path_raw_data(data_dir, run, det)
        estimation_kwargs = {
            "det": det,
            "time_segment": T,
            "window": w,
            "f_s": f_s,
            "num_segments": int(T_PSD / T),
        }
        task_func = partial(
            estimate_func,
            run=run,
            domain=domain,
            estimation_kwargs=estimation_kwargs,
            psd_path=psd_path,
            settings=settings
        )
        num_processes = settings["local"]["num_processes"]
        if num_processes > 1:
            with Pool(processes=num_processes) as pool:
                with tqdm(total=len(time_segments[det]), disable=not verbose) as pbar:
                    for _, i in enumerate(pool.imap_unordered(task_func, time_segments[det])):
                        pbar.update()

        else:
            with tqdm(total=len(time_segments[det]), disable=not verbose) as pbar:
                for _, i in enumerate(map(task_func, time_segments[det])):
                    pbar.update()

def lorentzian_eval(x, f0, A, Q):
    if f0 == 0 or A < 0:
        return np.zeros_like(x)
    delta_f = (x[-1] - x[0]) / 4
    truncate = np.where(np.abs(x - f0) <= delta_f, 1, np.exp(-np.abs(x - f0) / delta_f))
    # equivalent to a large delta_f. This helps fit adjacent spectral features without a gap in between
    # truncate = 1
    return truncate * A * (f0**4) / ((x * f0) ** 2 + Q**2 * (f0**2 - x**2) ** 2)


def get_psds_from_params_dict(params, frequencies, scale_factor, smoothen=False):

    xs = params["x_positions"]
    ys = params["y_values"]
    spectral_features = params["spectral_features"]
    variance = params["variance"]

    num_psds = ys.shape[0]
    num_spectral_segments = params["spectral_features"].shape[1]
    frequency_segments = np.array_split(
        np.arange(frequencies.shape[0]), num_spectral_segments
    )

    psds = np.zeros((num_psds, len(frequencies)))
    for i in range(num_psds):
        spline = scipy.interpolate.CubicSpline(xs, ys[i, :])
        base_noise = spline(frequencies)

        lorentzians = np.array([])
        for j, seg in enumerate(frequency_segments):
            f0, A, Q = spectral_features[i, j, :]
            lorentzian = lorentzian_eval(frequencies[seg], f0, A, Q)
            # small amplitudes are not modeled to maintain smoothness
            if np.max(lorentzian) <= 3 * variance:
                lorentzian = np.zeros_like(frequencies[seg])
            lorentzians = np.concatenate((lorentzians, lorentzian), axis=0)
        assert lorentzians.shape == frequencies.shape

        if smoothen:
            psds[i, :] = np.exp(base_noise + lorentzians) / scale_factor
        else:
            psds[i, :] = (
                np.exp(np.random.normal(base_noise + lorentzians, variance))
                / scale_factor
            )
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(16,9))
        # plt.xlim(20, 2000)
        # plt.ylim(1.e-47, 1.e-42)
        # # plt.loglog(frequencies, np.exp(np.random.normal(base_noise + lorentzians, variance)) / scale_factor)
        # plt.loglog(frequencies, psds[i, :])
        # plt.savefig("/work/jwildberger/dingo-dev/review/ASD_datasets/debug.pdf")
        # exit(1)
    return psds


# TODO: make this more generic to load arbitrary numpy files as dict
def load_params_from_files(psd_paths):
    times = np.zeros(len(psd_paths))
    # initialize parameter dict
    psd = np.load(join(psd_paths[0]), allow_pickle=True).item()

    parameters = {
        "x_positions": psd["parameters"]["x_positions"],
        "y_values": np.zeros(
            (len(psd_paths), psd["parameters"]["x_positions"].shape[0])
        ),
        "spectral_features": np.zeros(
            (
                len(psd_paths),
                psd["parameters"]["spectral_features"].shape[0],
                psd["parameters"]["spectral_features"].shape[1],
            )
        ),
        "variance": psd["parameters"]["variance"],
    }

    for ind, filename in enumerate(psd_paths):
        psd = np.load(filename, allow_pickle=True).item()
        parameters["spectral_features"][ind, :, :] = psd["parameters"][
            "spectral_features"
        ]
        parameters["y_values"][ind, :] = psd["parameters"]["y_values"]
        times[ind] = psd["time"][0]

    return parameters, times


# def create_dataset_from_files(data_dir: str, settings: dict):
#
#     """
#     Creates a .hdf5 ASD datset file for an observing run using the estimated detector PSDs.
#
#     Parameters
#     ----------
#     data_dir : str
#         Path to the directory where the PSD dataset will be stored
#     settings : dict
#         Dictionary of settings that are used for the dataset generation
#     -------
#     """
#
#     f_min = 0
#     f_max = settings["dataset_settings"].get("f_max", (settings["dataset_settings"]["f_s"] / 2))
#     T_PSD = settings["dataset_settings"]["T_PSD"]
#     T_gap = settings["dataset_settings"]["T_gap"]
#     T = settings["dataset_settings"]["T"]
#
#     delta_f = 1 / T
#     domain = build_domain(
#         {
#             "type": "FrequencyDomain",
#             "f_min": f_min,
#             "f_max": f_max,
#             "delta_f": delta_f,
#             "window_factor": None,
#         }
#     )
#     ind_min, ind_max = domain.min_idx, domain.max_idx
#
#     dataset_dict = {
#         "settings": {"dataset_settings": settings["dataset_settings"], "domain_dict": domain.domain_dict}
#     }
#     asds_dict = {}
#     gps_times_dict = {}
#
#     for ifo in settings["dataset_settings"]["detectors"]:
#
#         path_raw_psds = get_path_raw_data(
#             data_dir, settings["dataset_settings"]["observing_run"], ifo, T_PSD, T_gap
#         )
#         filenames = [el for el in os.listdir(path_raw_psds) if el.endswith(".npy")]
#
#         Nf = ind_max - ind_min + 1
#         asds = np.zeros((len(filenames), Nf))
#         times = np.zeros(len(filenames))
#
#         for ind, filename in enumerate(filenames):
#             psd = np.load(join(path_raw_psds, filename), allow_pickle=True).item()
#             asds[ind, :] = np.sqrt(psd["psd"][ind_min : ind_max + 1])
#             times[ind] = psd["time"][0]
#
#         asds_dict[ifo] = asds
#         gps_times_dict[ifo] = times
#
#     dataset_dict["asds"] = asds_dict
#     dataset_dict["gps_times"] = gps_times_dict
#
#     dataset = ASDDataset(dictionary=dataset_dict)
#     dataset.to_file(file_name=join(data_dir, f"asds_{settings['dataset_settings']['observing_run']}.hdf5"))
