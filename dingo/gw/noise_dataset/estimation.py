import argparse
import os
import pickle
from functools import partial
from io import StringIO
from multiprocessing import Pool
from os.path import join

import numpy as np
import requests
import yaml
from tqdm import tqdm

from dingo.gw.domains import build_domain
from dingo.gw.download_strain_data import download_psd
from dingo.gw.gwutils import get_window
from dingo.gw.noise_dataset.ASD_dataset import ASDDataset
from dingo.gw.noise_dataset.parameterization import parameterize_single_psd
from dingo.gw.noise_dataset.utils import get_path_raw_data

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

    Parameters
    ----------
    data_dir
    settings

    Returns
    -------

    """

    time_segments = {}

    run = settings["observing_run"]
    T_PSD = settings["T_PSD"]
    T_gap = settings["T_gap"]
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
            valid_segments = valid_segments[
                :num_psds_max
            ]  # random.sample(valid_segments, num_psds_max)

        time_segments[detector] = valid_segments

    filename = join(data_dir, "tmp", run, "psd_time_segments.pkl")
    with open(filename, "wb") as f:
        pickle.dump(time_segments, f)
    return time_segments


def estimate_func(seg, domain, estimation_kwargs, psd_path, settings, override=False):

    start, end = seg[0], seg[1]
    filename = join(psd_path, f"asd_{start}.hdf5")

    parameterization_settings = settings.get("parameterization_settings", None)

    # nothing to do
    if os.path.exists(filename) and not parameterization_settings:
        return

    dataset_dict = {
        "settings": {
            "dataset_settings": settings["dataset_settings"],
            "domain_dict": domain.domain_dict,
        }
    }

    psd = None
    det = estimation_kwargs["det"]

    parameterized = False
    try:
        dataset = ASDDataset(file_name=filename)
        parameterized = hasattr(dataset, "parameters") and not override
        print(parameterized)
        if parameterized:
            return

        dataset.update_domain(domain.domain_dict)
        psd = dataset.asds[det][0] ** 2

    # if file doesn't exist or new domain is incompatible, download PSD
    except (FileNotFoundError, ValueError):
        psd = download_psd(time_start=start, **estimation_kwargs)
    # only parameterize, if settings are passed and any existing parameterization should be overwritten
    if parameterization_settings:
        dataset_dict["settings"]["parameterization_settings"] = parameterization_settings
        params = parameterize_single_psd(psd, domain, parameterization_settings)
        dataset_dict["parameters"] = {det: params}

    asd = np.sqrt(psd[domain.min_idx: domain.max_idx + 1])
    gps_time = start

    dataset_dict["asds"] = {det: np.array([asd])}
    dataset_dict["gps_times"] = {det: np.array([gps_time])}

    dataset = ASDDataset(dictionary=dataset_dict)
    dataset.to_file(file_name=filename)

    return


def download_and_estimate_psds(
    data_dir: str,
    settings: dict,
    time_segments: dict,
    verbose=False,
    override=False
):
    dataset_settings = settings["dataset_settings"]
    run = dataset_settings["observing_run"]
    detectors = (
        time_segments.keys()
    )  # time_segments may only contain a subset of all detectors for parallelization

    f_min = dataset_settings.get("f_min", 0)
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
            domain=domain,
            estimation_kwargs=estimation_kwargs,
            psd_path=psd_path,
            settings=settings,
            override=override
        )
        num_processes = settings["local"]["num_processes"]
        if num_processes > 1:
            with Pool(processes=num_processes) as pool:
                with tqdm(total=len(time_segments[det]), disable=not verbose) as pbar:
                    for _, i in enumerate(
                        pool.imap_unordered(task_func, time_segments[det])
                    ):
                        pbar.update()

        else:
            with tqdm(total=len(time_segments[det]), disable=not verbose) as pbar:
                for _, i in enumerate(map(task_func, time_segments[det])):
                    pbar.update()


def download_and_estimate_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path where the PSD data is to be stored. Must contain a 'settings.yaml' file.",
    )
    parser.add_argument(
        "--settings_file",
        type=str,
        required=True,
        help="Path to a settings file in case two different datasets are generated in the sam directory",
    )
    parser.add_argument(
        "--time_segments_file",
        type=str,
        default=None,
        help="Path to a file containing the time segments for which PSDs should be estimated",
    )

    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()

    # Load settings
    with open(args.settings_file, "r") as f:
        settings = yaml.safe_load(f)

    with open(args.time_segments_file, "rb") as f:
        time_segments = pickle.load(f)

    download_and_estimate_psds(args.data_dir, settings, time_segments, override=args.override)
