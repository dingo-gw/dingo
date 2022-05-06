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
Contains links for PSD segment lists with quality label BURST_CAT2 from the Gravitationa Wave Open Science Center.
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


def get_valid_segments(segs, T_PSD, T_gap):
    """
    Given the segments `segs` and the time constraints `T_PSD`, `delta_T`, return all segments
    that can be used to estimate a PSD

    Parameters
    ----------
    segs : Tuple[int, int, int]
        Contains the start- and end gps_times as well as their difference that have been fetched from the GWOSC website
    T_PSD : int
        number of seconds used to estimate PSD
    T_gap : int
        number of seconds between two adjacent PSDs. May be negative to indicate an overlap

    Returns
    -------
    All segments that can be used to estimate a PSD
    """
    segs = np.array(segs, dtype=int)
    segs = segs[segs[:, 2] >= T_PSD]

    valid_segs = []
    for idx in range(segs.shape[0]):
        seg = segs[idx, :]
        start_time = seg[0]
        end_time = start_time + T_PSD

        while end_time in range(seg[0], seg[1] + 1):
            valid_segs.append((start_time, end_time))
            start_time = end_time + T_gap
            end_time = start_time + T_PSD

    return valid_segs


def get_path_raw_data(data_dir, run, detector, T_PSD=1024, T_gap=0):
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
    T_PSD : str
        number of seconds used to estimate PSD
    T_gap : str
        number of seconds between two adjacent PSDs

    Returns
    -------
    the path where the data is stored
    """
    return join(data_dir, "tmp", run, detector, str(T_PSD) + "_" + str(T_gap))


def download_and_estimate_PSDs(
        data_dir: str, run: str, detector: str, settings: dict, verbose=False
):
    """
    Download segment lists from the official GWOSC website that have the BURST_CAT_2 quality label. A .npy file
    is created for every PSD that will be in the final dataset. These are stored in data_dir/tmp and may be removed
    once the final dataset has been created.

    Parameters
    ----------
    data_dir : str
        Path to the directory where the PSD dataset will be stored
    run : str
        Observing run that is used for the PSD dataset generation
    detector : str
        Detector that is used for the PSD dataset generation
    settings : dict
        Dictionary of settings that are used for the dataset generation
    verbose : bool
        If true, there will be a progress bar indicating 

    -------

    """ ""

    key = run + "_" + detector
    urls = URL_DIRECTORY[key]

    starts, stops, durations = [], [], []
    for url in urls:
        r = requests.get(url, allow_redirects=True)
        c = StringIO(r.content.decode("utf-8"))
        starts_seg, stops_seg, durations_seg = np.loadtxt(c, dtype="int", unpack=True)
        starts = np.hstack([starts, starts_seg])
        stops = np.hstack([stops, stops_seg])
        durations = np.hstack([durations, durations_seg])

    T_PSD = settings["T_PSD"]
    T_gap = settings["T_gap"]
    T = settings["T"]
    f_s = settings["f_s"]

    window_kwargs = {
        "f_s": f_s,
        "roll_off": settings["window"]["roll_off"],
        "type": settings["window"]["type"],
        "T": T,
    }
    w = get_window(window_kwargs)

    path_raw_psds = get_path_raw_data(data_dir, run, detector, T_PSD, T_gap)
    os.makedirs(path_raw_psds, exist_ok=True)

    valid_segments = get_valid_segments(
        list(zip(starts, stops, durations)), T_PSD=T_PSD, T_gap=T_gap
    )

    num_psds_max = settings["num_psds_max"]
    if num_psds_max >= 1:
        valid_segments = random.sample(valid_segments, num_psds_max)

    print(
        f"Fetching data and computing Welch's estimate of {len(valid_segments)} valid segments:\n"
    )

    for index, (start, end) in enumerate(tqdm(valid_segments, disable=not verbose)):
        filename = join(path_raw_psds, "psd_{:05d}.npy".format(index))

        if not os.path.exists(filename):
            psd = download_psd(
                det=detector,
                time_start=start,
                time_segment=T,
                window=w,
                f_s=f_s,
                num_segments=int(T_PSD / T),
            )
            np.save(
                filename,
                {
                    "detector": detector,
                    "segment": (index, start, end),
                    "time": (start, end),
                    "psd": psd,
                    "tukey window": {
                        "f_s": f_s,
                        "roll_off": settings["window"]["roll_off"],
                        "T": T,
                    },
                },
            )


def create_dataset_from_files(
        data_dir: str, run: str, detectors: List[str], settings: dict
):

    """
    Creates a .hdf5 ASD datset file for an observing run using the estimated detector PSDs.

    Parameters
    ----------
    data_dir : str
        Path to the directory where the PSD dataset will be stored
    run : str
        Observing run that is used for the ASD dataset generation
    detectors : List[str]
        Detector data that is used for the ASD dataset generation
    settings : dict
        Dictionary of settings that are used for the dataset generation
    -------
    """

    f_min = 0
    f_max = settings["f_s"] / 2
    T_PSD = settings["T_PSD"]
    T_gap = settings["T_gap"]
    T = settings["T"]

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
    ind_min, ind_max = domain.min_idx, domain.max_idx

    dataset_dict = {"settings": {"dataset_settings": settings, "domain_dict": domain.domain_dict}}
    asds_dict = {}
    gps_times_dict = {}

    for ifo in detectors:

        path_raw_psds = get_path_raw_data(data_dir, run, ifo, T_PSD, T_gap)
        filenames = [el for el in os.listdir(path_raw_psds) if el.endswith(".npy")]

        Nf = ind_max - ind_min + 1
        asds = np.zeros((len(filenames), Nf))
        times = np.zeros(len(filenames))

        for ind, filename in enumerate(filenames):
            psd = np.load(join(path_raw_psds, filename), allow_pickle=True).item()
            asds[ind, :] = np.sqrt(psd["psd"][ind_min : ind_max + 1])
            times[ind] = psd["time"][0]

        asds_dict[ifo] = asds
        gps_times_dict[ifo] = times

    dataset_dict["asds"] = asds_dict
    dataset_dict["gps_times"] = gps_times_dict

    dataset = ASDDataset(dictionary=dataset_dict)
    dataset.to_file(file_name=join(data_dir, f"asds_{run}.hdf5"))
