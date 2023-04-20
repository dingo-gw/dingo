import argparse
import os
import pickle
from os.path import join

import numpy as np
import yaml
from tqdm import tqdm

from dingo.gw.domains import build_domain
from dingo.gw.download_strain_data import estimate_single_psd
from dingo.gw.gwutils import get_window
from dingo.gw.noise.asd_dataset import ASDDataset
from dingo.gw.noise.utils import psd_data_path


def download_and_estimate_psds(
    data_dir: str,
    settings: dict,
    time_segments: dict,
    verbose=False,
):
    """
    Downloads strain data for the specified time segments and estimates PSDs based on these

    Parameters
    ----------
    data_dir : str
        Path to the directory where the PSD dataset will be stored
    settings : dict
        Settings that determine the segments
    time_segments : dict
        specifying the time segments used for downloading the data
    verbose : bool
        optional parameter determining if progress should be printed
    Returns
    -------
    A dictionary containing the paths to the dataset files
    """
    dataset_settings = settings["dataset_settings"]
    run = dataset_settings["observing_run"]
    detectors = (
        time_segments.keys()
    )  # time_segments may only contain a subset of all detectors for parallelization

    f_min = dataset_settings.get("f_min", 0)
    f_max = dataset_settings.get("f_max", (dataset_settings["f_s"] / 2))
    T = dataset_settings["T"]
    time_psd = dataset_settings["time_psd"]
    f_s = dataset_settings["f_s"]

    channels = dataset_settings.get("channels", None)

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
    asd_filename_list = {det: [] for det in detectors}
    for det in detectors:
        psd_path = psd_data_path(data_dir, run, det)
        os.makedirs(psd_path, exist_ok=True)
        estimation_kwargs = {
            "time_segment": T,
            "window": w,
            "f_s": f_s,
            "num_segments": int(time_psd / T),
        }
        if channels:
            estimation_kwargs["channel"] = channels[det]
        else:
            estimation_kwargs["det"] = det

        for index, (start, end) in enumerate(
            tqdm(time_segments[det], disable=not verbose)
        ):
            filename = join(psd_path, f"asd_{start}.hdf5")
            asd_filename_list[det].append(filename)
            if not os.path.exists(filename):
                dataset_dict = {
                    "settings": {
                        "dataset_settings": settings["dataset_settings"],
                        "domain_dict": domain.domain_dict,
                    }
                }
                psd = estimate_single_psd(time_start=start, **estimation_kwargs)
                asd = np.sqrt(psd[domain.min_idx : domain.max_idx + 1])
                gps_time = start

                dataset_dict["asds"] = {det: np.array([asd])}
                dataset_dict["gps_times"] = {det: np.array([gps_time])}

                dataset = ASDDataset(dictionary=dataset_dict)
                dataset.to_file(file_name=filename)

    return asd_filename_list


def download_and_estimate_cli():
    """
    Command-line function to download strain data and estimate PSDs based on the data. Used for
    parallelized ASD dataset generation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path where the PSD data is to be stored.",
    )
    parser.add_argument(
        "--settings_file",
        type=str,
        required=True,
        help="Path to a settings file containing the settings for the dataset generation",
    )
    parser.add_argument(
        "--time_segments_file",
        type=str,
        default=None,
        help="Path to a file containing the time segments for which PSDs should be estimated",
    )

    args = parser.parse_args()

    # Load settings
    with open(args.settings_file, "r") as f:
        settings = yaml.safe_load(f)

    with open(args.time_segments_file, "rb") as f:
        time_segments = pickle.load(f)

    download_and_estimate_psds(args.data_dir, settings, time_segments)
