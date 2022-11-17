import copy
import yaml
import pickle
import argparse
from os.path import join
from typing import List

import numpy as np

from dingo.gw.noise_dataset.ASD_dataset import ASDDataset


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


def merge_datasets(data_dir, dataset_settings, time_segments, merged_name=None):
    """

    Parameters
    ----------
    data_dir
    dataset_settings
    time_segments
    merged_name

    Returns
    -------

    """

    print(f"Merging {len(time_segments)} datasets into one.")

    detectors = dataset_settings["detectors"]
    run = dataset_settings["observing_run"]

    asds_dict = dict(zip(detectors, [[] for i in range(len(detectors))]))
    gps_times_dict = dict(zip(detectors, [[] for i in range(len(detectors))]))
    merged_dict = {"asds": asds_dict, "gps_times": gps_times_dict}

    for det in detectors:
        file_dir = get_path_raw_data(data_dir, run, det)
        for seg in time_segments[det]:
            start_time = seg[0]
            filename = join(file_dir, f"asd_{start_time}.hdf5")
            try:
                # TODO: should this structure be kept? Or e.g. tuples of GPS time and ASD?
                dataset = ASDDataset(filename)
                asds_dict[det].append(dataset.asds[det][0])
                gps_times_dict[det].append(dataset.gps_times[det])
            except FileNotFoundError:
                print(f"file {filename} not found. Skipping it...")

        asds_dict[det] = np.array(asds_dict[det])
        gps_times_dict[det] = np.array(gps_times_dict[det])

    # copy settings from last dataset
    merged_dict["settings"] = copy.deepcopy(dataset.settings)

    merged = ASDDataset(dictionary=merged_dict)
    if merged_name is None:
        merged_name = f"asds_{run}.hdf5"
    merged.to_file(join(data_dir, merged_name))


def merge_datasets_cli():
    """
    Command-line function to combine a collection of datasets into one. Used for
    parallelized waveform generation.
    """

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
        required=True,
        help="Path to a file containing the time segments for which PSDs should be estimated",
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default=None,
        help="File name of merged dataset",
    )
    args = parser.parse_args()

    with open(args.settings_file, "r") as f:
        settings = yaml.safe_load(f)

    with open(args.time_segments_file, "rb") as f:
        time_segments = pickle.load(f)

    merge_datasets(
        args.data_dir, settings["dataset_settings"], time_segments, args.out_name
    )
