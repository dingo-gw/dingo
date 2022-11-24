import argparse
import copy
import pickle
from os.path import join

import numpy as np
import scipy
import yaml

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


def get_index_for_elem(arr, elem):
    return (np.abs(arr - elem)).argmin()


def lorentzian_eval(x, f0, A, Q, delta_f=None):

    if f0 == 0 or A < 0:
        return np.zeros_like(x)

    # used to truncate tails of Lorentzian if necessary. Will have no effect, if delta_f sufficiently large
    truncate = (
        np.where(np.abs(x - f0) <= delta_f, 1, np.exp(-np.abs(x - f0) / delta_f))
        if delta_f
        else np.ones_like(x)
    )

    return truncate * A * (f0**4) / ((x * f0) ** 2 + Q**2 * (f0**2 - x**2) ** 2)


def reconstruct_psds_from_parameters(
    parameters_dict, domain, parameterization_settings, smoothen=False
):

    xs = parameters_dict["x_positions"]
    ys_list = parameters_dict["y_values"]
    spectral_features_list = parameters_dict["spectral_features"]

    if spectral_features_list.ndim == 2 and ys_list.ndim == 1:
        spectral_features_list = spectral_features_list[np.newaxis]
        ys_list = ys_list[np.newaxis]

    num_psds = ys_list.shape[0]
    assert num_psds == spectral_features_list.shape[0]

    sigma = parameterization_settings["sigma"]
    frequencies = domain.sample_frequencies

    num_spectral_segments = spectral_features_list.shape[1]
    frequency_segments = np.array_split(
        np.arange(frequencies.shape[0]), num_spectral_segments
    )

    psds = []
    for i in range(num_psds):
        ys = ys_list[i, :]
        spectral_features = spectral_features_list[i, :, :]
        spline = scipy.interpolate.CubicSpline(xs, ys)
        base_noise = spline(frequencies)

        lorentzians = np.array([])
        for j, seg in enumerate(frequency_segments):
            f0, A, Q = spectral_features[j, :]
            lorentzian = lorentzian_eval(frequencies[seg], f0, A, Q)
            # small amplitudes are not modeled to maintain smoothness
            if np.max(lorentzian) <= 3 * sigma:
                lorentzian = np.zeros_like(frequencies[seg])
            lorentzians = np.concatenate((lorentzians, lorentzian), axis=0)
        assert lorentzians.shape == frequencies.shape

        if smoothen:
            psd = np.exp(base_noise + lorentzians)
        else:
            psd = np.exp(np.random.normal(base_noise + lorentzians, sigma))
        psds.append(psd)
    return np.array(psds)


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

    asds_dict = dict(zip(detectors, [[] for _ in range(len(detectors))]))
    gps_times_dict = dict(zip(detectors, [[] for _ in range(len(detectors))]))

    smoothen = dataset_settings.get("smoothen", False)
    parameters_dict = dict(zip(detectors, [{} for i in range(len(detectors))]))

    for det in detectors:
        file_dir = get_path_raw_data(data_dir, run, det)
        filenames = [join(file_dir, f"asd_{seg[0]}.hdf5") for seg in time_segments[det]]
        datasets = [ASDDataset(filename) for filename in filenames]

        parameters = dict(
            zip(["x_positions", "y_values", "spectral_features"], [None, [], []])
        )

        for dataset in datasets:
            asds_dict[det].append(dataset.asds[det][0])
            gps_times_dict[det].append(dataset.gps_times[det])

            try:
                parameters["y_values"].append(dataset.parameters[det]["y_values"])
                parameters["spectral_features"].append(
                    dataset.parameters[det]["spectral_features"]
                )
            except AttributeError:
                pass

        if parameters["y_values"] and parameters["spectral_features"]:

            parameters["y_values"] = np.array(parameters["y_values"])
            parameters["spectral_features"] = np.array(parameters["spectral_features"])

            parameters["x_positions"] = dataset.parameters[det]["x_positions"]
            parameters_dict[det] = parameters
        if smoothen:
            psds = reconstruct_psds_from_parameters(
                parameters,
                dataset.domain,
                dataset.settings["parameterization_settings"],
                smoothen=True,
            )
            asds_dict[det] = np.sqrt(
                psds[:, dataset.domain.min_idx : dataset.domain.max_idx + 1]
            )
        else:
            asds_dict[det] = np.array(asds_dict[det])
        gps_times_dict[det] = np.array(gps_times_dict[det])

    merged_dict = {
        "asds": asds_dict,
        "gps_times": gps_times_dict,
        "parameters": parameters_dict,
    }
    # copy settings from last dataset
    merged_dict["settings"] = copy.deepcopy(dataset.settings)

    merged = ASDDataset(dictionary=merged_dict)

    if merged_name is None:
        merged_name = f"asds_{run}.hdf5"
    merged.to_file(join(data_dir, merged_name))
    return merged


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
