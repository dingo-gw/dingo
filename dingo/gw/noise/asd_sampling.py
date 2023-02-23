import copy
import argparse
import yaml
from os.path import join

import numpy as np
from scipy import stats
from abc import ABC, abstractmethod
from dingo.gw.noise.asd_parameterization import parameterize_single_psd
from dingo.gw.noise.asd_dataset import ASDDataset
from dingo.gw.noise.utils import (
    get_index_for_elem,
    reconstruct_psds_from_parameters,
)
from gwpy.time import tconvert


def load_rescaling_psd(filename, detector, parameterization_settings=None):
    rescaling_psd_dataset = ASDDataset(filename)
    try:
        params = rescaling_psd_dataset.parameters[detector]
    except:
        dataset_dict = rescaling_psd_dataset.to_dictionary()
        psd = rescaling_psd_dataset.asds[detector][0] ** 2
        params = parameterize_single_psd(
            psd, rescaling_psd_dataset.domain, parameterization_settings
        )
        dataset_dict["parameters"] = {detector: params}
        new_dataset = ASDDataset(dictionary=dataset_dict)
        new_dataset.to_file(filename)

    return params


class KDE:
    def __init__(self, asd_dataset, sampling_settings):
        self.asd_dataset = asd_dataset
        self.domain = asd_dataset.domain
        # try:
        self.parameter_dicts = asd_dataset.parameters
        self.sampling_settings = sampling_settings

        detectors = asd_dataset.parameters.keys()
        self.spectral_kde = {det: [] for det in detectors}
        self.broadband_kde = {det: [] for det in detectors}

    def fit(self, weights=None):

        for det, param_dict in self.parameter_dicts.items():
            y_values = param_dict["y_values"]
            x_positions = param_dict["x_positions"]
            spectral_features = param_dict["spectral_features"]

            for i in range(spectral_features.shape[1]):
                try:
                    spectral_kde = stats.gaussian_kde(
                        spectral_features[:, i, :].T,
                        bw_method=float(self.sampling_settings["bandwidth_spectral"]),
                        weights=weights,
                    )
                except np.linalg.LinAlgError:
                    print(
                        "Warning: Singular Matrix encountered in spectral KDE. Adding small Gaussian noise..."
                    )
                    perturbed_features = spectral_features[:, i, :] + np.random.normal(
                        0, 1.0e-4, size=spectral_features[:, i, :].shape
                    )
                    spectral_kde = stats.gaussian_kde(
                        perturbed_features.T,
                        bw_method=float(self.sampling_settings["bandwidth_spectral"]),
                        weights=weights,
                    )
                self.spectral_kde[det].append(spectral_kde)

            split_indices = [0, len(x_positions)]
            split_indices += [
                get_index_for_elem(x_positions, f) + 1
                for f in self.sampling_settings["split_frequencies"]
            ]
            split_indices = sorted(split_indices)

            for i in range(len(split_indices) - 1):
                vals = y_values[:, split_indices[i] : split_indices[i + 1]].T
                kde_vals = stats.gaussian_kde(
                    vals, bw_method=float(self.sampling_settings["bandwidth_spline"])
                )
                self.broadband_kde[det].append(kde_vals)

    def sample(self):

        num_samples = self.sampling_settings["num_samples"]
        smoothen = self.sampling_settings.get("smoothen", False)
        domain = self.asd_dataset.domain

        parameters_dicts = copy.deepcopy(self.parameter_dicts)
        asds_dict = {}

        for det, param_dict in self.parameter_dicts.items():
            asds_dict[det] = []
            xs = param_dict["x_positions"]

            features = np.zeros((num_samples, len(self.spectral_kde[det]), 3))
            for i, kde in enumerate(self.spectral_kde[det]):
                features[:, i, :] = kde.resample(size=num_samples).T

            parameters_dicts[det]["spectral_features"] = features

            y_values = np.zeros((num_samples, 0))

            for i, kde in enumerate(self.broadband_kde[det]):
                y_values = np.concatenate(
                    (y_values, kde.resample(size=num_samples).T), axis=1
                )

            # rescale base noise if a psd has been passed
            if "rescaling_psd_paths" in self.sampling_settings:
                try:
                    rescaling_params = load_rescaling_psd(
                        self.sampling_settings["rescaling_psd_paths"][det],
                        det,
                        self.asd_dataset.settings["parameterization_settings"],
                    )
                except KeyError:
                    continue

                y_values_mean = np.mean(y_values, axis=0)
                y_values = (
                    y_values - y_values_mean[None, :] + rescaling_params["y_values"]
                )
            parameters_dicts[det]["y_values"] = y_values

            # reconstruct sampled parameters back to frequency space

            psds = reconstruct_psds_from_parameters(
                parameters_dicts[det],
                domain,
                self.asd_dataset.settings["parameterization_settings"],
                smoothen=smoothen,
            )
            asds_dict[det] = np.sqrt(psds[:, domain.min_idx : domain.max_idx + 1])

        dataset_dict = {}
        dataset_dict["settings"] = copy.deepcopy(self.asd_dataset.settings)
        dataset_dict["settings"]["sampling_settings"] = self.sampling_settings
        dataset_dict["asds"] = asds_dict
        dataset_dict["gps_times"] = self.asd_dataset.gps_times
        dataset_dict["parameters"] = parameters_dicts
        return ASDDataset(dictionary=dataset_dict)


def resample_dataset_cli():

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
        "--out_name",
        type=str,
        default=None,
        help="File name of merged dataset",
    )
    args = parser.parse_args()

    with open(args.settings_file, "r") as f:
        settings = yaml.safe_load(f)

    run = settings["dataset_settings"]["observing_run"]
    filename = (
        args.out_name if args.out_name else join(args.data_dir, f"asds_{run}.hdf5")
    )

    sampling_settings = settings.get("sampling_settings", None)

    asd_dataset = ASDDataset(filename)
    if sampling_settings:
        kde = KDE(asd_dataset, sampling_settings)
        kde.fit()
        dataset = kde.sample()
        dataset.to_file(filename)
