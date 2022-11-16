import copy

import numpy as np
from scipy import stats
from abc import ABC, abstractmethod
from dingo.gw.ASD_dataset.parameterization import parameterize_single_psd
from gwpy.time import tconvert


class Sampler(ABC):
    @abstractmethod
    def __init__(self, settings):
        self.settings = settings

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class KDE(Sampler):
    def __init__(self, settings, frequencies, data_dict=None, rescaling_psd=None):
        super().__init__(settings)
        self.frequencies = frequencies
        self.spectral_fit = None
        self.y_fit = None
        self.rescaled_psd = rescaling_psd

        if data_dict is not None:
            self.data_dict = data_dict
            self.fit(data_dict)

    def __call__(self, *args, **kwargs):
        pass

    def fit(self, data_dict):

        y_values = data_dict["y_values"]
        spectral_features = data_dict["spectral_features"]
        num_spectral_segments = spectral_features.shape[1]

        weights = None

        if self.rescaled_psd:
            spectral_features = np.concatenate(
                (
                    spectral_features,
                    np.expand_dims(
                        self.rescaled_psd["parameters"]["spectral_features"],
                        0,
                    ),
                ),
                axis=0,
            )

            weights = [1.0] * spectral_features.shape[0]
            weights[-1] = (
                spectral_features.shape[0] / 4
            )  # 20% weighting on the last sample

        if spectral_features.shape[0] < 2:
            raise RuntimeError(
                "At least 2 PSDs are needed for computing the KDEs and {} were "
                "supplied".format(spectral_features.shape[0])
            )

        self.spectral_fit = [
            stats.gaussian_kde(
                spectral_features[:, i, :].T,
                bw_method=float(self.settings["bw_spectral"]),
                weights=weights,
            )
            for i in range(spectral_features.shape[1])
        ]

        y_fit = []

        def get_index_for_elem(arr, elem):
            return (np.abs(arr - elem)).argmin()

        split_indices = [get_index_for_elem(data_dict["x_positions"], f) + 1 for f in self.settings["split_frequencies"]]
        split_indices.append(len(data_dict["x_positions"]))
        split_indices.insert(0, 0)

        for i in range(len(split_indices) - 1):
            vals = y_values[:, split_indices[i]:split_indices[i+1]].T
            kde_vals = stats.gaussian_kde(
                vals, bw_method=float(self.settings["bw_spline"])
            )
            y_fit.append(kde_vals)

        self.y_fit = y_fit

    def sample(self, size=5000):

        data_dict = {}  # copy.deepcopy(self.data_dict)
        data_dict["x_positions"] = self.data_dict["x_positions"]
        data_dict["variance"] = self.data_dict["variance"]

        if not (self.y_fit and self.spectral_fit):
            raise ValueError("KDEs have not been fit yet. Please run fit first.")

        features = np.zeros((size, len(self.spectral_fit), 3))
        for i, kde in enumerate(self.spectral_fit):
            features[:, i, :] = kde.resample(size=size).T
        data_dict["spectral_features"] = features

        y_values = np.zeros((size, 0))
        for i, kde in enumerate(self.y_fit):
            y_values = np.concatenate((y_values, kde.resample(size=size).T), axis=1)

        # rescale base noise if a psd has been passed
        if self.rescaled_psd:
            assert np.array_equal(
                self.rescaled_psd["parameters"]["x_positions"],
                data_dict["x_positions"],
            )
            assert (
                self.rescaled_psd["parameters"]["spectral_features"].shape[0]
                == data_dict["spectral_features"].shape[1]
            )
            y_values_mean = np.mean(y_values, axis=0)
            y_values = (
                y_values
                - y_values_mean[None, :]
                + self.rescaled_psd["parameters"]["y_values"]
            )
        data_dict["y_values"] = y_values

        return data_dict


def build_sampler(settings, frequencies, data_dict, rescaling_psd=None):
    if "method" not in settings:
        raise ValueError('Missing key "method" in sampling settings')
    sampling_method = settings["method"]
    if sampling_method.lower() == "kde":
        return KDE(settings, frequencies, data_dict, rescaling_psd=rescaling_psd)
    else:
        raise NotImplementedError("Unknown sampling method")