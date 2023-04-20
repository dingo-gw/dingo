import copy

import numpy as np
from scipy import stats
from dingo.gw.noise.synthetic.asd_parameterization import fit_broadband_noise
from dingo.gw.noise.asd_dataset import ASDDataset
from dingo.gw.noise.synthetic.utils import (
    get_index_for_elem,
)


class KDE:
    """
    Kernel Density Estimation (KDE) class for sampling ASDs.
    """
    def __init__(self, parameter_dict, sampling_settings):
        """
        Parameters
        ----------
        parameter_dict : dict
            Dictionary containing the parameters of the ASDs used for fitting the synthetic distribution.
        sampling_settings : dict
            Dictionary containing the settings for the sampling.

        """

        self.parameter_dicts = parameter_dict
        self.settings = sampling_settings

        detectors = parameter_dict.keys()
        self.spectral_kde = {det: [] for det in detectors}
        self.broadband_kde = {det: [] for det in detectors}

    def fit(self, weights=None):
        """
        Fit the KDEs to the parameters saved in 'self.parameter_dict'.
        Parameters
        ----------
        weights : array_like, optional
            Weights for the KDEs. If None, all weights are set to 1.
        """

        for det, param_dict in self.parameter_dicts.items():
            y_values = param_dict["y_values"]
            x_positions = param_dict["x_positions"]
            spectral_features = param_dict["spectral_features"]

            for i in range(spectral_features.shape[1]):
                try:
                    spectral_kde = stats.gaussian_kde(
                        spectral_features[:, i, :].T,
                        bw_method=float(self.settings["bandwidth_spectral"]),
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
                        bw_method=float(self.settings["bandwidth_spectral"]),
                        weights=weights,
                    )
                self.spectral_kde[det].append(spectral_kde)

            split_indices = [0, len(x_positions)]
            split_indices += [
                get_index_for_elem(x_positions, f) + 1
                for f in self.settings["split_frequencies"]
            ]
            split_indices = sorted(split_indices)

            for i in range(len(split_indices) - 1):
                vals = y_values[:, split_indices[i]: split_indices[i + 1]].T
                kde_vals = stats.gaussian_kde(
                    vals, bw_method=float(self.settings["bandwidth_spline"])
                )
                self.broadband_kde[det].append(kde_vals)

    def sample(self, num_samples, rescaling_ys=None):
        """
        Sample a synthetic ASD dataset from the fitted KDEs
        ----------
        Parameters:
        num_samples (int): Number of samples to draw.
        rescaling_ys (dict): Optional dictionary of spline y-values used for rescaling the base noise.
        """

        parameters_dicts = copy.deepcopy(self.parameter_dicts)
        asds_dict = {}

        for det, param_dict in self.parameter_dicts.items():
            asds_dict[det] = []

            features = np.zeros((num_samples, len(self.spectral_kde[det]), 3))
            for i, kde in enumerate(self.spectral_kde[det]):
                features[:, i, :] = kde.resample(size=num_samples).T

            parameters_dicts[det]["spectral_features"] = features

            y_values = np.zeros((num_samples, 0))

            for i, kde in enumerate(self.broadband_kde[det]):
                y_values = np.concatenate(
                    (y_values, kde.resample(size=num_samples).T), axis=1
                )
            # rescale base noise
            if rescaling_ys:
                y_values_mean = np.mean(y_values, axis=0)
                y_values = (
                    y_values - y_values_mean[None, :] + rescaling_ys[det]
                )
            parameters_dicts[det]["y_values"] = y_values

        return parameters_dicts


def get_rescaling_params(filenames, parameterization_settings):
    """
    Get the parameters of the ASDs that are used for rescaling.
    Parameters
    ----------
    filenames : dict
        Dictionary containing the paths to the ASD files.
    parameterization_settings : dict
        Dictionary containing the settings for the parameterization.
    """
    parameters = {}
    for det, asd_path in filenames.items():
        asd_dataset = ASDDataset(asd_path)
        psd = asd_dataset.asds[det][0] ** 2

        # we only need the parameterized broadband noise
        _, ys = fit_broadband_noise(
            domain=asd_dataset.domain,
            psd=np.log(psd),
            num_spline_positions=parameterization_settings["num_spline_positions"],
            sigma=float(parameterization_settings["sigma"]),
        )
        parameters[det] = ys

    return parameters
