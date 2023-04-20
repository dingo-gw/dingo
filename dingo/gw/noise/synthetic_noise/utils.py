import numpy as np
import scipy


def get_index_for_elem(arr, elem):
    return (np.abs(arr - elem)).argmin()


def lorentzian_eval(x, f0, A, Q, delta_f=None):
    """
    Evaluates a Lorentzian function at the given frequencies.
    Parameters
    ----------
    x: array_like
        Frequencies at which the Lorentzian is evaluated.
    f0: float
        Center frequency of the Lorentzian.
    A: float
        Amplitude of the Lorentzian.
    Q: float
        Parameter determining the width of the Lorentzian
    delta_f: float, optional
        If given, the Lorentzian is truncated
    Returns
    -------
    array_like
    """
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
    parameters_dict, domain, parameterization_settings
):
    """
    Reconstructs the PSDs from the parameters.
    Parameters
    ----------
    parameters_dict : dict
        Dictionary containing the parameters of the PSDs.
    domain : dingo.gw.noise.domain.Domain
        Domain object containing the frequencies at which the PSDs are evaluated.
    parameterization_settings : dict
        Dictionary containing the settings for the parameterization.
    Returns
    -------
    array_like

    """
    smoothen = parameterization_settings.get("smoothen", False)

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
