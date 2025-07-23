from typing import Optional, Tuple
import numpy as np
from scipy.interpolate import interp1d
import torch

from bilby.gw.detector import PowerSpectralDensity
from dingo.gw.dataset.generate_dataset import generate_parameters_and_polarizations
from dingo.gw.domains import (
    build_domain,
    MultibandedFrequencyDomain,
    UniformFrequencyDomain,
)
from dingo.gw.domains.evaluate_multibanded_domain import evaluate_multibanding_main
from dingo.gw.domains.multibanded_frequency_domain import (
    floor_to_power_of_2,
    get_band_nodes_for_adaptive_decimation,
)
from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.waveform_generator import NewInterfaceWaveformGenerator, WaveformGenerator


def compute_wf_difference_per_decimation_factor(
    decimation_factors: list,
    waveforms: np.ndarray,
    ufd: UniformFrequencyDomain,
    waveforms_2x: np.ndarray,
    difference_over_full_window: bool = False,
) -> Tuple[list, list]:
    """
    Compute difference between standard waveform and waveform with 2x resolution for different decimation_factors.

    Parameters
    ----------
    decimation_factors: list
        List of decimation factors to try.
    waveforms: np.ndarray
        Whitened waveforms with the desired resolution (1x).
    ufd: UniformFrequencyDomain
        Domain of waveforms.
    waveforms_2x: np.ndarray
        Waveforms with the double the desired resolution (2x) which serves as the ground truth.
    difference_over_full_window: bool
        Whether to accumulate the differences between the ground truth waveform and the decimated waveform over the
        full window or only compute it at the center of the window.

    Returns
    ----------
    diffs_per_decimation_factor: list
        Differences between the ground truth waveform and the decimated waveform for each decimation factor.
    freqs_per_decimation_factor: list
        Frequency bins corresponding to the decimated waveform for each decimation factor.
    """
    data = waveforms.real
    data_2x = waveforms_2x.real
    freq = ufd.sample_frequencies
    assert freq.shape[0] == data.shape[-1]

    freqs_per_decimation_factor = []
    diffs_per_decimation_factor = []
    for decimation_factor in decimation_factors:
        # Construct convolution kernel
        kernel = (
            torch.ones((1, 1, decimation_factor), dtype=torch.float64)
            / decimation_factor
        )
        conv = lambda d: torch.nn.functional.conv1d(
            torch.tensor(d, dtype=torch.float64)[:, None, :],
            kernel,
            padding=0,
            stride=decimation_factor,
        ).squeeze()
        # Apply conv to data
        data_decimated = conv(data).numpy()

        if difference_over_full_window:
            # Compare the decimated value to all values within the decimation window
            # (most conservative estimate for difference between original and decimated data)
            f_ref = freq
            data_ref = data
            data_decimated = np.repeat(data_decimated, decimation_factor, axis=-1)
            n_pad = data_ref.shape[1] - data_decimated.shape[1]
            data_decimated = np.pad(
                data_decimated, ((0, 0), (0, n_pad)), "constant", constant_values=0
            )
        else:
            # Compare the decimated value only to the waveform value at the frequency value in the middle of each band
            f_ref = conv(freq[None, :])
            inds_ref = (f_ref / (ufd.delta_f / 2)).type(torch.int32)
            data_ref = data_2x[:, inds_ref]

        diff = np.abs(data_ref - data_decimated)
        diff_perc = np.percentile(diff, 95, axis=0)
        diff_perc = np.maximum.accumulate(diff_perc[..., ::-1], axis=-1)[..., ::-1]
        diffs_per_decimation_factor.append(diff_perc)
        freqs_per_decimation_factor.append(f_ref.numpy())

    return diffs_per_decimation_factor, freqs_per_decimation_factor


def compute_max_decimation_factor(
    decimation_factors: list,
    diffs_per_decimation_factor: list,
    freqs_per_decimation_factor: list,
    ufd: UniformFrequencyDomain,
    threshold: float,
) -> np.ndarray:
    """
    Compute the maximal decimation factor for each frequency bin.

    Parameters
    ----------
    decimation_factors: list
        List of decimation factors that were used.
    diffs_per_decimation_factor: list
        Differences between the ground truth waveform and the decimated waveform for each decimation factor.
        Output of compute_wf_difference_per_decimation_factor.
    freqs_per_decimation_factor
        Frequency bins corresponding to the decimated waveform for each decimation factor.
        Output of compute_wf_difference_per_decimation_factor.
    ufd: UniformFrequencyDomain
        Original domain (1x) of the waveforms.
    threshold: float
        Threshold specifying the maximally allowed difference between the ground truth waveform and the
        decimated waveform.

    Returns
    -------
    max_decimation_factor: np.ndarray
        Array of maximally allowed decimation factors for each frequency bin of the UFD.
    """
    # Start from maximal decimation of 1 for full domain
    max_dec_factor = np.ones(len(ufd()))
    f = ufd()

    f_max_threshold = []
    # Loop over decimation factors
    for decimation_factor, f, diff in zip(
        decimation_factors, freqs_per_decimation_factor, diffs_per_decimation_factor
    ):
        # Find frequency above which the difference is smaller than the threshold,
        # i.e., use this decimation factor above f_max
        f_max = f[np.argmax(diff < threshold)]
        if f_max_threshold:
            assert f_max_threshold[-1] <= f_max
        f_max_threshold.append(f_max)
        # Set larger decimation factor above f_max
        max_dec_factor[int(f_max / ufd.delta_f) :] = decimation_factor
    return max_dec_factor


def get_band_nodes_for_adaptive_decimation_transformer(
    max_dec_factor_array: np.ndarray,
    max_dec_factor_global: int = np.inf,
    min_mfd_bins_per_band: int = 1,
) -> Tuple[int, list[int]]:
    """
    Sets up adaptive multibanding for decimation. The 1D array max_dec_factor_array has
    the same length as the original, and contains the maximal acceptable decimation
    factors for each bin. max_dec_factor_global further specifies the maximum
    decimation factor.

    Parameters
    ----------
    max_dec_factor_array: np.ndarray
        Array with maximal decimation factor for each bin. Monotonically increasing.
    max_dec_factor_global: int = np.inf
        Global maximum for decimation factor.

    Returns
    -------
    initial_downsampling: int
        Downsampling factor of band 0.
    band_nodes: list[int]
        List with nodes for bands.
        Band j consists of indices [nodes[j]:nodes[j+1].
    """
    if len(max_dec_factor_array.shape) != 1:
        raise ValueError("max_dec_factor_array needs to be 1D array.")
    if not (max_dec_factor_array[1:] >= max_dec_factor_array[:-1]).all():
        raise ValueError("max_dec_factor_array needs to increase monotonically.")
    max_dec_factor_array = np.clip(max_dec_factor_array, None, max_dec_factor_global)
    N = len(max_dec_factor_array)
    dec_factor = int(max(1, floor_to_power_of_2(max_dec_factor_array[0])))
    band_nodes = [0]
    # Increment by entire token
    upper = dec_factor * min_mfd_bins_per_band
    initial_downsampling = dec_factor
    while upper - 1 < N:
        assert dec_factor <= max_dec_factor_array[upper]
        if upper - 1 + dec_factor * min_mfd_bins_per_band >= N:
            # Conclude while loop, append upper as last node
            band_nodes.append(upper)
        elif dec_factor * 2 <= max_dec_factor_array[upper]:
            # Conclude previous band
            band_nodes.append(upper)
            assert (band_nodes[-1] - band_nodes[-2]) % min_mfd_bins_per_band == 0
            # Enter new band
            dec_factor *= 2
        # Increment by entire token
        upper += dec_factor * min_mfd_bins_per_band

    return initial_downsampling, band_nodes


def compute_multibanding_nodes_via_local_difference(
    settings_wfd: dict,
    max_diff_threshold: float,
    maximal_delta_f_max_from_time_shifts: float,
    asd_file_path: Optional[str] = "aLIGO_ZERO_DET_high_P_asd.txt",
    decimation_factors: Optional[list] = (2 ** np.arange(8)).tolist(),
    num_samples: int = 1000,
    num_processes_wfd_generation: int = 1,
    token_size: int = 16,
) -> dict:
    """
    Parameters
    ----------
    settings_wfd: dict
        Waveform dataset settings with uniform frequency domain.
    max_diff_threshold: float
        Maximally allowed threshold for the difference between the original and decimated waveform.
    maximal_delta_f_max_from_time_shifts: float
        Maximal delta f generated by the largest possible time shift. If we sample from the extremal values of the geocent time prior (e.g. [-0.1, 0.1] ms), we can get time-shifts with a large effect on the original waveform. Therefore, we restrict the maximal size of MFD bands by defining maximal_delta_f_max_from_time_shifts (e.g. =2)
    asd_file_path: Optional[str]
        Path to design sensitivity ASD file.
    decimation_factors: Optional[list]
        List of decimation factors to try.
    num_samples: Optional[int]
        Number of waveforms for which to compute the MFD and the mismatch.
    num_processes_wfd_generation: Optional[int]
        Number of processes to use for generating the waveforms.
    token_size: Optional[int]
        Token size for transformer. This value corresponds to the minimal number of frequency bins per band.
    Returns
    -------
    mfd_domain_dict: dict
        Settings of generated MFD domain
    """

    # Build domains
    ufd = build_domain(settings=settings_wfd["domain"])
    ufd_2x = build_domain(settings={**ufd.domain_dict, **{"delta_f": ufd.delta_f / 2}})

    # Build prior
    prior = build_prior_with_defaults(prior_settings=settings_wfd["intrinsic_prior"])

    # Generate waveforms in a domain with twice the resolution of ufd, such that we can later compare
    # the decimated mfd waveforms to the high-resolution waveform as a reference.
    if settings_wfd["waveform_generator"].get("new_interface", False):
        waveform_generator = NewInterfaceWaveformGenerator(
            domain=ufd_2x,
            **settings_wfd["waveform_generator"],
        )
    else:
        waveform_generator = WaveformGenerator(
            domain=ufd_2x,
            **settings_wfd["waveform_generator"],
        )
    # waveform_generator = NewInterfaceWaveformGenerator(domain=ufd_twice_res, **settings_wfd["waveform_generator"])
    parameters, polarizations_2x = generate_parameters_and_polarizations(
        waveform_generator=waveform_generator,
        prior=prior,
        num_samples=num_samples,
        num_processes=num_processes_wfd_generation,
    )
    # Down-sample polarization to match standard ufd resolution
    polarizations = {k: v[..., ::2] for k, v in polarizations_2x.items()}
    assert polarizations["h_plus"][0].shape == ufd().shape

    # Load ASD
    psd = PowerSpectralDensity(asd_file=asd_file_path)
    asd_interp = interp1d(
        psd.frequency_array, psd.asd_array, bounds_error=False, fill_value=np.inf
    )
    asd_2x = asd_interp(ufd_2x.sample_frequencies)
    # Remove bump around 480 Hz
    asd_2x[np.where((ufd_2x() > 477) & (ufd_2x() < 483))[0]] = asd_2x[
        np.argmin(np.abs(ufd_2x() - 477))
    ]
    # Down-sample ASD to match standard ufd resolution
    asd = asd_2x[::2]

    # Whiten data
    data = {k: v / asd / ufd.noise_std for k, v in polarizations.items()}
    data_2x = {
        k: v / asd_2x / ufd.noise_std for k, v in polarizations_2x.items()
    }  # not ufd_2x.noise_std!

    # Decimate data for different decimation factors and calculate difference
    diffs, freqs = compute_wf_difference_per_decimation_factor(
        decimation_factors=decimation_factors,
        waveforms=data["h_cross"],
        ufd=ufd,
        waveforms_2x=data_2x["h_cross"],
    )
    # Compute maximal decimation factor per ufd frequency bin
    max_dec_factor = compute_max_decimation_factor(
        decimation_factors=decimation_factors,
        diffs_per_decimation_factor=diffs,
        freqs_per_decimation_factor=freqs,
        ufd=ufd,
        threshold=max_diff_threshold,
    )

    # Get band nodes
    if token_size is not None:
        initial_downsampling, band_nodes_indices = (
            get_band_nodes_for_adaptive_decimation_transformer(
                max_dec_factor_array=max_dec_factor[ufd.min_idx :],
                max_dec_factor_global=int(
                    maximal_delta_f_max_from_time_shifts / ufd.delta_f
                ),  # = df * T
                min_mfd_bins_per_band=token_size,
            )
        )
    else:
        initial_downsampling, band_nodes_indices = (
            get_band_nodes_for_adaptive_decimation(
                max_dec_factor_array=max_dec_factor[ufd.min_idx :],
                max_dec_factor_global=int(
                    maximal_delta_f_max_from_time_shifts / ufd.delta_f
                ),  # = df * T
            )
        )
    # Transform downsampling factor and band nodes from indices to frequencies
    delta_f_initial = ufd.delta_f * initial_downsampling
    mfd_nodes = ufd()[ufd.min_idx :][np.array(band_nodes_indices)]
    print("Nodes of multibanded frequency domain:", mfd_nodes)
    # Compute compression factor
    mfd = MultibandedFrequencyDomain(
        nodes=mfd_nodes, delta_f_initial=delta_f_initial, base_domain=ufd
    )
    print("Compression factor:", len(ufd()[ufd.frequency_mask]) / len(mfd()))
    mfd_settings_wfd = settings_wfd.copy()
    mfd_settings_wfd["domain"] = mfd.domain_dict
    # Compute mismatch between ufd and mfd
    evaluate_multibanding_main(num_samples=num_samples, settings=mfd_settings_wfd)

    return mfd.domain_dict
