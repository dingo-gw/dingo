from typing import Optional
import numpy as np

from dingo.gw.domains import MultibandedFrequencyDomain, UniformFrequencyDomain
from gw.domains import multibanded_frequency_domain


class DecimateAll(object):
    """Transform operator for decimation to multibanded frequency domain."""

    def __init__(
        self,
        multibanded_frequency_domain: MultibandedFrequencyDomain,
    ):
        """
        Parameters
        ----------
        multibanded_frequency_domain: MultibandedFrequencyDomain
            New domain of the decimated data. Original data must be in
            multibanded_frequency_domain.base_domain
        """
        self.multibanded_frequency_domain = multibanded_frequency_domain

    def __call__(self, input_sample: dict) -> dict:
        """
        Parameters
        ----------
        input_sample : dict
            For each decimation_key in self.decimation_keys, sample[decimation_key]
            should be (1) a dict with arrays containing data to be transformed,
            or (2) an array with data to be transformed.

        Returns
        -------
        dict of the same form as the input, but with transformed (decimated) data.
        """
        sample = input_sample.copy()
        decimate_recursive(sample, self.multibanded_frequency_domain)
        return sample


def decimate_recursive(d: dict, mfd: MultibandedFrequencyDomain):
    """
    In-place decimation of nested dicts of arrays.

    Parameters
    ----------
    d : dict
        Nested dictionary to decimate.
    mfd : MultibandedFrequencyDomain
    """
    for k, v in d.items():
        if isinstance(v, dict):
            decimate_recursive(v, mfd)
        elif isinstance(v, np.ndarray):
            if v.shape[-1] == len(mfd.base_domain):
                d[k] = mfd.decimate(v)
        else:
            raise ValueError(f"Cannot decimate item of type {type(v)}.")


class DecimateWaveformsAndASDS(object):
    """Transform operator for decimation of unwhitened waveforms and corresponding ASDS
    to multibanded frequency domain (MFD).


    For decimation, we have two options.


    1) decimation_mode = whitened
    In this case, the GW data is whitened first,

            dw = d / ASD,

    and then decimated to the MFD,

            dw_mfd = decimate(dw).

    In this case, the effective ASD in the MFD is given by

            ASD_mfd = 1 / decimate(1 / ASD).

    See [1] for details. ASD_mfd can then be provided to the inference network.


    2) decimation_mode = unwhitened
    In this case, the GW data is first decimated,

            d_mfd = decimate(d)

    and then whitened.

            dw_mfd = d_mfd / ASD_mfd.

    In this case, the ASD_mfd that whitens the data d_mfd is given by

            ASD_mfd = decimate(ASD ** 2) ** 0.5.

    In other words, in this case we need to decimate the *PSD*. See [1] for details.


    Method 1) better preserves the signal.

    [1] https://github.com/dingo-gw/dingo/blob/fede5c01524f3e205acf5750c0a0f101ff17e331/binary_neutron_stars/prototyping/psd_decimation.ipynb
    """

    def __init__(
        self,
        multibanded_frequency_domain: MultibandedFrequencyDomain,
        decimation_mode: str,
    ):
        """
        Parameters
        ----------
        multibanded_frequency_domain: MultibandedFrequencyDomain
            New domain of the decimated data. Original data must be in
            multibanded_frequency_domain.base_domain
        decimation_mode: str
            One of ["whitened", "unwhitened"]. Determines whether decimation is
            performed on whitened data or on unwhitened data. See class docstring for
            details.
        """
        self.multibanded_frequency_domain = multibanded_frequency_domain
        if decimation_mode not in ["whitened", "unwhitened"]:
            raise ValueError(
                f"Unsupported decimation mode {decimation_mode}, needs to be one of "
                f'["whitened", "unwhitened"].'
            )
        self.decimation_mode = decimation_mode

    def __call__(self, input_sample: dict) -> dict:
        """
        Parameters
        ----------
        input_sample : dict
            Values of sample["waveform"] should be arrays containing waveforms to be
            transformed, Values of sample["asds"] should be arrays containing asds
            to be transformed.

        Returns
        -------
        dict of the same form as the input, but with transformed (decimated) waveforms
        and asds.
        """
        sample = input_sample.copy()

        # Only decimate the data if it is in the base domain. If it has already been
        # decimated, do not change it.

        if check_sample_in_domain(
            sample, self.multibanded_frequency_domain.base_domain
        ):
            if self.decimation_mode == "whitened":
                whitened_waveforms = {
                    k: v / sample["asds"][k] for k, v in sample["waveform"].items()
                }
                whitened_waveforms_dec = {
                    k: self.multibanded_frequency_domain.decimate(v)
                    for k, v in whitened_waveforms.items()
                }
                asds_dec = {
                    k: 1 / self.multibanded_frequency_domain.decimate(1 / v)
                    for k, v in sample["asds"].items()
                }
                # color the whitened waveforms with the effective asd
                waveform_dec = {
                    k: v * asds_dec[k] for k, v in whitened_waveforms_dec.items()
                }
                sample["waveform"] = waveform_dec
                sample["asds"] = asds_dec

            elif self.decimation_mode == "unwhitened":
                sample["waveform"] = {
                    k: self.multibanded_frequency_domain.decimate(v)
                    for k, v in sample["waveform"].items()
                }
                sample["asds"] = {
                    k: self.multibanded_frequency_domain.decimate(v**2) ** 0.5
                    for k, v in sample["asds"].items()
                }

            else:
                raise NotImplementedError()

        return sample


class MaskDataForFrequencyRangeUpdate(object):
    """
    Set waveform to zero and ASD to one according to minimum_frequency, maximum_frequency, and suppress_range.
    """

    def __init__(
        self,
        domain: UniformFrequencyDomain | MultibandedFrequencyDomain,
        minimum_frequency: Optional[float | dict[str, float]] = None,
        maximum_frequency: Optional[float | dict[str, float]] = None,
        suppress_range: Optional[list[float, float] | dict[str, list[float]]] = None,
        ifos: Optional[list[str]] = None,
        print_output: bool = False,
    ):
        """
        Parameters
        ----------
        minimum_frequency: Optional[float | dict[str, float]]
            Update of f_min, if float, the same value will be used for all detectors.
        maximum_frequency: Optional[float | dict[str, float]]
            Update of f_max, if float, the same value will be used for all detectors.
        suppress_range: list[float, float] | dict[str, list[float, float]] | None
            Suppress ranges [f_min, f_max], either for all detectors or for individual detectors. Frequency bins
            corresponding to f_min and f_max are excluded.
        ifos: list[str]
            List of detectors
        print_output: bool
            Whether to write print statements to the console.
        """
        self.sample_frequencies = domain.sample_frequencies
        self.frequency_mask = domain.frequency_mask
        self.minimum_frequency = minimum_frequency
        self.maximum_frequency = maximum_frequency
        self.suppress_range = suppress_range
        # Check that minimum-/maximum frequency is provided for each detector
        if isinstance(minimum_frequency, dict) and ifos is not None:
            ifos_min = [i for i in minimum_frequency.keys()]
            if not set(ifos).issubset(ifos_min):
                raise ValueError(
                    f"minimum-frequency={minimum_frequency} doesn't contain information about all "
                    f"detectors present in event: {ifos}."
                )
        if isinstance(maximum_frequency, dict) and ifos is not None:
            ifos_max = [i for i in maximum_frequency.keys()]
            if not set(ifos).issubset(ifos_max):
                raise ValueError(
                    f"maximum-frequency={maximum_frequency} doesn't contain information about all "
                    f"detectors present in event: {ifos}."
                )

        if print_output:
            print(
                f"Transform MaskDataForFrequencyRangeUpdate activated:"
                f"  Settings: \n"
                f"    - Minimum_frequency update: {self.minimum_frequency}\n"
                f"    - Maximum_frequency update: {self.maximum_frequency}\n"
                f"    - Suppress range: {self.suppress_range}\n"
            )

    def __call__(self, input_sample: dict) -> dict:
        """
        Parameters
        ----------
        input_sample: Dict
            Values for keys
            - 'waveform', shape [batch_size, num_frequency_bins]
            - 'asds', shape [batch_size, num_frequency_bins]

        Returns
        ----------
        sample: Dict
            input_sample with modified value for key
            - 'waveform'
            - 'asds'

        """
        sample = input_sample.copy()
        # Get frequency masks for full domain
        frequency_masks = create_mask_based_on_frequency_update(
            sample_frequencies=self.sample_frequencies,
            detectors=[d for d in sample["waveform"].keys()],
            minimum_frequency=self.minimum_frequency,
            maximum_frequency=self.maximum_frequency,
            suppress_range=self.suppress_range,
        )

        # Update waveforms and ASDs based on masks
        for d in sample["waveform"].keys():
            # Combine with frequency mask from domain
            combined_mask = np.where(self.frequency_mask, ~frequency_masks[d], False)
            # (1) Zero waveform
            sample["waveform"][d][..., combined_mask] = 0.0
            # (2) Set ASD to 1.
            sample["asds"][d][..., combined_mask] = 1.0

        return sample


def create_mask_based_on_frequency_update(
    sample_frequencies: np.array,
    detectors: list[str],
    minimum_frequency: Optional[float | dict[str, float]] = None,
    maximum_frequency: Optional[float | dict[str, float]] = None,
    suppress_range: Optional[list[float, float] | dict[str, list[float, float]]] = None,
) -> dict[str : np.array]:
    """
    Parameters
    ----------
    sample_frequencies: np.array
        Frequency values for which we want to create a mask. Assumed to be the same for all detectors.
    detectors: list[str]
        List of detector names.
    minimum_frequency: Optional[float | dict[str, float]]
        Update of f_min, if float, the same value will be used for all detectors.
    maximum_frequency: Optional[float | dict[str, float]]
        Update of f_max, if float, the same value will be used for all detectors.
    suppress_range: list[float, float] | dict[str, list[float, float]] | None
        Suppress ranges [f_min, f_max], either for all detectors or for individual detectors. Frequency bins
        corresponding to f_min and f_max are excluded.

    Returns
    ----------
    frequency_masks: dict
        Dictionary providing a boolean mask for the sample frequencies of each detector based on provided frequency
        updates. True for values to keep, False for values to mask.
    """
    # We only modify the valid frequency values and assume that values corresponding to frequency_mask = False have
    # already been adjusted
    frequency_masks = {
        i: np.ones_like(sample_frequencies, dtype=bool) for i in detectors
    }
    for d in detectors:
        # Update frequency_masks based on minimum_frequency
        if minimum_frequency is not None:
            # Same for all detectors
            if isinstance(minimum_frequency, float):
                mask_min = sample_frequencies >= minimum_frequency
            # Different for each detector
            elif isinstance(minimum_frequency, dict):
                mask_min = sample_frequencies >= minimum_frequency[d]
            else:
                raise ValueError(
                    f"minimum-frequency has to be dict or float, not {type(minimum_frequency)}. "
                )
            frequency_masks[d] = np.logical_and(frequency_masks[d], mask_min)

        # Update frequency_masks based on maximum_frequency
        if maximum_frequency is not None:
            # Same for all detectors
            if isinstance(maximum_frequency, float):
                mask_max = sample_frequencies <= maximum_frequency
            # Different for each detector
            elif isinstance(maximum_frequency, dict):
                mask_max = sample_frequencies <= maximum_frequency[d]
            else:
                raise ValueError(
                    f"maximum-frequency has to be dict or float, not {type(maximum_frequency)}. "
                )
            frequency_masks[d] = np.logical_and(frequency_masks[d], mask_max)

        # Update frequency_masks based on suppress_range
        if suppress_range is not None:
            # Same for all detectors
            if isinstance(suppress_range, list):
                f_min_lower, f_max_upper = suppress_range
                mask_lower = sample_frequencies >= f_min_lower
                mask_upper = sample_frequencies <= f_max_upper
                mask_interval = ~np.logical_and(mask_lower, mask_upper)
                frequency_masks[d] = np.logical_and(frequency_masks[d], mask_interval)
            # Different for each detector
            elif isinstance(suppress_range, dict) and d in suppress_range:
                f_min_lower, f_max_upper = suppress_range[d]
                mask_lower = sample_frequencies >= f_min_lower
                mask_upper = sample_frequencies <= f_max_upper
                mask_interval = ~np.logical_and(mask_lower, mask_upper)
                frequency_masks[d] = np.logical_and(frequency_masks[d], mask_interval)

    return frequency_masks


def check_sample_in_domain(sample, domain: UniformFrequencyDomain) -> bool:
    lengths = []
    base_domain_length = len(domain)
    for k in ["waveform", "asds"]:
        lengths += [d.shape[-1] for d in sample[k].values()]
    if all(l == base_domain_length for l in lengths):
        return True
    else:
        return False
