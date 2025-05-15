from typing import Optional
import numpy as np

from dingo.gw.domains import MultibandedFrequencyDomain, UniformFrequencyDomain


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


def check_sample_in_domain(sample, domain: UniformFrequencyDomain) -> bool:
    lengths = []
    base_domain_length = len(domain)
    for k in ["waveform", "asds"]:
        lengths += [d.shape[-1] for d in sample[k].values()]
    if all(l == base_domain_length for l in lengths):
        return True
    else:
        return False


class CropMaskStrainRandom(object):
    """Apply random cropping of strain, by masking waveform and ASD outside the crop."""

    def __init__(
        self,
        domain: UniformFrequencyDomain | MultibandedFrequencyDomain,
        f_min_upper: Optional[float] = None,
        f_max_lower: Optional[float] = None,
        deterministic: bool = False,
        cropping_probability: float = 1.0,
        independent_detectors: bool = True,
        independent_lower_upper: bool = True,
    ):
        """
        Parameters
        ----------
        domain: UniformFrequencyDomain | MultibandedFrequencyDomain
            Domain of the waveform data, has to be a frequency domain type.
        f_min_upper: float
            New f_min is sampled in range [domain.f_min, f_min_upper].
            Sampling of f_min is uniform in bins (not in frequency) when the frequency
            domain is not uniform (e.g., MultibandedFrequencyDomain).
        f_max_lower: float
            New f_max is sampled in range [domain.f_max, f_max_lower].
            Sampling of f_max is uniform in bins (not in frequency) when the frequency
            domain is not uniform (e.g., MultibandedFrequencyDomain).
        deterministic: bool
            If True, don't sample truncation range, but instead always truncate to range
            [f_min_upper, f_max_lower]. This is used for inference.
        cropping_probability: float
            probability for a given sample to be cropped
        independent_detectors: bool
            If True, crop boundaries are sampled independently for different detectors.
        independent_lower_upper: bool
            If True, the cropping probability is applied to lower and upper boundaries
            individually. If False, then with a probability of P = cropping_probability
            both lower and upper cropping is applied, and with 1-P, no cropping is
            applied from either direction.
        """
        self.check_inputs(
            domain, f_min_upper, f_max_lower, cropping_probability, deterministic
        )
        self._deterministic = deterministic
        self.cropping_probability = cropping_probability
        self.independent_detectors = independent_detectors
        self.independent_lower_upper = independent_lower_upper
        frequencies = domain()[domain.min_idx :]
        self.len_domain = len(frequencies)

        if f_max_lower is not None:
            self._idx_bound_f_max = np.argmin(np.abs(f_max_lower - frequencies))
        else:
            self._idx_bound_f_max = self.len_domain - 1

        if f_min_upper is not None:
            self._idx_bound_f_min = np.argmin(np.abs(f_min_upper - frequencies))
        else:
            self._idx_bound_f_min = 0

    def sample_upper_bound_indices(self, shape: list[int]) -> np.ndarray:
        """Sample indices for upper crop boundaries."""
        if self._deterministic:
            return np.ones(shape) * self._idx_bound_f_max
        else:
            return np.random.randint(self._idx_bound_f_max, self.len_domain, shape)

    def sample_lower_bound_indices(self, shape: list[int]) -> np.ndarray:
        """Sample indices for lower crop boundaries."""
        if self._deterministic:
            return np.ones(shape) * self._idx_bound_f_min
        else:
            # self._idx_bound_f_min is inclusive bound, so need to add 1
            return np.random.randint(0, self._idx_bound_f_min + 1, shape)

    def check_inputs(
        self,
        domain: UniformFrequencyDomain | MultibandedFrequencyDomain,
        f_min_upper: float,
        f_max_lower: float,
        cropping_probability: float,
        deterministic: bool,
    ):
        # check domain
        if not isinstance(domain, (UniformFrequencyDomain, MultibandedFrequencyDomain)):
            raise ValueError(
                f"Domain should be a frequency domain type, got {type(domain)}."
            )
        # check validity of ranges
        if f_min_upper is not None:
            if not domain.f_min < f_min_upper < domain.f_max:
                raise ValueError(
                    f"Expected f_min_upper in domain range [{domain.f_min},"
                    f" {domain.f_max}], got {f_min_upper}."
                )
        if f_max_lower is not None:
            if not domain.f_min < f_max_lower < domain.f_max:
                raise ValueError(
                    f"Expected f_max_lower in domain range [{domain.f_min},"
                    f" {domain.f_max}], got {f_max_lower}."
                )
        if f_min_upper and f_max_lower and f_min_upper >= f_max_lower:
            raise ValueError(
                f"Expected f_min_upper < f_max_lower, got {f_min_upper}, {f_max_lower}."
            )
        if not 0 <= cropping_probability <= 1.0:
            raise ValueError(
                f"Cropping probability should be in [0, 1], got {cropping_probability}."
            )
        # check that no non-trivial cropping probability is set when deterministic = True
        if deterministic and cropping_probability < 1.0:
            raise ValueError(
                f"cropping_probability must be 1.0 when deterministic = True, got "
                f"{cropping_probability}."
            )

    def __call__(self, input_sample: dict) -> dict:
        """
        Parameters
        ----------
        input_sample : dict
            sample["waveform"]: Dict with values being arrays containing waveforms,
            or torch Tensor with the waveform.

        Returns
        -------
        dict of the same form as the input, but with transformed (crop-masked) waveforms.
        """
        sample = input_sample.copy()
        strain = sample["waveform"]
        if strain.shape[-1] != self.len_domain:
            raise ValueError(
                f"Expected waveform input of shape [..., {self.len_domain}], "
                f"got {strain.shape}."
            )

        # Sample boundary indices for crops
        constant_ax = 3 - self.independent_detectors
        lower = self.sample_lower_bound_indices(strain.shape[:-constant_ax])
        upper = self.sample_upper_bound_indices(strain.shape[:-constant_ax])

        # Only apply crops to a fraction of self.cropping_probability
        if self.cropping_probability < 1:
            mask = np.random.uniform(size=lower.shape) <= self.cropping_probability
            lower = np.where(mask, lower, 0)
            if self.independent_lower_upper:
                mask = np.random.uniform(size=lower.shape) <= self.cropping_probability
            upper = np.where(mask, upper, self.len_domain)

        # Broadcast boundaries and apply cropping
        mask_lower = np.arange(self.len_domain) >= lower[(...,) + (None,) * constant_ax]
        mask_upper = np.arange(self.len_domain) <= upper[(...,) + (None,) * constant_ax]
        strain = np.where(mask_lower, strain, 0)
        strain = np.where(mask_upper, strain, 0)
        sample["waveform"] = strain

        return sample


class MaskDataForFrequencyRangeUpdate(object):
    """
    Set waveform to zero and ASD to one according to minimum_frequency and maximum_frequency.
    """

    def __init__(
        self,
        domain: UniformFrequencyDomain | MultibandedFrequencyDomain,
        minimum_frequency: Optional[float | dict[str, float]] = None,
        maximum_frequency: Optional[float | dict[str, float]] = None,
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
        ifos: list[str]
            List of detectors
        print_output: bool
            Whether to write print statements to the console.
        """
        self.sample_frequencies = domain.sample_frequencies
        self.frequency_mask = domain.frequency_mask
        self.minimum_frequency = minimum_frequency
        self.maximum_frequency = maximum_frequency
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
            )

    def __call__(self, input_sample: dict) -> dict:
        """
        Parameters
        ----------
        input_sample: Dict
            Values for keys
            - 'waveform', dict containing waveform for each detector
            - 'asds', dict containing asds for each detector

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
        )

        # Update waveforms and ASDs based on masks
        for d in sample["waveform"].keys():
            # Exclude frequency values where domain.frequency_mask = False because we assume that these values
            # have already been adjusted.
            combined_mask = np.where(self.frequency_mask, ~frequency_masks[d], False)
            # (1) Set waveform to 0.
            sample["waveform"][d][..., combined_mask] = 0.0
            # (2) Set ASD to 1.
            sample["asds"][d][..., combined_mask] = 1.0

        return sample


def create_mask_based_on_frequency_update(
    sample_frequencies: np.array,
    detectors: list[str],
    minimum_frequency: Optional[float | dict[str, float]] = None,
    maximum_frequency: Optional[float | dict[str, float]] = None,
) -> dict[str : np.array]:
    """
    Creates a mask for each detector containing True for sample_frequencies not affected by the frequency updates
    for minimum and maximum frequencies.

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

    return frequency_masks
