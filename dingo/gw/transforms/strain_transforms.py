import numpy as np
from typing import Optional, Union

from dingo.gw.domains import FrequencyDomain


class CropMaskStrainRandom(object):
    """Apply random cropping of strain, by masking waveform and ASD outside the crop."""

    def __init__(
        self,
        domain: Union[FrequencyDomain],
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
        domain: Union[FrequencyDomain]
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
            [f_min_lower, f_max_lower]. This is used for inference.
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

        # # initialize functions to sample the
        # if f_max_lower is not None:
        #     idx_bound_f_max = np.argmin(np.abs(f_max_lower - frequencies))
        #     self.sample_idx_upper = lambda s: np.random.randint(
        #         idx_bound_f_max, self.len_domain, s
        #     )
        #     if deterministic:
        #         self.sample_idx_upper = lambda s: np.ones(s) * idx_bound_f_max
        # else:
        #     self.sample_idx_upper = lambda s: np.ones(s) * self.len_domain
        # if f_min_upper is not None:
        #     idx_bound_f_min = np.argmin(np.abs(f_min_upper - frequencies))
        #     self.sample_idx_lower = lambda s: np.random.randint(0, idx_bound_f_min, s)
        #     if deterministic:
        #         self.sample_idx_lower = lambda s: np.ones(s) * idx_bound_f_min
        # else:
        #     self.sample_idx_lower = lambda s: np.zeros(s)

    def sample_upper_bound_indices(self, shape):
        """Sample indices for upper crop boundaries."""
        if self._deterministic:
            return np.ones(shape) * self._idx_bound_f_max
        else:
            return np.random.randint(self._idx_bound_f_max, self.len_domain, shape)

    def sample_lower_bound_indices(self, shape):
        """Sample indices for lower crop boundaries."""
        if self._deterministic:
            return np.ones(shape) * self._idx_bound_f_min
        else:
            # self._idx_bound_f_min is inclusive bound, so need to add 1
            return np.random.randint(0, self._idx_bound_f_min + 1, shape)

    def check_inputs(
        self, domain, f_min_upper, f_max_lower, cropping_probability, deterministic
    ):
        # check domain
        if not isinstance(domain, (FrequencyDomain,)):
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

    def __call__(self, input_sample: dict):
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

        # The strain has shape (B, D, C, N) or (D, C, N) for non-batched data.
        #   B: batch_size
        #   D: num_detectors
        #   C: num_channels, typically 3 for (strain.real, strain.imag, asd)
        #   N: frequency bins, self.len_domain
        #
        # We crop the strain by masking: strain[..., :lower] = 0, strain[..., upper:] = 0.
        # - Cropping/masking always uses the same boundary indices (lower, upper) along
        #   the channel dimension.
        # - If self.independent_detectors = True, (lower, upper) is sampled
        #   independently for the different detectors.
        # - Cropping is only applied to a fraction of the data, specified by
        #   self.cropping_probability.
        # - If self.independent_lower_upper = True,

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
