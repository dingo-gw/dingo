import numpy as np

from dingo.gw.domains import FrequencyDomain


class StrainTokenization(object):
    """
    Divide frequency bins into frequency segments of equal length and add frequency information and
    encoding of blocks (i.e. interferometers in GW use case) to sample. It is assumed that f_min
    and f_max are the same for all blocks, that all waveforms contain the same number of blocks
    and that the ordering of the blocks within 'waveform' is fixed.
    """

    def __init__(
        self,
        num_tokens: int,
        domain: FrequencyDomain,
        normalize_frequency: bool = False,
    ):
        """
        Parameters
        ----------
        num_tokens: int
            Number of tokens into which the frequency bins should be divided.
        domain: FrequencyDomain
            Contains information such as f_min, f_max, df, etc.
        normalize_frequency: bool
            Whether to normalize the frequency bins for the positional encoding

        """
        num_f = domain.frequency_mask_length
        self.num_bins_per_token = np.ceil(num_f / num_tokens).astype(int)
        f_token_width = self.num_bins_per_token * domain.delta_f
        self.token_indices = np.arange(
            0,
            num_tokens * self.num_bins_per_token,
            self.num_bins_per_token,
            dtype=int,
        )
        assert num_tokens == len(self.token_indices)
        self.f_min_per_token = np.arange(domain.f_min, domain.f_max, f_token_width)
        self.f_max_per_token = self.f_min_per_token + f_token_width - domain.delta_f
        self.num_padded_f_bins = int(num_tokens * self.num_bins_per_token - num_f)
        self.normalize_freq = normalize_frequency
        self.f_min = domain.f_min
        self.f_max = self.f_max_per_token.max()
        self.num_tokens = num_tokens

    def __call__(self, input_sample):
        """
        Parameters
        ----------
        input_sample: Dict
            Value for key 'waveform':
            Sample of shape [num_blocks, num_channels, num_bins]
            where num_blocks = number of detectors in GW use case,
            num_channels>=3 (real, imag, auxiliary channels, e.g. asd),
            and num_bins=number of frequency bins.

        Returns
        ----------
        sample: Dict
            input_sample with modified value for key
            - 'waveform', shape [num_blocks, num_channels, num_tokens, num_bins_per_token]
            and additional keys
            - 'position', shape [num_blocks]
            - 'f_min_per_token', shape [num_tokens]
            - 'f_max_per_token', shape [num_tokens]
        """
        sample = input_sample.copy()

        # pad last dimension
        strain = np.pad(
            sample["waveform"],
            ((0, 0), (0, 0), (0, self.num_padded_f_bins)),
            "constant",
        )
        num_blocks, num_channels = strain.shape[0], strain.shape[1]
        strain = strain.reshape(
            num_blocks, num_channels, self.num_tokens, self.num_bins_per_token
        )
        strain = np.moveaxis(strain, 2, 0)
        strain = strain.reshape(
            self.num_tokens, num_blocks, num_channels * self.num_bins_per_token
        )

        sample["waveform"] = strain
        detector_dict = {"H1": 0, "L1": 1, "V1": 2}
        detectors = np.array([detector_dict[key] for key in input_sample["asds"]])
        if self.normalize_freq:
            f_min_per_token = (self.f_min_per_token - self.f_min) / (
                self.f_max - self.f_min
            )
            f_max_per_token = (self.f_max_per_token - self.f_min) / (
                self.f_max - self.f_min
            )
        else:
            f_min_per_token = self.f_min_per_token
            f_max_per_token = self.f_max_per_token

        token_position = np.empty((strain.shape[0], len(detectors), 2))
        token_position[..., 0] = np.repeat(
            f_min_per_token[..., None], len(detectors), axis=1
        )
        token_position[..., 1] = np.repeat(
            f_max_per_token[..., None], len(detectors), axis=1
        )
        # token_position[..., 2] = np.tile(detectors, self.num_tokens)
        sample["position"] = token_position
        sample["blocks"] = detectors

        return sample
