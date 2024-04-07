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
        domain: FrequencyDomain,
        num_tokens: int = None,
        token_size: int = None,
        normalize_frequency: bool = False,
    ):
        """
        Parameters
        ----------
        domain: FrequencyDomain
            Contains domain information, e.g., f_min, f_max, delta_f
        num_tokens: int
            Number of tokens into which the frequency bins should be divided.
        token_size: int
            Number of frequency bins per token. It is necessary to specify one of
            num_tokens or token_size.
        normalize_frequency: bool
            Whether to normalize the frequency bins for the positional encoding

        """
        if num_tokens is not None and token_size is not None:
            raise ValueError("Cannot specify both num_tokens and token_size.")

        num_f = domain.frequency_mask_length

        if num_tokens is not None:
            # To calculate the token length, we round down and truncate slightly the domain
            # at the upper end. This means we don't have to zero-pad, improving the
            # consistency between tokens. However, we lose some high-frequency information
            # (hopefully not too important).
            self.num_bins_per_token = num_f // num_tokens
        elif token_size is not None:
            self.num_bins_per_token = token_size
            num_tokens = num_f // token_size
        else:
            raise ValueError(
                "It is necessary to specify either num_tokens or " "token_size."
            )
        self.f_min_per_token = domain.sample_frequencies[
            domain.min_idx :: self.num_bins_per_token
        ][:num_tokens]
        self.f_max_per_token = domain.sample_frequencies[
            domain.min_idx + self.num_bins_per_token - 1 :: self.num_bins_per_token
        ][:num_tokens]
        print(
            f"Tokenization:\n"
            f"  Token width {self.num_bins_per_token} frequency bins; {num_tokens} "
            f"tokens total\n"
            f"  First token width {self.f_min_per_token[1] - self.f_min_per_token[0]} "
            f"Hz\n "
            f"  Last token width {self.f_min_per_token[-1] - self.f_min_per_token[-2]} "
            f"Hz\n"
            f"  Truncating at maximum frequency of {self.f_max_per_token[-1]} Hz"
        )
        self.total_frequency_bins = num_tokens * self.num_bins_per_token
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
            - 'blocks', shape [num_blocks]
            - 'f_min_per_token', shape [num_tokens]
            - 'f_max_per_token', shape [num_tokens]
        """
        sample = input_sample.copy()

        # Truncate
        strain = sample["waveform"][..., : self.total_frequency_bins]

        # pad last dimension
        # strain = np.pad(
        #     sample["waveform"],
        #     ((0, 0), (0, 0), (0, self.num_padded_f_bins)),
        #     "constant",
        # )
        strain = strain.reshape(
            strain.shape[0], strain.shape[1], self.num_tokens, self.num_bins_per_token
        )  # blocks, channels, seq, features
        num_blocks = strain.shape[0]
        num_channels = strain.shape[1]
        strain = np.moveaxis(strain, 2, 0)  # seq, blocks, channels, features
        strain = strain.reshape(
            self.num_tokens * num_blocks, num_channels * self.num_bins_per_token
        )  # seq, features

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

        token_position = np.empty((strain.shape[0], 3))
        token_position[:, 0] = np.repeat(f_min_per_token, len(detectors))
        token_position[:, 1] = np.repeat(f_max_per_token, len(detectors))
        token_position[:, 2] = np.tile(detectors, self.num_tokens)
        sample["position"] = token_position

        return sample
