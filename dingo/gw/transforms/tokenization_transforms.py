import numpy as np

from dingo.gw.domains import FrequencyDomain


class StrainTokenization(object):
    """
    Divide frequency bins into frequency segments of equal bin-length and add frequency
    information and encoding of blocks (i.e. interferometers in GW use case) to sample.
    It is assumed that f_min and f_max are the same for all blocks, that all waveforms
    contain the same number of blocks and that the ordering of the blocks within
    'waveform' is fixed.
    """

    def __init__(
        self,
        domain: FrequencyDomain,
        num_tokens: int = None,
        token_size: int = None,
        normalize_frequency: bool = False,
        single_tokenizer: bool = False,
        print_bool: bool = True,
    ):
        """
        Parameters
        ----------
        domain: FrequencyDomain
            Contains domain information, e.g., f_min, f_max, delta_f. Works with
            FrequencyDomain and MultibandedFrequencyDomain.
        num_tokens: int
            Number of tokens into which the domain should be divided. [Optional]
        token_size: int
            Number of frequency bins per token. It is necessary to specify one of
            num_tokens or token_size. [Optional]
        normalize_frequency: bool
            Whether to normalize the frequency bins for the positional encoding.
            [Default: False]
        single_tokenizer: bool
            Whether to use the StrainTokenization implementation designed for a
            single tokenizer
        print_bool: bool
            Whether to write print statements to the console.
        """
        if num_tokens is not None and token_size is not None:
            raise ValueError("Cannot specify both num_tokens and token_size.")

        num_f = domain.frequency_mask_length
        if num_tokens is not None:
            self.num_bins_per_token = np.ceil(num_f / num_tokens).astype(int)
        elif token_size is not None:
            self.num_bins_per_token = token_size
            num_tokens = np.ceil(num_f / self.num_bins_per_token).astype(int)
        else:
            raise ValueError(
                "It is necessary to specify either num_tokens or token_size."
            )
        if type(domain.delta_f) is float:
            delta_fs = np.array([domain.delta_f] * num_tokens)
        elif type(domain.delta_f) is np.ndarray:
            delta_fs = domain.delta_f
        else:
            raise ValueError(f"domain.delta_f must be either float or np.ndarray, but is {type(domain.delta_f)}.")

        self.f_min_per_token = domain.sample_frequencies[domain.min_idx :: self.num_bins_per_token][:num_tokens]
        self.f_max_per_token = domain.sample_frequencies[
                               domain.min_idx + self.num_bins_per_token - 1 :: self.num_bins_per_token
                               ][:num_tokens]
        if len(self.f_min_per_token) != len(self.f_max_per_token):
            # Extrapolate last band
            f_token_widths = self.num_bins_per_token * delta_fs
            f_max_pad = self.f_max_per_token[-1] + f_token_widths[-1]
            self.f_max_per_token = np.append(self.f_max_per_token, f_max_pad)
            self.num_padded_f_bins = int((f_max_pad - domain.sample_frequencies[-1]) / delta_fs[-1])
        else:
            self.num_padded_f_bins = 0
        assert num_tokens == len(self.f_min_per_token) == len(self.f_max_per_token)
        self.normalize_freq = normalize_frequency
        self.single_tokenizer = single_tokenizer
        self.f_min = domain.f_min
        self.f_max = self.f_max_per_token[-1]
        self.num_tokens = num_tokens

        if print_bool:
            print(
                f"Tokenization:\n"
                f"  Token width {self.num_bins_per_token} frequency bins; {num_tokens} "
                f"tokens per detector\n"
                f"  First token width {self.f_min_per_token[1] - self.f_min_per_token[0]} "
                f"Hz\n"
                f"  Last token width {self.f_min_per_token[-1] - self.f_min_per_token[-2]} "
                f"Hz\n"
                f"  Extrapolating to maximum frequency of {self.f_max_per_token[-1]} Hz"
            )

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
            - 'waveform', shape [num_tokens, num_blocks, num_channels * num_bins_per_token]
            and additional keys
            - 'position', shape [num_tokens, num_blocks, 2]
            - 'blocks', shape [num_blocks]
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
        if self.single_tokenizer:
            strain = strain.reshape(
                self.num_tokens * num_blocks, num_channels * self.num_bins_per_token
            )
        else:
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

        if self.single_tokenizer:
            token_position = np.empty((strain.shape[0], 3))
            token_position[:, 0] = np.repeat(f_min_per_token, len(detectors))
            token_position[:, 1] = np.repeat(f_max_per_token, len(detectors))
            token_position[:, 2] = np.tile(detectors, self.num_tokens)
            sample["position"] = token_position
        else:
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
