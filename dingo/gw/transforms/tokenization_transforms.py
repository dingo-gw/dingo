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
        num_tokens_per_detector: int = None,
        token_size: int = None,
        normalize_frequency: bool = False,
        single_tokenizer: bool = False,
        print_output: bool = True,
    ):
        """
        Parameters
        ----------
        domain: FrequencyDomain
            Contains domain information, e.g., f_min, f_max, delta_f. Works with
            FrequencyDomain and MultibandedFrequencyDomain.
        num_tokens_per_detector: int
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
        print_output: bool
            Whether to write print statements to the console.
        """
        if num_tokens_per_detector is not None and token_size is not None:
            raise ValueError("Cannot specify both num_tokens and token_size.")
        # Rename num_tokens to num_tokens_per_block!!!!
        num_f = domain.frequency_mask_length
        if num_tokens_per_detector is not None:
            self.num_bins_per_token = np.ceil(num_f / num_tokens_per_detector).astype(int)
        elif token_size is not None:
            self.num_bins_per_token = token_size
            num_tokens_per_detector = np.ceil(num_f / self.num_bins_per_token).astype(int)
        else:
            raise ValueError(
                "It is necessary to specify either num_tokens or token_size."
            )
        if type(domain.delta_f) is float:
            delta_fs = np.array([domain.delta_f] * num_tokens_per_detector)
        elif type(domain.delta_f) is np.ndarray:
            delta_fs = domain.delta_f
        else:
            raise ValueError(f"domain.delta_f must be either float or np.ndarray, but is {type(domain.delta_f)}.")

        self.f_min_per_token = domain.sample_frequencies[domain.min_idx :: self.num_bins_per_token][:num_tokens_per_detector]
        self.f_max_per_token = domain.sample_frequencies[
                               domain.min_idx + self.num_bins_per_token - 1 :: self.num_bins_per_token
                               ][:num_tokens_per_detector]
        if len(self.f_min_per_token) != len(self.f_max_per_token):
            # Extrapolate last band
            f_token_widths = self.num_bins_per_token * delta_fs
            f_max_pad = self.f_max_per_token[-1] + f_token_widths[-1]
            self.f_max_per_token = np.append(self.f_max_per_token, f_max_pad)
            self.num_padded_f_bins = int((f_max_pad - domain.sample_frequencies[-1]) / delta_fs[-1])
        else:
            self.num_padded_f_bins = 0
        assert num_tokens_per_detector == len(self.f_min_per_token) == len(self.f_max_per_token)
        self.normalize_freq = normalize_frequency
        self.single_tokenizer = single_tokenizer
        self.f_min = domain.f_min
        self.f_max = self.f_max_per_token[-1]
        self.num_tokens_per_detector = num_tokens_per_detector

        if print_output:
            print(
                f"Tokenization:\n"
                f"  Token width {self.num_bins_per_token} frequency bins; {num_tokens_per_detector} "
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
            - 'waveform', shape [num_tokens, num_features] =
            [num_blocks * num_tokens_per_block, num_channels * num_bins_per_token]
            and additional keys
            - 'position', shape [num_blocks, num_tokens, 3]
               contains information [f_min, f_max, block]
            - 'drop_token_mask', shape [num_tokens]
               contains information about which tokens to drop. Defined as False = keep token, True = drop token,
               due torch transformer convention ("positions with a True value are not allowed to participate in the
               attention",
               see https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer ).
               Initialized to use all tokens, is modified by subsequent transforms.
        """
        sample = input_sample.copy()

        # Pad last dimension
        strain = np.pad(
            sample["waveform"],
            ((0, 0), (0, 0), (0, self.num_padded_f_bins)),
            "constant",
        )
        # Reshape strain to shape [seq_length, num_features]
        # = [num_blocks * num_tokens_per_detector, num_channels * num_bins_per_token]
        num_blocks, num_channels = strain.shape[0], strain.shape[1]
        strain = strain.reshape(
            num_blocks, num_channels, self.num_tokens_per_detector, self.num_bins_per_token
        )
        strain = np.moveaxis(strain, 2, 1)
        sample["waveform"] = strain.reshape(
            num_blocks * self.num_tokens_per_detector, num_channels * self.num_bins_per_token
        )
        # Prepare position information for each token
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

        num_tokens = num_blocks * self.num_tokens_per_detector
        token_position = np.empty((num_tokens, 3))
        token_position[:, 0] = np.repeat(f_min_per_token, num_blocks)
        token_position[:, 1] = np.repeat(f_max_per_token, num_blocks)
        token_position[:, 2] = np.repeat(detectors, self.num_tokens_per_detector)
        sample["position"] = token_position
        # Convention of torch transformer: positions with a True value are not allowed to participate in the attention
        # see https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer
        sample["drop_token_mask"] = np.zeros(num_tokens, dtype=bool)

        return sample


class DropFrequencyValues(object):
    """
    Randomly drop tokens for tokens corresponding to specific frequency values or ranges.
    """

    def __init__(
        self,
        print_output: bool = True,
    ):
        """
        Parameters
        ----------
        print_output: bool
            Whether to write print statements to the console.
        """
        if print_output:
            print("Transform DropFrequencyValues activated.")

    def __call__(self, input_sample):
        """
        Parameters
        ----------
        input_sample: Dict
            Values for keys
            - 'waveform':
            Sample of shape [num_tokens, num_features] =
            [num_blocks * num_tokens_per_block, num_channels * num_bins_per_token]
            where num_blocks = number of detectors in GW use case,
            num_channels>=3 (real, imag, auxiliary channels, e.g. asd),
            and num_bins = number of frequency bins.
            - 'position', shape [num_tokens, 3]
               contains information [f_min, f_max, block]
            - 'drop_token_mask', shape [num_tokens]

        Returns
        ----------
        sample: Dict
            input_sample with modified value for key
            - 'drop_token_mask', shape [num_tokens]

        """
        num_tokens = input_sample["waveform"].shape[0]
        blocks = input_sample["position"][:, 2]
        num_blocks = len(np.unique(blocks))
        num_tokens_per_block = num_tokens // num_blocks

        # Options (for each detector):
        # (1) cut in frequency domain, where we mask the upper or lower part, i.e. [f_min, f_cut] or [f_cut, f_max]
        # - sample index for f_cut uniformly, sample whether to mask upper or lower range
        # (2) drop frequency range:
        # - sample index for f_min and f_max, mask everything in between

        indices = np.zeros([num_blocks, 2])
        # Decide whether to cut or mask frequency range for each block
        apply_cut = np.random.choice([False, True], size=num_blocks)

        # Sample indices for cuts
        cut_indices = np.random.choice(np.arange(num_tokens_per_block), size=num_blocks)
        # Decide whether to mask tokens above or below the cut
        lower_range = apply_cut * np.random.choice([False, True], size=num_blocks)
        upper_range = apply_cut * ~lower_range
        # Insert indices for both options
        if np.sum(lower_range) > 0:
            indices[lower_range, :] = np.array([[0, cut_indices[i]] for i in range(num_blocks) if lower_range[i]])
        if np.sum(upper_range) > 0:
            indices[upper_range, :] = np.array([[cut_indices[i], num_tokens_per_block-1] for i in range(num_blocks)
                                                if upper_range[i]])

        # Sample min and max indices for masking token range
        indices_mask = np.random.choice(np.arange(num_tokens_per_block), size=[num_blocks, 2], replace=False)
        indices[~apply_cut, :] = np.sort(indices_mask, axis=-1)[~apply_cut, :]

        # Convert to absolute indices
        indices = np.array([indices[i, :] + i*num_tokens_per_block for i in range(num_blocks)], dtype=int)

        # Construct mask
        mask_blocks = np.zeros_like(blocks, dtype=bool)
        for b in range(num_blocks):
            mask_blocks[indices[b, 0]:indices[b, 1]] = True

        # Modify mask
        input_sample["drop_token_mask"] = np.logical_or(input_sample["drop_token_mask"], mask_blocks)

        return input_sample


class DropDetectors(object):
    """
    Randomly drop detectors.
    """

    def __init__(
            self,
            print_output: bool = True,
    ):
        """
        Parameters
        ----------
        print_output: bool
            Whether to write print statements to the console.
        """
        if print_output:
            print("Transform DropDetectors activated.")

    def __call__(self, input_sample):
        """
        Parameters
        ----------
        input_sample: Dict
            Values for keys
            - 'waveform':
            Sample of shape [num_tokens, num_features] =
            [num_blocks * num_tokens_per_block, num_channels * num_bins_per_token]
            where num_blocks = number of detectors in GW use case,
            num_channels>=3 (real, imag, auxiliary channels, e.g. asd),
            and num_bins = number of frequency bins.
            - 'position', shape [num_tokens, 3]
               contains information [f_min, f_max, block]
            - 'drop_token_mask', shape [num_tokens]

        Returns
        ----------
        sample: Dict
            input_sample with modified value for key
            - 'drop_token_mask', shape [num_tokens]

        """
        blocks = input_sample['position'][..., 2]
        num_blocks = len(np.unique(blocks))

        # Decide how many detectors to drop (either none, or one less than the number of detectors present)
        drop_n_blocks = np.random.randint(low=0, high=num_blocks)
        if drop_n_blocks > 0:
            # Decide which detectors
            detectors = np.unique(blocks)
            detectors_to_drop = np.random.choice(detectors, drop_n_blocks, replace=False)
            # Create mask such that tokens corresponding to dropped detectors are False
            mask_detectors = np.isin(blocks, detectors_to_drop)
            # Modify mask
            input_sample["drop_token_mask"] = np.logical_or(input_sample["drop_token_mask"], mask_detectors)

        return input_sample
