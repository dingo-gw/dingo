import numpy as np

from dingo.gw.domains import FrequencyDomain, MultibandedFrequencyDomain


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
        domain: FrequencyDomain | MultibandedFrequencyDomain,
        num_tokens_per_block: int = None,
        token_size: int = None,
        normalize_frequency: bool = False,
        single_tokenizer: bool = False,
        print_output: bool = True,
    ):
        """
        Parameters
        ----------
        domain: FrequencyDomain or MultiBandedFrequencyDomain
            Contains domain information, e.g., f_min, f_max, delta_f. Works with
            FrequencyDomain and MultibandedFrequencyDomain.
        num_tokens_per_block: int
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
        if num_tokens_per_block is not None and token_size is not None:
            raise ValueError("Cannot specify both num_tokens and token_size.")

        num_f = domain.frequency_mask_length
        if num_tokens_per_block is not None:
            self.num_bins_per_token = np.ceil(num_f / num_tokens_per_block).astype(int)
        elif token_size is not None:
            self.num_bins_per_token = token_size
            num_tokens_per_block = np.ceil(num_f / self.num_bins_per_token).astype(int)
        else:
            raise ValueError(
                "It is necessary to specify either num_tokens or token_size."
            )
        # We assume that we have the same f_min, f_max, and delta_f for all data points in the batch
        if isinstance(domain, FrequencyDomain):
            assert isinstance(domain.delta_f, float), (
                "Expected domain.delta_f of FrequencyDomain to be float, but "
                "received {domain.delta_f}"
            )
            delta_fs = np.array([domain.delta_f] * num_tokens_per_block)
        elif isinstance(domain, MultibandedFrequencyDomain):
            delta_fs = domain.delta_f
        else:
            raise ValueError(
                f"domain.delta_f must be either float or np.ndarray, but is {type(domain.delta_f)}."
            )
        # Construct f_min_per_token and f_max_per_token
        self.f_min_per_token = domain.sample_frequencies[
            domain.min_idx :: self.num_bins_per_token
        ][:num_tokens_per_block]
        self.f_max_per_token = domain.sample_frequencies[
            domain.min_idx + self.num_bins_per_token - 1 :: self.num_bins_per_token
        ][:num_tokens_per_block]
        if len(self.f_min_per_token) != len(self.f_max_per_token):
            # Extrapolate last band
            f_token_widths = self.num_bins_per_token * delta_fs
            f_max_pad = self.f_max_per_token[-1] + f_token_widths[-1]
            self.f_max_per_token = np.append(self.f_max_per_token, f_max_pad)
            self.num_padded_f_bins = int(
                (f_max_pad - domain.sample_frequencies[-1]) / delta_fs[-1]
            )
        else:
            self.num_padded_f_bins = 0
        if not (
            num_tokens_per_block
            == len(self.f_min_per_token)
            == len(self.f_max_per_token)
        ):
            raise ValueError(
                "f_min_per_token and f_max_per_token are not of length num_tokens_per_block."
            )
        self.normalize_freq = normalize_frequency
        self.f_min = domain.f_min
        self.f_max = self.f_max_per_token[-1]
        self.num_tokens_per_detector = num_tokens_per_block

        if print_output:
            print(
                f"Tokenization:\n"
                f"  Token width {self.num_bins_per_token} frequency bins; {num_tokens_per_block} "
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
                Sample of shape [batch_size, num_blocks, num_channels, num_bins]
                where num_blocks = number of detectors in GW use case,
                num_channels>=3 (real, imag, auxiliary channels, e.g. asd),
                and num_bins=number of frequency bins.
            Value for key 'asds':
                Dictionary containing asd for each detector. This transform only accesses the asds keys to determine
                which detectors are involved and does not modify the asds.

        Returns
        ----------
        sample: Dict
            input_sample with modified value for key
            - 'waveform', shape [batch_size, num_tokens, num_features] =
               [batch_size, num_blocks * num_tokens_per_block, num_channels * num_bins_per_token]
               and additional keys
            - 'position', shape [batch_size, num_blocks, num_tokens, 3]
               where last dimension contains information [f_min, f_max, block]
            - 'drop_token_mask', shape [batch_size, num_tokens]
               contains information about which tokens to drop. Defined as False = keep token, True = drop token,
               due torch transformer convention ("positions with a True value are not allowed to participate in the
               attention",
               see https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer ).
               Initialized to use all tokens, is modified by subsequent transforms.
        """
        sample = input_sample.copy()

        # Pad last dimension
        pad_width = [
            (
                (0, 0)
                if i < len(sample["waveform"].shape) - 1
                else (0, self.num_padded_f_bins)
            )
            for i in range(len(sample["waveform"].shape))
        ]
        strain = np.pad(sample["waveform"], pad_width, "constant")
        # Reshape strain to shape [seq_length, num_features]
        # = [num_blocks * num_tokens_per_detector, num_channels * num_bins_per_token]
        num_blocks, num_channels = strain.shape[-3], strain.shape[-2]
        # (1) Reshape strain from [..., num_bins] to [..., num_tokens_per_detector, num_bins_per_token]
        # This step divides the frequency range into equal segments of num_bins_per_token and stacks these segments,
        # resulting in: [[waveform(f_min_token1), ..., waveform(f_max_token1)],
        #                [waveform(f_min_token2), ..., waveform(f_max_token2)],
        #                ...,
        #                [waveform(f_min_tokenN), ..., waveform(f_max_tokenN)]]
        strain = strain.reshape(
            *strain.shape[:-1], self.num_tokens_per_detector, self.num_bins_per_token
        )
        # (2) Reshape strain from [..., num_blocks, num_channels, num_tokens_per_detector, num_bins_per_token]
        # to [..., num_blocks, num_tokens_per_detector, num_channels, num_bins_per_token]
        strain = np.moveaxis(strain, source=-2, destination=-3)
        # (3) Reshape strain from [..., num_blocks, num_tokens_per_detector, num_channels, num_bins_per_token]
        # to [..., num_blocks*num_tokens_per_detector, num_channels*num_bins_per_token]
        # = [..., num_tokens, num_features]
        # The ordering of the tokens is relevant for the frequency and detector embedding: We want
        # [H1_token1, H1_token2,..., H1_tokenN, L1_token1,..., L1_tokenN, V1_token1,..., V1_tokenN]
        # The ordering of the elements in the last dimension is irrelevant since these values get jointly mapped to the
        # embedding dimension of the transformer. However, for completeness: In each token, the three channels are
        # stacked: [real, imag, asd]. To plot the real part, you can loop over the tokens of the first detector
        # and collect the first num_features/3 values.
        sample["waveform"] = strain.reshape(
            *strain.shape[:-4],
            num_blocks * self.num_tokens_per_detector,
            num_channels * self.num_bins_per_token,
        )
        # Prepare position information for each token
        detector_dict = {"H1": 0, "L1": 1, "V1": 2}
        if strain.shape[:-4] == ():
            detectors = np.array(
                [[detector_dict[k] for k, v in input_sample["asds"].items()]]
            )
        else:
            detectors = np.array(
                [
                    [detector_dict[k] for _ in range(len(v))]
                    for k, v in input_sample["asds"].items()
                ]
            ).T
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
        token_position = np.empty((*strain.shape[:-4], num_tokens, 3))
        # Treat sample without batch dimension separately because repeat with repeats=() throws error
        if strain.shape[:-4] == ():
            token_position[..., 0] = np.tile(f_min_per_token, num_blocks)
            token_position[..., 1] = np.tile(f_max_per_token, num_blocks)
        else:
            token_position[..., 0] = np.repeat(
                np.expand_dims(np.tile(f_min_per_token, num_blocks), axis=0),
                *strain.shape[:-4],
                axis=0,
            )
            token_position[..., 1] = np.repeat(
                np.expand_dims(np.tile(f_max_per_token, num_blocks), axis=0),
                *strain.shape[:-4],
                axis=0,
            )
        token_position[..., 2] = np.repeat(
            detectors, self.num_tokens_per_detector, axis=1
        )
        sample["position"] = token_position.astype(sample["waveform"].dtype)
        # Convention of torch transformer: positions with a True value are not allowed to participate in the attention
        # see https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer
        sample["drop_token_mask"] = np.zeros(
            [*strain.shape[:-4], num_tokens], dtype=bool
        )

        return sample


class DropFrequencyValues(object):
    """
    Randomly drop tokens for tokens corresponding to specific frequency values or ranges.
    """

    def __init__(
        self,
        domain: FrequencyDomain | MultibandedFrequencyDomain,
        drop_f_settings: dict | None = None,
        print_output: bool = True,
    ):
        """
        Parameters
        ----------
        domain: FrequencyDomain | MultibandedFrequencyDomain
        drop_f_settings: dict
            Contains settings for the DropFrequencyValues transform.
        print_output: bool
            Whether to write print statements to the console.
        """
        self.domain = domain
        self.f_cut = False
        if "f_cut" in drop_f_settings:
            self.f_cut = True
            self.p_cut = drop_f_settings["f_cut"].get("p_cut", 0.2)
            self.f_max_lower_cut = drop_f_settings["f_cut"].get(
                "f_max_lower_cut", self.domain.f_max
            )
            self.f_min_upper_cut = drop_f_settings["f_cut"].get(
                "f_min_upper_cut", self.domain.f_min
            )
            self.p_same_cut_all_detectors = drop_f_settings["f_cut"].get(
                "p_same_cut_all_detectors", 0.2
            )
        self.mask_glitch = False
        if "mask_glitch" in drop_f_settings:
            self.mask_glitch = True
            self.p_glitch_per_detector = drop_f_settings["mask_glitch"].get(
                "p_glitch_per_detector", 0.2
            )
            self.glitch_f_min = drop_f_settings["mask_glitch"].get(
                "f_min", self.domain.f_min
            )
            self.glitch_f_max = drop_f_settings["mask_glitch"].get(
                "f_min", self.domain.f_max
            )
            self.glitch_max_width = drop_f_settings["mask_glitch"].get(
                "max_width", (self.domain.f_max - self.domain.f_min) / 2
            )
        if print_output:
            print(f"Transform DropFrequencyValues activated:")
            if self.f_cut:
                print(
                    f"  Frequency cut activated, settings: \n"
                    f"    - Probability of a cut happening: {self.p_glitch_per_detector}\n"
                    f"    - Lower cut sampled from [{self.domain.f_min}, {self.f_max_lower_cut}]\n"
                    f"    - Upper cut sampled from [{self.f_min_upper_cut}, {self.domain.f_max}]\n"
                    f"    - Probability to apply the same cut on all detectors: {self.p_same_cut_all_detectors}"
                )
            else:
                print("   Frequency cut not activated.")
            if self.mask_glitch:
                print(
                    f"  Masking of glitch activated, settings: \n"
                    f"    - Probability of a glitch happening per detector: {self.p_glitch_per_detector}\n"
                    f"    - Glitch range sampled from [{self.glitch_f_min}, {self.glitch_f_max}]\n"
                    f"    - Maximal width of glitch: {self.glitch_max_width}"
                )
            else:
                print("   Masking of glitch not activated.")
            if not self.f_cut and not self.mask_glitch:
                raise ValueError(
                    "Arguments for drop_frequency_range.f_cut and drop_frequency_range.mask_glitch are "
                    "both missing although the DropFrequencyValues transform is activated. Either provide"
                    "arguments for f_cut or mask_glitch or remove transform from settings."
                )

    def __call__(self, input_sample):
        """
        Parameters
        ----------
        input_sample: Dict
            Values for keys
            - 'waveform':
                Sample of shape [batch_size, num_tokens, num_features] =
                [batch_size, num_blocks * num_tokens_per_block, num_channels * num_bins_per_token]
                where num_blocks = number of detectors in GW use case,
                num_channels>=3 (real, imag, auxiliary channels, e.g. asd),
                and num_bins = number of frequency bins.
            - 'position', shape [batch_size, num_tokens, 3]
               contains information [f_min, f_max, block]
            - 'drop_token_mask', shape [batch_size, num_tokens]

        Returns
        ----------
        sample: Dict
            input_sample with modified value for key
            - 'drop_token_mask', shape [batch_size, num_tokens]

        """
        num_tokens = input_sample["waveform"].shape[-2]
        blocks = input_sample["position"][..., 2]
        num_blocks = len(np.unique(blocks))
        num_tokens_per_block = num_tokens // num_blocks

        # Options (for each detector):
        # (1) cut in frequency domain, where we remove the upper or lower part, i.e. [f_min, f_cut] or [f_cut, f_max]
        # - sample whether to mask upper or lower range
        # - sample index for f_cut from [f_min, f_max_lower_cut] or [f_min_upper_cut, f_max] in uniform frequency domain
        # - convert to multibanded frequency domain
        # - relate index in multibanded frequency domain to token
        # (2) mask frequency range:
        # - sample index for f_mask_lower and f_mask_upper in uniform frequency domain
        # - optional: convert to multibanded frequency domain
        # - mask everything in between, i.e. [f_mask_lower, f_mask_upper],
        #   resulting in [f_min, f_mask_lower] + [f_mask_upper, f_max]

        indices = np.zeros([num_blocks, 2])
        # Decide whether to cut or mask frequency range for each block
        apply_cut = np.random.choice([False, True], size=num_blocks)

        # Sample cut index as well as min and max indices for mask
        if isinstance(self.domain, FrequencyDomain):
            # Sample cut indices in uniform frequency domain
            indices_cut = np.random.choice(
                np.arange(self.domain.frequency_mask_length), size=num_blocks
            )
            # Sample mask indices in uniform frequency domain
            indices_mask = np.random.choice(
                np.arange(self.domain.frequency_mask_length),
                size=[num_blocks, 2],
                replace=False,
            )
        elif isinstance(self.domain, MultibandedFrequencyDomain):
            # Sample cut indices in uniform frequency domain
            f_cut = np.random.choice(
                self.domain.base_domain.sample_frequencies[
                    self.domain.base_domain.frequency_mask
                ],
                size=num_blocks,
            )
            # Find closest frequency value and corresponding index in multibanded frequency domain
            indices_cut = np.argmin(
                np.abs(np.subtract.outer(f_cut, self.domain.sample_frequencies)), axis=1
            )
            # Sample mask indices in uniform frequency domain
            f_mask = np.random.choice(
                self.domain.base_domain.sample_frequencies[
                    self.domain.base_domain.frequency_mask
                ],
                size=[num_blocks, 2],
                replace=False,
            )
            # Find closest frequency value and corresponding index in multibanded frequency domain
            indices_mask = np.argmin(
                np.abs(np.subtract.outer(f_mask, self.domain.sample_frequencies)),
                axis=2,
            )
        else:
            raise ValueError(
                f"self.domain is of type {type(self.domain).__name__} but should be either FrequencyDomain or MultibandedFrequencyDomain."
            )

        # Decide whether to mask tokens above or below the cut
        lower_range = apply_cut * np.random.choice([False, True], size=num_blocks)
        upper_range = apply_cut * ~lower_range
        # Insert indices for both (above and below) options
        if np.sum(lower_range) > 0:
            indices[lower_range, :] = np.array(
                [[0, indices_cut[i]] for i in range(num_blocks) if lower_range[i]]
            )
        if np.sum(upper_range) > 0:
            indices[upper_range, :] = np.array(
                [
                    [indices_cut[i], num_tokens_per_block - 1]
                    for i in range(num_blocks)
                    if upper_range[i]
                ]
            )

        # Insert mask indices where there is no cut
        indices[~apply_cut, :] = np.sort(indices_mask, axis=-1)[~apply_cut, :]

        # Convert to absolute transformer token indices
        indices = np.array(
            [indices[i, :] + i * num_tokens_per_block for i in range(num_blocks)],
            dtype=int,
        )

        # Construct mask
        mask_blocks = np.zeros_like(blocks, dtype=bool)
        for b in range(num_blocks):
            mask_blocks[indices[b, 0] : indices[b, 1]] = True

        # Modify mask
        input_sample["drop_token_mask"] = np.logical_or(
            input_sample["drop_token_mask"], mask_blocks
        )

        return input_sample

        #### OLD CODE ####
        # # Sample indices for cuts
        # cut_indices = np.random.choice(np.arange(num_tokens_per_block), size=num_blocks)
        # # Decide whether to mask tokens above or below the cut
        # lower_range = apply_cut * np.random.choice([False, True], size=num_blocks)
        # upper_range = apply_cut * ~lower_range
        # # Insert indices for both options
        # if np.sum(lower_range) > 0:
        #     indices[lower_range, :] = np.array([[0, cut_indices[i]] for i in range(num_blocks) if lower_range[i]])
        # if np.sum(upper_range) > 0:
        #     indices[upper_range, :] = np.array([[cut_indices[i], num_tokens_per_block-1] for i in range(num_blocks)
        #                                         if upper_range[i]])
        #
        # # Sample min and max indices for masking token range
        # indices_mask = np.random.choice(np.arange(num_tokens_per_block), size=[num_blocks, 2], replace=False)
        # indices[~apply_cut, :] = np.sort(indices_mask, axis=-1)[~apply_cut, :]
        #
        # # Convert to absolute indices
        # indices = np.array([indices[i, :] + i*num_tokens_per_block for i in range(num_blocks)], dtype=int)
        #
        # # Construct mask
        # mask_blocks = np.zeros_like(blocks, dtype=bool)
        # for b in range(num_blocks):
        #     mask_blocks[indices[b, 0]:indices[b, 1]] = True
        #
        # # Modify mask
        # input_sample["drop_token_mask"] = np.logical_or(input_sample["drop_token_mask"], mask_blocks)
        #
        # return input_sample


class DropDetectors(object):
    """
    Randomly drop detectors.
    """

    def __init__(
        self,
        num_blocks: int,
        p_drop_012_detectors: list | None = None,
        p_drop_hlv: dict | None = None,
        print_output: bool = True,
    ):
        """
        Parameters
        ----------
        p_drop_012_detectors: list[float]
            Specifies the categorical probability distribution for how many detectors to drop, in ascending order
            example: [0.1, 0.6, 0.3] = [10% probability to drop 0 detectors (=3 detector setup), 60 % probability for
            2 detector setup, 30% probability for 1 detector setup]
        p_drop_hlv: dict
            Specifies the categorical probability distribution for which specific detectors to drop, order: H1, L1, V1
            example: {'H1': 0.1, 'L1': 0.2, 'V1': 0.7] = 10 % probability to drop H1, 20 % probability to drop L1,
            70% probability to drop V1
        print_output: bool
            Whether to write print statements to the console.
        """
        self.num_blocks = num_blocks
        if p_drop_012_detectors is None:
            p_drop_012_detectors = [1 / num_blocks for _ in range(num_blocks)]
        if np.sum(p_drop_012_detectors) != 1:
            raise ValueError(
                f"p_drop_012_detectors {p_drop_012_detectors} does not sum to 1."
            )
        self.p_drop_012_detectors = p_drop_012_detectors
        if p_drop_hlv is None:
            p_drop_hlv = {
                ["H1", "L1", "V1"][k]: 1 / num_blocks for k in range(num_blocks)
            }
        if np.sum(list(p_drop_hlv.values())) != 1:
            raise ValueError(f"p_drop_hlv {p_drop_hlv} does not sum to 1.")
        # Update keys equivalently to tokenization transform
        detector_dict = {"H1": 0, "L1": 1, "V1": 2}
        self.p_drop_hlv = {detector_dict[k]: v for k, v in p_drop_hlv.items()}

        if len(p_drop_012_detectors) > num_blocks:
            raise ValueError(
                f"p_drop_num_detectors {self.p_drop_012_detectors} contains more options than"
                f"detectors available: {num_blocks}. You need to specify a categorical probability"
                f"value for dropping 0, ..., {num_blocks - 1} detectors."
            )
        if len(self.p_drop_hlv) != num_blocks:
            raise ValueError(
                f"Provided values for p_drop_hlv={self.p_drop_hlv} is inconsistent with number of "
                f"detectors: {num_blocks}. You need to specify a categorical probability value for each "
                f"detector."
            )
        if print_output:
            print(
                f"Transform DropDetectors activated: \n"
                f"  Probabilities for dropping {[i for i in range(num_blocks)]} detectors are {self.p_drop_012_detectors}.\n"
                f"  Probabilities for specific detectors are {self.p_drop_hlv}."
            )

    def __call__(self, input_sample):
        """
        Parameters
        ----------
        input_sample: Dict
            Values for keys
            - 'waveform':
            Sample of shape [batch_size, num_tokens, num_features] =
            [batch_size, num_blocks * num_tokens_per_block, num_channels * num_bins_per_token]
            where num_blocks = number of detectors in GW use case,
            num_channels>=3 (real, imag, auxiliary channels, e.g. asd),
            and num_bins = number of frequency bins.
            - 'position', shape [batch_size, num_tokens, 3]
               contains information [f_min, f_max, block]
            - 'drop_token_mask', shape [batch_size, num_tokens]

        Returns
        ----------
        sample: Dict
            input_sample with modified value for key
            - 'drop_token_mask', shape [batch_size, num_tokens]

        """
        blocks = input_sample["position"][..., 2]
        num_blocks = len(np.unique(blocks))
        detectors = np.unique(blocks)

        # Convert p_drop_hlv dict to list
        p_drop_hlv = [self.p_drop_hlv[k] for k in detectors]

        # Decide how many detectors to drop (either none, or one less than the number of detectors present)
        # for each element in batch_size
        drop_n_blocks = np.random.choice(
            [i for i in range(num_blocks)],
            p=self.p_drop_012_detectors,
            size=[*blocks.shape[:-1]],
        )
        if np.sum(drop_n_blocks) != 0:
            # Treat drop 1 vs. 2 blocks separately because which detectors to drop varies
            # with the number of detectors to drop
            for n in [i for i in np.unique(drop_n_blocks) if i > 0]:
                # Construct mask for which batch indices require updates
                mask_mod = np.where(drop_n_blocks == n, True, False)
                # Decide which detectors
                detectors_to_drop = np.apply_along_axis(
                    np.random.choice,
                    axis=1,
                    arr=np.repeat(
                        np.expand_dims(detectors, 0), repeats=np.sum(mask_mod), axis=0
                    ),
                    p=p_drop_hlv,
                    size=n,
                    replace=False,
                )
                # Create mask such that tokens corresponding to dropped detectors are True
                # (1) Drop one detector
                mask_detectors = np.where(
                    blocks[mask_mod].T == detectors_to_drop[:, 0], True, False
                ).T
                if detectors_to_drop.shape[-1] > 1:
                    # (2) Update mask to include dropping of any further detector
                    for i in range(1, detectors_to_drop.shape[-1]):
                        mask_detectors_i = np.where(
                            blocks[mask_mod].T == detectors_to_drop[:, i], True, False
                        ).T
                        mask_detectors = np.logical_or(mask_detectors_i, mask_detectors)
                # Keep drop=True from previous transforms with logical OR
                mask_detectors = np.logical_or(
                    input_sample["drop_token_mask"][mask_mod], mask_detectors
                )
                # Update mask
                input_sample["drop_token_mask"][mask_mod] = mask_detectors

        return input_sample
