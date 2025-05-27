from typing import Optional
import numpy as np
from copy import deepcopy

from dingo.gw.domains import UniformFrequencyDomain, MultibandedFrequencyDomain
from dingo.gw.gwutils import add_defaults_for_missing_ifos

DETECTOR_DICT = {"H1": 0, "L1": 1, "V1": 2}
DETECTOR_DICT_INVERSE = {0: "H1", 1: "L1", 2: "V1"}


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
        domain: UniformFrequencyDomain | MultibandedFrequencyDomain,
        num_tokens_per_block: int = None,
        token_size: int = None,
        normalize_frequency: bool = False,
        single_tokenizer: bool = False,
        drop_last_token: bool = False,
        print_output: bool = True,
    ):
        """
        Parameters
        ----------
        domain: UniformFrequencyDomain or MultiBandedFrequencyDomain
            Contains domain information, e.g., f_min, f_max, delta_f. Works with
            UniformFrequencyDomain and MultibandedFrequencyDomain.
        num_tokens_per_block: int
            Number of tokens into which the domain should be divided. [Optional]
        token_size: int
            Number of frequency bins per token. It is necessary to specify one of
            num_tokens or token_size. [Optional]
        drop_last_token: bool
            Whether to drop the last token of each block if it is incomplete. False pads the last token with zeros.
        print_output: bool
            Whether to write print statements to the console.
        """
        assert normalize_frequency is False and single_tokenizer is False
        if num_tokens_per_block is not None and token_size is not None:
            raise ValueError("Cannot specify both num_tokens and token_size.")

        self.drop_last_token = drop_last_token
        num_f = domain.frequency_mask_length
        if num_tokens_per_block is not None:
            if num_f % num_tokens_per_block != 0:
                self.num_bins_per_token = np.ceil(num_f / num_tokens_per_block).astype(
                    int
                )
                if self.drop_last_token:
                    num_tokens_per_block -= 1
            else:
                self.num_bins_per_token = int(num_f / num_tokens_per_block)
        elif token_size is not None:
            self.num_bins_per_token = token_size
            if num_f % self.num_bins_per_token != 0:
                if self.drop_last_token:
                    num_tokens_per_block = np.floor(num_f / token_size).astype(int)
                else:
                    num_tokens_per_block = np.ceil(num_f / token_size).astype(int)
            else:
                num_tokens_per_block = int(num_f / token_size)
        else:
            raise ValueError(
                "It is necessary to specify either num_tokens or token_size."
            )

        # We assume that we have the same f_min, f_max, and delta_f for all data points in the batch
        if isinstance(domain, UniformFrequencyDomain):
            assert isinstance(domain.delta_f, float), (
                "Expected domain.delta_f of UniformFrequencyDomain to be float, but "
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

        self.num_padded_f_bins = 0.0
        if (
            len(self.f_min_per_token) > len(self.f_max_per_token)
            and not self.drop_last_token
        ):
            # Extrapolate last band
            f_token_widths = self.num_bins_per_token * delta_fs
            f_max_pad = self.f_max_per_token[-1] + f_token_widths[-1]
            self.f_max_per_token = np.append(self.f_max_per_token, f_max_pad)
            self.num_padded_f_bins = int(
                (f_max_pad - domain.sample_frequencies[-1]) / delta_fs[-1]
            )

        if not (
            num_tokens_per_block
            == len(self.f_min_per_token)
            == len(self.f_max_per_token)
        ):
            raise ValueError(
                "f_min_per_token and f_max_per_token are not of length num_tokens_per_block."
            )
        self.num_tokens_per_detector = num_tokens_per_block
        # Ensure that tokenization is compatible with the position of the MFD nodes
        if isinstance(domain, MultibandedFrequencyDomain):
            check_compatibility_of_mfd_nodes_with_tokenization(
                f_mins=self.f_min_per_token,
                f_maxs=self.f_max_per_token,
                mfd_nodes=domain.nodes,
                drop_last_token=self.drop_last_token,
            )

        if print_output:
            print(
                f"Tokenization:\n"
                f"    - Token width: {self.num_bins_per_token} frequency bins; {num_tokens_per_block} "
                f"tokens per detector\n"
                f"    - Dropping last incomplete token: {self.drop_last_token}\n"
                f"    - First token width {self.f_min_per_token[1] - self.f_min_per_token[0]} "
                f"Hz\n"
                f"    - Last token width {self.f_min_per_token[-1] - self.f_min_per_token[-2]} "
                f"Hz\n"
            )
            if self.num_padded_f_bins > 0:
                print(
                    f"    - Extrapolating to maximum frequency of {self.f_max_per_token[-1]} Hz"
                )

    def __call__(self, input_sample: dict) -> dict:
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

        # Reshape strain from shape [batch_size, num_blocks, num_channels, num_bins] to [seq_length, num_features]
        # = [num_blocks * num_tokens_per_detector, num_channels * num_bins_per_token]
        # (0) Cut/Pad num_bins such that it is compatible with num_tokens_per_detector * num_bins_per_token
        if self.num_padded_f_bins == 0:
            # Only take bins that fit into the tokens
            strain = sample["waveform"][
                ..., : self.num_tokens_per_detector * self.num_bins_per_token
            ]
        else:
            # Pad last dimension with 0.
            pad_width = [
                (
                    (0, 0)
                    if i < len(sample["waveform"].shape) - 1
                    else (0, self.num_padded_f_bins)
                )
                for i in range(len(sample["waveform"].shape))
            ]
            strain = np.pad(sample["waveform"], pad_width, "constant")

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
        if strain.shape[:-4] == ():
            detectors = np.array(
                [[DETECTOR_DICT[k] for k, v in input_sample["asds"].items()]],
                dtype=strain.dtype,
            )
        else:
            detectors = np.array(
                [
                    [DETECTOR_DICT[k] for _ in range(len(v))]
                    for k, v in input_sample["asds"].items()
                ],
                dtype=strain.dtype,
            ).T
        num_tokens = num_blocks * self.num_tokens_per_detector
        token_position = np.empty(
            (*strain.shape[:-4], num_tokens, 3), dtype=strain.dtype
        )
        # Treat sample without batch dimension separately because repeat with repeats=() throws error
        if strain.shape[:-4] == ():
            token_position[..., 0] = np.tile(self.f_min_per_token, num_blocks)
            token_position[..., 1] = np.tile(self.f_max_per_token, num_blocks)
        else:
            token_position[..., 0] = np.repeat(
                np.expand_dims(np.tile(self.f_min_per_token, num_blocks), axis=0),
                *strain.shape[:-4],
                axis=0,
            )
            token_position[..., 1] = np.repeat(
                np.expand_dims(np.tile(self.f_max_per_token, num_blocks), axis=0),
                *strain.shape[:-4],
                axis=0,
            )
        token_position[..., 2] = np.repeat(
            detectors, self.num_tokens_per_detector, axis=1
        )
        sample["position"] = token_position
        # Convention of torch transformer: positions with a True value are not allowed to participate in the attention
        # see https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer
        sample["drop_token_mask"] = np.zeros(
            [*strain.shape[:-4], num_tokens], dtype=bool
        )

        return sample


def check_compatibility_of_mfd_nodes_with_tokenization(
    f_mins: np.array, f_maxs: np.array, mfd_nodes: np.array, drop_last_token: bool
):
    """
    Check whether the nodes of the multibanded frequency domain are located between tokens. This means that for token
    frequencies $[f_\mathrm{min}^{(i)}, f_\mathrm{max}^{(i)}]$, the nodes have to be located in
    $[f_\mathrm{max}^{(i)}, f_\mathrm{min}^{(i+1)}]$.

    This is required since we have to make sure that the strain values within one token are equally spaced
    (i.e., same $\Delta f$).

    Parameters
    ----------
    f_mins: np.array
        f_min for all tokens of a single detector. shape (num_tokens,)
    f_maxs: np.array
        f_max for all tokens of a single detector. shape (num_tokens,)
    mfd_nodes: np.array
        Nodes of multibanded frequency domain
    drop_last_token: bool
        Whether the last token was dropped or not. If False, the last node can appear within the frequency range of the
        last token.
    """

    # Construct left and right bounds for intervals
    left_bounds = np.concatenate([[0], f_maxs[:-1]])
    right_bounds = f_mins
    intervals = np.stack([left_bounds, right_bounds], axis=1)

    # Check that each node is in one interval
    covered = np.any(
        (mfd_nodes[:, None] >= intervals[:, 0])
        & (mfd_nodes[:, None] <= intervals[:, 1]),
        axis=1,
    )

    # Last node can be larger than last f_max or node can be located in the last token if we extrapolate the last token
    if not covered[-1] and (mfd_nodes[~covered][0] > f_maxs[-1] or not drop_last_token):
        covered[-1] = True

    if not np.all(covered):
        raise ValueError(
            f"Nodes of MFD {mfd_nodes} are not compatible with tokenization, nodes {mfd_nodes[~covered]} fall within "
            f"a token."
        )


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
        num_blocks: int
            Number of blocks (= detectors) in GW use case.
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
        if not np.isclose(np.sum(p_drop_012_detectors), 1.0, rtol=1e-6, atol=1e-12):
            raise ValueError(
                f"p_drop_012_detectors {p_drop_012_detectors} does not sum to 1."
            )
        self.p_drop_012_detectors = p_drop_012_detectors
        if p_drop_hlv is None:
            p_drop_hlv = {
                ["H1", "L1", "V1"][k]: 1 / num_blocks for k in range(num_blocks)
            }
        if not np.isclose(
            np.sum(list(p_drop_hlv.values())), 1.0, rtol=1e-6, atol=1e-12
        ):
            raise ValueError(f"p_drop_hlv {p_drop_hlv} does not sum to 1.")
        # Update keys equivalently to tokenization transform
        self.p_drop_hlv = {DETECTOR_DICT[k]: v for k, v in p_drop_hlv.items()}

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
                f"    - Probabilities for dropping {[i for i in range(num_blocks)]} detectors are "
                f"{self.p_drop_012_detectors}.\n"
                f"    - Probabilities for specific detectors are {self.p_drop_hlv}."
            )

    def __call__(self, input_sample: dict) -> dict:
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


class DropFrequenciesToUpdateRange(object):
    """
    Randomly drop tokens such that f_min and f_max of the frequency range are updated.

    This transform does the following things:
    * Decides whether to apply a cut to each element of the batch based on p_cut.
    * Decides whether to treat the detectors individually or apply the same cut to all detectors.
    * Decides whether to cut upper or lower end or both (potentially for each detector).
    * Samples f_cut from [f_min, f_max_lower_cut] and/or [f_min_upper_cut, f_max] in UFD (potentially for each
      detector).
    * Converts frequency values to tokens and creates a token mask removing [f_min, f_lower_cut] and/or
      [f_upper_cut, f_max] (potentially for each detector).
    """

    def __init__(
        self,
        domain: UniformFrequencyDomain | MultibandedFrequencyDomain,
        p_cut: float,
        f_max_lower_cut: float,
        f_min_upper_cut: float,
        p_same_cut_all_detectors: float,
        p_lower_upper_both: Optional[list] = None,
        print_output: bool = True,
    ):
        """
        Parameters
        ----------
        domain: UniformFrequencyDomain | MultibandedFrequencyDomain
            Domain corresponding to the data being transformed.
        p_cut: float
            Probability of applying a cut to each element of the batch.
        f_max_lower_cut: float
            Maximal frequency value to cut at the lower end of the frequency domain. f_min_lower_cut is sampled from
            [f_min, f_max_lower_cut] in UFD.
        f_min_upper_cut: float
            Minimal frequency value to cut at the upper end of the frequency domain. f_max_upper_cut is sampled from
            [f_min_upper_cut, f_max] in UFD.
        p_same_cut_all_detectors: float
            Probability of applying the same cut to all detectors.
        p_lower_upper_both: list[float]
            List of probabilities explaining with what probability we either cut at the lower, at the upper, or at both
            ends. Order: [p_lower, p_upper, p_both]
        print_output: bool
            Whether to write print statements to the console.
        """

        self.domain = domain
        self.p_cut = p_cut
        self.f_max_lower_cut = f_max_lower_cut
        self.f_min_upper_cut = f_min_upper_cut
        self.p_same_cut_all_detectors = p_same_cut_all_detectors
        if p_lower_upper_both is None:
            p_lower_upper_both = np.array([0.4, 0.4, 0.2])
        self.p_lower_upper_both = p_lower_upper_both
        if not np.isclose(np.sum(self.p_lower_upper_both), 1.0, rtol=1e-6, atol=1e-12):
            raise ValueError(
                f"p_lower_upper_both {self.p_lower_upper_both} does not sum to 1. "
            )
        if print_output:
            print(
                f"Transform DropFrequencyValues activated: \n"
                f"    - Probability of a cut happening: {self.p_cut}\n"
                f"    - Lower cut sampled from [{self.domain.f_min}, {self.f_max_lower_cut}]\n"
                f"    - Upper cut sampled from [{self.f_min_upper_cut}, {self.domain.f_max}]\n"
                f"    - Probability to apply the same cut on all detectors: {self.p_same_cut_all_detectors} "
            )

    def __call__(self, input_sample: dict) -> dict:
        """
        Parameters
        ----------
        input_sample: Dict
            Values for keys
            - 'waveform':
                Sample of shape [batch_size, num_tokens, num_features]
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

        # Cut in frequency domain, where we remove the upper, lower or both part(s),
        #     i.e. [f_min, f_cut], [f_cut, f_max], or [f_cut_min, f_cut_max]
        # - Decide whether to apply a cut for each sample
        # - Decide whether to treat the detectors individually or apply the same cut to all detectors
        # - Decide whether to mask upper or lower range or both (potentially for each detector)
        # - Sample index for f_cut from [f_min, f_max_lower_cut] and/or [f_min_upper_cut, f_max]
        #   in uniform frequency domain (potentially for each detector)
        # - Convert frequency values to token mask

        batch_size = [*blocks.shape[:-1]] if blocks.shape[:-1] != () else [1]
        # Decide whether to apply a cut for each sample
        apply_cut = np.random.choice(
            [True, False], p=[self.p_cut, 1 - self.p_cut], size=batch_size
        )

        # Decide whether to treat the detectors individually or apply the same cut to all detectors
        same_cut_all_detectors = np.where(
            apply_cut,
            np.random.choice(
                [True, False],
                p=[self.p_same_cut_all_detectors, 1 - self.p_same_cut_all_detectors],
                size=batch_size,
            ),
            False,
        )
        batch_block_size = (
            [*blocks.shape[:-1], num_blocks]
            if blocks.shape[:-1] != ()
            else [1, num_blocks]
        )
        # (1) Different cut is applied to every detector
        # Decide whether to mask upper or lower range or both (potentially for each detector)
        lower_upper_both_separate = np.random.choice(
            ["lower", "upper", "both"], p=self.p_lower_upper_both, size=batch_block_size
        )
        mask_lower_separate = np.logical_or(
            lower_upper_both_separate == "lower", lower_upper_both_separate == "both"
        )
        mask_upper_separate = np.logical_or(
            lower_upper_both_separate == "upper", lower_upper_both_separate == "both"
        )
        # Combine with masks (a) whether we apply a cut and (b) whether we apply it to a single detector
        ones_vec = np.ones((1, num_blocks), dtype=bool)
        mask_lower_separate_combined = np.logical_and.reduce(
            (
                mask_lower_separate,
                apply_cut[..., None] * ones_vec,
                ~same_cut_all_detectors[..., None] * ones_vec,
            )
        )
        mask_upper_separate_combined = np.logical_and.reduce(
            (
                mask_upper_separate,
                apply_cut[..., None] * ones_vec,
                ~same_cut_all_detectors[..., None] * ones_vec,
            )
        )
        # Sample f_cut from [f_min, f_max_lower_cut] and/or [f_min_upper_cut, f_max] in UFD for each detector
        if isinstance(self.domain, UniformFrequencyDomain):
            f_values_base_domain = self.domain.sample_frequencies[
                self.domain.frequency_mask
            ]
        elif isinstance(self.domain, MultibandedFrequencyDomain):
            f_values_base_domain = self.domain.base_domain.sample_frequencies[
                self.domain.base_domain.frequency_mask
            ]
        else:
            raise ValueError(f"Unknown domain type: {self.domain}")
        f_lower_separate = np.where(
            mask_lower_separate_combined,
            np.random.choice(
                f_values_base_domain[f_values_base_domain <= self.f_max_lower_cut],
                replace=True,
                size=batch_block_size,
            ),
            -1,
        )
        f_upper_separate = np.where(
            mask_upper_separate_combined,
            np.random.choice(
                f_values_base_domain[f_values_base_domain >= self.f_min_upper_cut],
                replace=True,
                size=batch_block_size,
            ),
            np.inf,
        )

        # Construct mask: f_cut_lower >= f_min_per_token and f_cut_upper <= f_max_per_token
        token_mask_separate_lower = (
            np.repeat(f_lower_separate, repeats=num_tokens_per_block, axis=-1)
            >= input_sample["position"][..., 0]
        )
        token_mask_separate_upper = (
            np.repeat(f_upper_separate, repeats=num_tokens_per_block, axis=-1)
            <= input_sample["position"][..., 1]
        )

        # Combine into one mask
        token_mask_separate = np.logical_or(
            token_mask_separate_lower, token_mask_separate_upper
        )

        # (2) Same cut is applied to all detectors
        # Decide whether to mask upper or lower or both
        lower_upper_both_same = np.random.choice(
            ["lower", "upper", "both"], p=self.p_lower_upper_both, size=batch_size
        )
        mask_lower_same = np.logical_or(
            lower_upper_both_same == "lower", lower_upper_both_same == "both"
        )
        mask_upper_same = np.logical_or(
            lower_upper_both_same == "upper", lower_upper_both_same == "both"
        )
        # Combine with masks (a) whether we apply a cut and (b) whether we apply it to all detectors
        mask_lower_combined = np.logical_and.reduce(
            (mask_lower_same, apply_cut, same_cut_all_detectors)
        )
        mask_upper_combined = np.logical_and.reduce(
            (mask_upper_same, apply_cut, same_cut_all_detectors)
        )
        # Sample f_cut from [f_min, f_max_lower_cut] and/or [f_min_upper_cut, f_max] in UFD
        f_lower_same = np.where(
            mask_lower_combined,
            np.random.choice(
                f_values_base_domain[f_values_base_domain <= self.f_max_lower_cut],
                replace=True,
                size=batch_size,
            ),
            -1,
        )
        f_upper_same = np.where(
            mask_upper_combined,
            np.random.choice(
                f_values_base_domain[f_values_base_domain >= self.f_min_upper_cut],
                replace=True,
                size=batch_size,
            ),
            np.inf,
        )
        # Construct mask: f_cut_lower >= f_min_per_token and f_cut_upper <= f_max_per_token
        # (Assume that all detectors have same f_min and f_max values)
        f_mins = input_sample["position"][..., 0:num_tokens_per_block, 0]
        f_maxs = input_sample["position"][..., 0:num_tokens_per_block, 1]
        token_mask_same_lower = f_lower_same[:, np.newaxis] >= f_mins
        token_mask_same_upper = f_upper_same[:, np.newaxis] <= f_maxs

        # Combine into one mask
        token_mask_same_one_detector = np.logical_or(
            token_mask_same_lower, token_mask_same_upper
        )
        # Duplicate for number of detectors
        token_mask_same = np.tile(token_mask_same_one_detector, reps=num_blocks)

        # Modify mask
        if len(input_sample["drop_token_mask"].shape) == 1:
            token_mask_separate = token_mask_separate.squeeze()
            token_mask_same = token_mask_same.squeeze()
        input_sample["drop_token_mask"] = np.logical_or.reduce(
            (input_sample["drop_token_mask"], token_mask_separate, token_mask_same)
        )

        return input_sample


class DropFrequencyInterval(object):
    """
    Randomly drop tokens corresponding to specific frequency interval.

    This transform does the following things:
    * Decides whether to mask a frequency interval per detector based on p_per_detector.
    * Samples f_lower from [f_min, f_max - max_width].
    * Samples f_upper from [f_lower, f_lower + max_width].
    * Converts f_lower and f_upper to tokens and creates a token mask removing all tokens in [f_lower, f_upper].
    """

    def __init__(
        self,
        domain: UniformFrequencyDomain | MultibandedFrequencyDomain,
        p_per_detector: float,
        f_min: float,
        f_max: float,
        max_width: float,
        print_output: bool = True,
    ):
        """
        Parameters
        ----------
        domain: UniformFrequencyDomain | MultibandedFrequencyDomain
            Domain corresponding to the data being transformed.
        p_per_detector: float
            Probability of dropping tokens (corresponding to a frequency interval) independently per detector.
        f_min: float
            Minimal frequency value for which we allow tokens to be dropped.
        f_max: float
            Maximum frequency value for which we allow tokens to be dropped.
        max_width: float
            Maximal width of frequency interval that can be dropped.
        print_output: bool
            Whether to write print statements to the console.
        """
        self.domain = domain
        self.p_per_detector = p_per_detector
        self.interval_f_min = f_min if domain.f_min < f_min else domain.f_min
        self.interval_f_max = f_max if domain.f_max > f_max else domain.f_max
        interval_width = self.interval_f_max - self.interval_f_min
        self.interval_max_width = (
            max_width if max_width < interval_width else interval_width
        )
        if print_output:
            print(
                f"Transform DropFrequencyInterval activated:"
                f"  Settings: \n"
                f"    - Probability of dropping interval per detector: {self.p_per_detector}\n"
                f"    - Interval range sampled from [{self.interval_f_min}, {self.interval_f_max}]\n"
                f"    - Maximal width of interval: {self.interval_max_width}, but the effective interval can be larger "
                f"if {self.interval_f_min} and {self.interval_f_max} fall in the middle of a token."
            )

    def __call__(self, input_sample: dict) -> dict:
        """
        Parameters
        ----------
        input_sample: Dict
            Values for keys
            - 'waveform':
                Sample of shape [batch_size, num_tokens, num_features]
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

        # Mask frequency range:
        # - Decide whether to apply a mask for each detector
        # - Sample f_mask_lower and f_mask_upper in uniform frequency domain
        # - Get tokens corresponding to frequency values
        # - Mask everything in between, i.e. [f_mask_lower, f_mask_upper]

        batch_block_size = (
            [*blocks.shape[:-1], num_blocks]
            if blocks.shape[:-1] != ()
            else [1, num_blocks]
        )
        # Decide whether to cut or mask frequency range for each block
        mask_interval = np.random.choice(
            [True, False],
            p=[self.p_per_detector, 1 - self.p_per_detector],
            size=batch_block_size,
        )

        # Sample f_lower and f_upper in UFD
        if isinstance(self.domain, UniformFrequencyDomain):
            f_values_base_domain = self.domain.sample_frequencies[
                self.domain.frequency_mask
            ]
        elif isinstance(self.domain, MultibandedFrequencyDomain):
            f_values_base_domain = self.domain.base_domain.sample_frequencies[
                self.domain.base_domain.frequency_mask
            ]
        else:
            raise ValueError(f"Unknown domain type: {self.domain}")
        # f_lower from [interval_f_min, interval_f_max - interval_max_width]
        mask_f_vals_lower = np.logical_and(
            self.interval_f_min <= f_values_base_domain,
            f_values_base_domain <= self.interval_f_max - self.interval_max_width,
        )
        possible_f_vals_lower = f_values_base_domain[mask_f_vals_lower]
        f_lower_full = np.random.choice(
            possible_f_vals_lower, replace=True, size=batch_block_size
        )
        f_lower = np.where(mask_interval, f_lower_full, np.inf)

        # f_upper from [f_lower, f_lower + interval_max_width]
        # Sampling f_upper is more complicated because it depends on the f_lower sampled for each batch index and
        # detector
        mask_f_vals_upper = np.logical_and(
            f_lower_full[:, :, np.newaxis]
            <= f_values_base_domain[np.newaxis, np.newaxis, :],
            f_values_base_domain[np.newaxis, np.newaxis, :]
            <= f_lower_full[:, :, np.newaxis] + self.interval_max_width,
        )
        possible_indices_upper = np.stack(
            [
                np.apply_along_axis(
                    np.argwhere, arr=mask_f_vals_upper[:, b, :], axis=-1
                ).squeeze()
                for b in range(num_blocks)
            ],
            axis=-2,
        )
        possible_f_vals_upper = f_values_base_domain[possible_indices_upper]
        f_upper_no_mask = np.stack(
            [
                np.apply_along_axis(
                    np.random.choice, arr=possible_f_vals_upper[..., b, :], axis=-1
                )
                for b in range(num_blocks)
            ],
            axis=-1,
        )
        f_upper = np.where(mask_interval, f_upper_no_mask, -1.0)

        # Construct mask: f_lower <= f_maxs AND f_upper >= f_mins
        f_mins = input_sample["position"][..., 0]
        f_maxs = input_sample["position"][..., 1]
        token_mask_lower = (
            np.repeat(f_lower, repeats=num_tokens_per_block, axis=-1) <= f_maxs
        )
        token_mask_upper = (
            np.repeat(f_upper, repeats=num_tokens_per_block, axis=-1) >= f_mins
        )

        # Combine into one mask
        token_mask = np.logical_and(token_mask_lower, token_mask_upper)

        # Modify mask
        if len(input_sample["drop_token_mask"].shape) == 1:
            token_mask = token_mask.squeeze()
        input_sample["drop_token_mask"] = np.logical_or(
            input_sample["drop_token_mask"], token_mask
        )

        return input_sample


class DropRandomTokens(object):
    """
    Randomly drop tokens for data points. Whether tokens will be dropped depends on the drop probability p_drop.
    The number of tokens that will be dropped is sampled uniformly from [1, max_num_tokens], disregarding any domain
    information.
    """

    def __init__(
        self,
        p_drop: float,
        max_num_tokens: int,
        print_output: bool = True,
    ):
        """
        Parameters
        ----------
        p_drop: float
            Probability of dropping tokens from a data point.
        max_num_tokens: int
            Maximum number of tokens that can be dropped.
        print_output: bool
            Whether to write print statements to the console.
        """
        self.p_drop = p_drop
        self.max_num_tokens = max_num_tokens
        if print_output:
            print(
                f"Transform DropRandomTokens activated:"
                f"  Settings: \n"
                f"    - Probability of dropping tokens for each data point: {self.p_drop}\n"
                f"    - Maximal number of tokens that can be dropped: {self.max_num_tokens}\n."
            )

    def __call__(self, input_sample: dict) -> dict:
        """
        Parameters
        ----------
        input_sample: Dict
            Values for keys
            - 'waveform':
            Sample of shape [batch_size, num_tokens, num_features]
            - 'position', shape [batch_size, num_tokens, 3]
               contains information [f_min, f_max, block]
            - 'drop_token_mask', shape [batch_size, num_tokens]

        Returns
        ----------
        sample: Dict
            input_sample with modified value for key
            - 'position', shape [batch_size, num_tokens, 3]

        """
        sample_without_channel = input_sample["waveform"][..., 0]
        num_tokens = sample_without_channel.shape[-1]

        batch_size = (
            [*sample_without_channel.shape[:-1]]
            if sample_without_channel.shape[:-1] != ()
            else [1]
        )
        drop_mask = np.random.choice(
            [True, False],
            p=[self.p_drop, 1 - self.p_drop],
            replace=True,
            size=batch_size,
        )
        num_tokens_to_drop = np.random.choice(
            np.arange(1, self.max_num_tokens + 1), size=batch_size
        )

        batch_token_size = (
            [*sample_without_channel.shape]
            if sample_without_channel.shape[:-1] != ()
            else [1, num_tokens]
        )
        # Generate random values for all tokens
        random_scores = np.random.uniform(size=batch_token_size)
        # Sort the scores in ascending order, and get indices
        sorted_indices = np.argsort(random_scores, axis=-1)
        # Create an index mask for selecting top-k per row
        row_indices = np.arange(batch_size[0])[:, np.newaxis]
        token_ranks = np.arange(num_tokens)
        # For each row, get threshold index
        thresholds = num_tokens_to_drop[:, np.newaxis] > token_ranks
        # Build boolean mask
        token_mask = np.zeros(batch_token_size, dtype=bool)
        token_mask[row_indices, sorted_indices] = thresholds

        # Combine masks
        token_mask = np.logical_and(
            np.repeat(drop_mask[..., np.newaxis], repeats=num_tokens, axis=-1),
            token_mask,
        )

        # Modify mask
        if len(input_sample["drop_token_mask"].shape) == 1:
            token_mask = token_mask.squeeze()
        input_sample["drop_token_mask"] = np.logical_or(
            input_sample["drop_token_mask"], token_mask
        )

        return input_sample


class NormalizePosition(object):
    """
    Normalize f_min and f_max in position
    """

    def __call__(self, input_sample: dict) -> dict:
        """
        Parameters
        ----------
        input_sample: Dict
            Values for keys
            - 'waveform':
            Sample of shape [batch_size, num_tokens, num_features]
            - 'position', shape [batch_size, num_tokens, 3]
               contains information [f_min, f_max, block]
            - 'drop_token_mask', shape [batch_size, num_tokens]

        Returns
        ----------
        sample: Dict
            input_sample with modified value for key
            - 'position', shape [batch_size, num_tokens, 3]

        """
        position = input_sample["position"]
        f_min = position[..., 0].min()
        f_max = position[..., 1].max()
        position[..., 0] = (position[..., 0] - f_min) / (f_max - f_min)
        position[..., 1] = (position[..., 1] - f_min) / (f_max - f_min)
        input_sample["position"] = position

        return input_sample


class UpdateFrequencyRange(object):
    """
    Update token mask according to frequency range update
    """

    def __init__(
        self,
        minimum_frequency: Optional[float | dict[str, float]] = None,
        maximum_frequency: Optional[float | dict[str, float]] = None,
        suppress_range: Optional[
            list[float, float] | dict[str, list[float, float]]
        ] = None,
        domain: Optional[UniformFrequencyDomain | MultibandedFrequencyDomain] = None,
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
            Suppress ranges [f_min, f_max], either for all detectors or for individual detectors.
        domain: UniformFrequencyDomain | MultibandedFrequencyDomain
        ifos: list[str]
            List of detectors.
        print_output: bool
            Whether to write print statements to the console.
        """
        # Include defaults in case of missing minimum-/maximum frequency values per detector
        self.minimum_frequency = add_defaults_for_missing_ifos(
            object_to_update=minimum_frequency, update_value=domain.f_min, ifos=ifos
        )
        self.maximum_frequency = add_defaults_for_missing_ifos(
            object_to_update=maximum_frequency, update_value=domain.f_max, ifos=ifos
        )
        self.suppress_range = suppress_range

        if print_output:
            print(
                f"Transform UpdateFrequencyRange activated:"
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
            - 'waveform':
            Sample of shape [batch_size, num_tokens, num_features]
            - 'position', shape [batch_size, num_tokens, 3]
               contains information [f_min, f_max, block]
            - 'drop_token_mask', shape [batch_size, num_tokens]

        Returns
        ----------
        sample: Dict
            input_sample with modified value for key
            - 'drop_token_mask', shape [batch_size, num_tokens]

        """
        sample = input_sample.copy()
        blocks = np.unique(sample["position"][..., 2])
        num_blocks = len(blocks)
        num_tokens_per_block = sample["position"].shape[-2] // num_blocks

        # Assume that f_min is the same for all detectors
        f_min_per_token = sample["position"][..., 0]
        f_max_per_token = sample["position"][..., 1]
        f_min_per_token_single = f_min_per_token[:num_tokens_per_block]
        f_max_per_token_single = f_max_per_token[:num_tokens_per_block]

        mask = np.zeros_like(sample["drop_token_mask"], dtype=bool)
        # Update minimum_frequency
        if self.minimum_frequency is not None:
            # Same for all detectors
            if isinstance(self.minimum_frequency, float):
                mask_min = np.where(
                    f_min_per_token <= self.minimum_frequency, True, False
                )
                mask = np.logical_or(mask, mask_min)
            # Different for each detector
            elif isinstance(self.minimum_frequency, dict):
                for b in blocks:
                    if DETECTOR_DICT_INVERSE[b] in self.minimum_frequency:
                        mask_min = np.where(
                            f_min_per_token_single
                            <= self.minimum_frequency[DETECTOR_DICT_INVERSE[b]],
                            True,
                            False,
                        )
                        mask_b = np.where(sample["position"][..., 2] == b, True, False)
                        mask[mask_b] = np.logical_or(mask_min, mask[mask_b])

        # Update maximum_frequency
        if self.maximum_frequency is not None:
            # Same for all detectors
            if isinstance(self.maximum_frequency, float):
                mask_max = np.where(
                    f_max_per_token >= self.maximum_frequency, True, False
                )
                mask = np.logical_or(mask, mask_max)
            # Different for each detector
            elif isinstance(self.maximum_frequency, dict):
                for b in blocks:
                    if DETECTOR_DICT_INVERSE[b] in self.maximum_frequency:
                        mask_max = np.where(
                            f_max_per_token_single
                            >= self.maximum_frequency[DETECTOR_DICT_INVERSE[b]],
                            True,
                            False,
                        )
                        mask_b = np.where(sample["position"][..., 2] == b, True, False)
                        mask[mask_b] = np.logical_or(mask_max, mask[mask_b])

        # Update suppress_range
        if self.suppress_range is not None:
            # Same for all detectors
            if isinstance(self.suppress_range, list):
                f_min_lower, f_max_upper = self.suppress_range
                mask_lower = np.where(f_max_per_token >= f_min_lower, True, False)
                mask_upper = np.where(f_min_per_token <= f_max_upper, True, False)
                mask_interval = np.logical_and(mask_lower, mask_upper)
                mask = np.logical_or(mask, mask_interval)
            # Different for each detector
            elif isinstance(self.suppress_range, dict):
                for b in blocks:
                    if DETECTOR_DICT_INVERSE[b] in self.suppress_range:
                        f_min_lower, f_max_upper = self.suppress_range[
                            DETECTOR_DICT_INVERSE[b]
                        ]
                        mask_lower = np.where(
                            f_max_per_token_single >= f_min_lower, True, False
                        )
                        mask_upper = np.where(
                            f_min_per_token_single <= f_max_upper, True, False
                        )
                        mask_interval = np.logical_and(mask_lower, mask_upper)
                        mask_b = np.where(sample["position"][..., 2] == b, True, False)
                        mask[mask_b] = np.logical_or(mask_interval, mask[mask_b])

        # Update drop_token_mask
        sample["drop_token_mask"] = np.logical_or(mask, sample["drop_token_mask"])

        return sample
