from typing import Optional

import numpy as np

from dingo.gw.domains import UniformFrequencyDomain, MultibandedFrequencyDomain

DETECTOR_DICT = {"H1": 0, "L1": 1, "V1": 2}


class StrainTokenization:
    """
    Divide strain frequency bins into fixed-size tokens and attach per-token position
    information (f_min, f_max, detector index).

    The input waveform is expected to have shape
        [..., num_detectors, num_channels, num_bins]
    where num_channels >= 1 (e.g. real, imaginary, ASD).

    The output contains:
    - 'waveform':        [..., num_detectors * num_tokens_per_detector,
                               num_channels * num_bins_per_token]
    - 'position':        [..., num_tokens, 3]
                         last dim = [f_min, f_max, detector_index]
    - 'token_mask': [..., num_tokens] bool, False = keep token
                         (PyTorch transformer convention: True = masked out).
    """

    def __init__(
        self,
        domain: UniformFrequencyDomain | MultibandedFrequencyDomain,
        num_tokens_per_block: Optional[int] = None,
        token_size: Optional[int] = None,
        drop_last_token: bool = False,
        print_output: bool = True,
    ):
        """
        Parameters
        ----------
        domain:
            Domain carrying f_min, f_max, delta_f, sample_frequencies.
        num_tokens_per_block:
            Number of tokens per detector. Mutually exclusive with token_size.
        token_size:
            Number of frequency bins per token. Mutually exclusive with
            num_tokens_per_block.
        drop_last_token:
            If True and the bins do not divide evenly, drop the trailing incomplete
            token. If False, pad it with zeros.
        print_output:
            Write a summary to stdout on construction.
        """
        if (num_tokens_per_block is None) == (token_size is None):
            raise ValueError(
                "Specify exactly one of num_tokens_per_block or token_size."
            )

        num_f = domain.frequency_mask_length

        if token_size is not None:
            self.num_bins_per_token = token_size
            n_full = num_f // token_size
            remainder = num_f % token_size
            num_tokens_per_block = (
                n_full
                if (drop_last_token and remainder)
                else (n_full if remainder == 0 else n_full + 1)
            )
        else:
            remainder = num_f % num_tokens_per_block
            # Ceiling ensures the given number of tokens covers the full frequency range.
            self.num_bins_per_token = int(np.ceil(num_f / num_tokens_per_block))
            if drop_last_token and remainder:
                num_tokens_per_block -= 1

        self.drop_last_token = drop_last_token
        self.num_tokens_per_detector = num_tokens_per_block

        # f_min / f_max for every token (same for all detectors)
        freqs = domain.sample_frequencies
        start = domain.min_idx
        self.f_min_per_token = freqs[start :: self.num_bins_per_token][
            :num_tokens_per_block
        ]
        self.f_max_per_token = freqs[
            start + self.num_bins_per_token - 1 :: self.num_bins_per_token
        ][:num_tokens_per_block]

        # Number of zero-padding bins needed in the last token
        self.num_padded_f_bins = 0
        if (
            len(self.f_min_per_token) > len(self.f_max_per_token)
            and not drop_last_token
        ):
            # Last token is incomplete: extrapolate f_max
            if isinstance(domain, MultibandedFrequencyDomain):
                last_delta_f = domain.delta_f[-1]
            else:
                last_delta_f = domain.delta_f
            f_max_pad = (
                self.f_max_per_token[-1] + self.num_bins_per_token * last_delta_f
            )
            self.f_max_per_token = np.append(self.f_max_per_token, f_max_pad)
            self.num_padded_f_bins = int((f_max_pad - freqs[-1]) / last_delta_f)

        if not (
            num_tokens_per_block
            == len(self.f_min_per_token)
            == len(self.f_max_per_token)
        ):
            raise ValueError(
                "f_min_per_token and f_max_per_token lengths do not match num_tokens_per_block."
            )

        if isinstance(domain, MultibandedFrequencyDomain):
            _check_mfd_node_compatibility(
                f_mins=self.f_min_per_token,
                f_maxs=self.f_max_per_token,
                mfd_nodes=domain.nodes,
                drop_last_token=drop_last_token,
            )

        if print_output:
            print(
                f"StrainTokenization:\n"
                f"  token_size:             {self.num_bins_per_token} bins\n"
                f"  tokens per detector:    {self.num_tokens_per_detector}\n"
                f"  drop last token:        {self.drop_last_token}\n"
                f"  first token width:      {self.f_min_per_token[1] - self.f_min_per_token[0]:.3f} Hz\n"
                f"  last token width:       {self.f_min_per_token[-1] - self.f_min_per_token[-2]:.3f} Hz"
            )
            if self.num_padded_f_bins > 0:
                print(f"  zero-padded bins in last token: {self.num_padded_f_bins}")

    def __call__(self, input_sample: dict) -> dict:
        """
        Parameters
        ----------
        input_sample:
            Must contain:
            - 'waveform': array of shape [..., num_detectors, num_channels, num_bins]
            - 'asds':     dict {detector_name: asd_array} used to read detector order

        Returns
        -------
        dict with keys 'waveform', 'position', 'token_mask' (see class docstring).
        """
        sample = input_sample.copy()
        strain = sample["waveform"]
        *batch_dims, num_blocks, num_channels, _ = strain.shape

        # (0) Cut or zero-pad the frequency axis to a multiple of num_bins_per_token
        target_bins = self.num_tokens_per_detector * self.num_bins_per_token
        if self.num_padded_f_bins == 0:
            strain = strain[..., :target_bins]
        else:
            pad = [(0, 0)] * (strain.ndim - 1) + [(0, self.num_padded_f_bins)]
            strain = np.pad(strain, pad, mode="constant")

        # (1) Split frequency axis into tokens:
        #     [..., D, C, F] → [..., D, C, T, P]
        strain = strain.reshape(
            *batch_dims,
            num_blocks,
            num_channels,
            self.num_tokens_per_detector,
            self.num_bins_per_token,
        )

        # (2) Move channels before tokens:
        #     [..., D, C, T, P] → [..., D, T, C, P]
        strain = np.moveaxis(strain, source=-2, destination=-3)

        # (3) Flatten block + token, and channel + bin into the final two axes:
        #     [..., D, T, C, P] → [..., D*T, C*P]
        sample["waveform"] = strain.reshape(
            *batch_dims,
            num_blocks * self.num_tokens_per_detector,
            num_channels * self.num_bins_per_token,
        )

        # Position: [f_min, f_max, detector_index] per token
        num_tokens = num_blocks * self.num_tokens_per_detector
        token_f_min = np.tile(self.f_min_per_token, num_blocks)
        token_f_max = np.tile(self.f_max_per_token, num_blocks)
        detector_indices = np.array(
            [DETECTOR_DICT[k] for k in input_sample["asds"]], dtype=strain.dtype
        )
        token_detector = np.repeat(detector_indices, self.num_tokens_per_detector)
        token_position = np.stack([token_f_min, token_f_max, token_detector], axis=-1)

        if batch_dims:
            token_position = np.broadcast_to(
                token_position, (*batch_dims, num_tokens, 3)
            ).copy()

        sample["position"] = token_position
        sample["token_mask"] = np.zeros((*batch_dims, num_tokens), dtype=bool)

        return sample


class MaskRandomTokens(object):
    """
    Randomly mask tokens for data points.

    For each data point, first decides whether to apply any masking at all based on p_mask, then samples the number
    of tokens to mask uniformly from [1, max_num_tokens]. The masked tokens are chosen at random, disregarding any
    domain information.
    """

    def __init__(
        self,
        p_mask: float,
        max_num_tokens: int,
        print_output: bool = True,
    ):
        """
        Parameters
        ----------
        p_mask: float
            Probability of masking tokens from a data point.
        max_num_tokens: int
            Maximum number of tokens that can be masked.
        print_output: bool
            Whether to write print statements to the console.
        """
        self.p_mask = p_mask
        self.max_num_tokens = max_num_tokens
        if print_output:
            print(
                f"Transform MaskRandomTokens activated:\n"
                f"    - Probability of masking tokens for each data point: {self.p_mask}\n"
                f"    - Maximal number of tokens that can be masked: {self.max_num_tokens}"
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
            - 'token_mask', shape [batch_size, num_tokens]

        Returns
        ----------
        sample: Dict
            input_sample with modified value for key
            - 'token_mask', shape [batch_size, num_tokens]

        """
        sample_without_channel = input_sample["waveform"][..., 0]
        num_tokens = sample_without_channel.shape[-1]

        batch_size = (
            [*sample_without_channel.shape[:-1]]
            if sample_without_channel.shape[:-1] != ()
            else [1]
        )
        probs = [self.p_mask, 1 - self.p_mask]
        apply_mask = np.random.choice(
            [True, False],
            p=probs,
            replace=True,
            size=batch_size,
        )
        num_tokens_to_mask = np.random.choice(
            np.arange(1, self.max_num_tokens + 1), size=batch_size
        )

        batch_token_size = (
            [*sample_without_channel.shape]
            if sample_without_channel.shape[:-1] != ()
            else [1, num_tokens]
        )
        # Generate random values for all tokens
        random_scores = np.random.uniform(size=batch_token_size)
        # Sort the scores in ascending order and get indices
        sorted_indices = np.argsort(random_scores, axis=-1)
        # Create an index mask for selecting top-k per row
        row_indices = np.arange(batch_size[0])[:, np.newaxis]
        token_ranks = np.arange(num_tokens)
        # For each row, get threshold index
        thresholds = num_tokens_to_mask[:, np.newaxis] > token_ranks
        # Build boolean mask
        token_mask = np.zeros(batch_token_size, dtype=bool)
        token_mask[row_indices, sorted_indices] = thresholds

        # Combine masks
        token_mask = np.logical_and(
            np.repeat(apply_mask[..., np.newaxis], repeats=num_tokens, axis=-1),
            token_mask,
        )

        # Modify mask
        if len(input_sample["token_mask"].shape) == 1:
            token_mask = token_mask.squeeze()
        input_sample["token_mask"] = np.logical_or(
            input_sample["token_mask"], token_mask
        )

        return input_sample


class MaskDetectors(object):
    """
    Randomly mask detectors.
    """

    def __init__(
        self,
        num_blocks: int,
        p_mask_012_detectors: list | None = None,
        p_mask_hlv: dict | None = None,
        print_output: bool = True,
    ):
        """
        Parameters
        ----------
        num_blocks: int
            Number of blocks (= detectors) in GW use case.
        p_mask_012_detectors: list[float]
            Specifies the categorical probability distribution for how many detectors to mask, in ascending order.
            example: [0.1, 0.6, 0.3] = [10% probability to mask 0 detectors (=3 detector setup), 60 % probability for
            2 detector setup, 30% probability for 1 detector setup]
        p_mask_hlv: dict
            Specifies the categorical probability distribution for which specific detectors to mask, order: H1, L1, V1.
            example: {'H1': 0.1, 'L1': 0.2, 'V1': 0.7} = 10 % probability to mask H1, 20 % probability to mask L1,
            70% probability to mask V1
        print_output: bool
            Whether to write print statements to the console.
        """
        self.num_blocks = num_blocks
        if p_mask_012_detectors is None:
            p_mask_012_detectors = [1 / num_blocks for _ in range(num_blocks)]
        if not np.isclose(np.sum(p_mask_012_detectors), 1.0, rtol=1e-6, atol=1e-12):
            raise ValueError(
                f"p_mask_012_detectors {p_mask_012_detectors} does not sum to 1."
            )
        self.p_mask_012_detectors = p_mask_012_detectors
        if p_mask_hlv is None:
            p_mask_hlv = {
                ["H1", "L1", "V1"][k]: 1 / num_blocks for k in range(num_blocks)
            }
        if not np.isclose(
            np.sum(list(p_mask_hlv.values())), 1.0, rtol=1e-6, atol=1e-12
        ):
            raise ValueError(f"p_mask_hlv {p_mask_hlv} does not sum to 1.")
        # Update keys equivalently to tokenization transform
        self.p_mask_hlv = {DETECTOR_DICT[k]: v for k, v in p_mask_hlv.items()}

        if len(p_mask_012_detectors) > num_blocks:
            raise ValueError(
                f"p_mask_012_detectors {self.p_mask_012_detectors} contains more options than"
                f"detectors available: {num_blocks}. You need to specify a categorical probability"
                f"value for masking 0, ..., {num_blocks - 1} detectors."
            )
        if len(self.p_mask_hlv) != num_blocks:
            raise ValueError(
                f"Provided values for p_mask_hlv={self.p_mask_hlv} is inconsistent with number of "
                f"detectors: {num_blocks}. You need to specify a categorical probability value for each "
                f"detector."
            )
        if print_output:
            print(
                f"Transform MaskDetectors activated: \n"
                f"    - Probabilities for masking {[i for i in range(num_blocks)]} detectors are "
                f"{self.p_mask_012_detectors}.\n"
                f"    - Probabilities for specific detectors are {self.p_mask_hlv}."
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
            - 'token_mask', shape [batch_size, num_tokens]

        Returns
        ----------
        sample: Dict
            input_sample with modified value for key
            - 'token_mask', shape [batch_size, num_tokens]

        """
        blocks = input_sample["position"][..., 2]
        num_blocks = len(np.unique(blocks))
        detectors = np.unique(blocks)

        # Convert p_mask_hlv dict to list
        p_mask_hlv = [self.p_mask_hlv[k] for k in detectors]

        # Decide how many detectors to mask (either none, or one less than the number of detectors present)
        # for each element in batch_size
        mask_n_blocks = np.random.choice(
            [i for i in range(num_blocks)],
            p=self.p_mask_012_detectors,
            size=[*blocks.shape[:-1]],
        )
        if np.sum(mask_n_blocks) != 0:
            # Treat mask 1 vs. 2 blocks separately because which detectors to mask varies
            # with the number of detectors to mask
            for n in [i for i in np.unique(mask_n_blocks) if i > 0]:
                # Construct mask for which batch indices require updates
                mask_mod = np.where(mask_n_blocks == n, True, False)
                # Decide which detectors
                detectors_to_mask = np.apply_along_axis(
                    np.random.choice,
                    axis=1,
                    arr=np.repeat(
                        np.expand_dims(detectors, 0), repeats=np.sum(mask_mod), axis=0
                    ),
                    p=p_mask_hlv,
                    size=n,
                    replace=False,
                )
                # Create mask such that tokens corresponding to masked detectors are True
                # (1) Mask one detector
                mask_detectors = np.where(
                    blocks[mask_mod].T == detectors_to_mask[:, 0], True, False
                ).T
                if detectors_to_mask.shape[-1] > 1:
                    # (2) Update mask to include masking of any further detector
                    for i in range(1, detectors_to_mask.shape[-1]):
                        mask_detectors_i = np.where(
                            blocks[mask_mod].T == detectors_to_mask[:, i], True, False
                        ).T
                        mask_detectors = np.logical_or(mask_detectors_i, mask_detectors)
                # Keep mask=True from previous transforms with logical OR
                mask_detectors = np.logical_or(
                    input_sample["token_mask"][mask_mod], mask_detectors
                )
                # Update mask
                input_sample["token_mask"][mask_mod] = mask_detectors

        return input_sample


def _check_mfd_node_compatibility(
    f_mins: np.ndarray,
    f_maxs: np.ndarray,
    mfd_nodes: np.ndarray,
    drop_last_token: bool,
) -> None:
    """
    Verify that every MFD node falls in a gap between consecutive tokens, not inside
    a token. This is required so that all bins within a token share the same delta_f.

    Each node must lie in (f_max[i-1], f_min[i]) for some i.
    """
    left_bounds = np.concatenate([[0], f_maxs[:-1]])
    right_bounds = f_mins
    intervals = np.stack([left_bounds, right_bounds], axis=1)

    covered = np.any(
        (mfd_nodes[:, None] >= intervals[:, 0])
        & (mfd_nodes[:, None] <= intervals[:, 1]),
        axis=1,
    )

    # The last node may lie beyond the last token's f_max when not dropping the last token
    if not covered[-1] and (mfd_nodes[~covered][0] > f_maxs[-1] or not drop_last_token):
        covered[-1] = True

    if not np.all(covered):
        raise ValueError(
            f"MFD nodes {mfd_nodes[~covered]} fall within a token rather than "
            f"between tokens. Adjust token_size or MFD nodes."
        )
