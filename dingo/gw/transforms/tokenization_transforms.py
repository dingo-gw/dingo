from typing import Optional

import numpy as np

from dingo.gw.domains import UniformFrequencyDomain, MultibandedFrequencyDomain
from dingo.gw.gwutils import add_defaults_for_missing_detectors

DETECTOR_DICT = {"H1": 0, "L1": 1, "V1": 2}
DETECTOR_DICT_INVERSE = {v: k for k, v in DETECTOR_DICT.items()}


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
            self.num_padded_f_bins = (
                num_tokens_per_block * self.num_bins_per_token - num_f
            )

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
                f"  drop last token:        {self.drop_last_token}"
            )
            widths = np.diff(self.f_min_per_token)
            if len(widths):
                print(
                    f"  first token width:      {widths[0]:.3f} Hz\n"
                    f"  last token width:       {widths[-1]:.3f} Hz"
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
        *batch_dims, num_detectors, num_channels, _ = strain.shape

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
            num_detectors,
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
            num_detectors * self.num_tokens_per_detector,
            num_channels * self.num_bins_per_token,
        )

        # Position: [f_min, f_max, detector_index] per token
        num_tokens = num_detectors * self.num_tokens_per_detector
        token_f_min = np.tile(self.f_min_per_token, num_detectors)
        token_f_max = np.tile(self.f_max_per_token, num_detectors)
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
               contains information [f_min, f_max, detector_index]
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
            [batch_size, num_detectors * num_tokens_per_detector, num_channels * num_bins_per_token]
            where num_detectors = number of detectors in GW use case,
            num_channels>=3 (real, imag, auxiliary channels, e.g. asd),
            and num_bins = number of frequency bins.
            - 'position', shape [batch_size, num_tokens, 3]
               contains information [f_min, f_max, detector_index]
            - 'token_mask', shape [batch_size, num_tokens]

        Returns
        ----------
        sample: Dict
            input_sample with modified value for key
            - 'token_mask', shape [batch_size, num_tokens]

        """
        detector_indices = input_sample["position"][..., 2]
        num_detectors = len(np.unique(detector_indices))
        detectors = np.unique(detector_indices)

        # Convert p_mask_hlv dict to list
        p_mask_hlv = [self.p_mask_hlv[k] for k in detectors]

        # Decide how many detectors to mask (either none, or one less than the number of detectors present)
        # for each element in batch_size
        mask_n_blocks = np.random.choice(
            [i for i in range(num_detectors)],
            p=self.p_mask_012_detectors,
            size=[*detector_indices.shape[:-1]],
        )
        if np.sum(mask_n_blocks) != 0:
            # Treat mask 1 vs. 2 detectors separately because which detectors to mask varies
            # with the number of detectors to mask
            for n in [i for i in np.unique(mask_n_blocks) if i > 0]:
                # Construct mask for which batch indices require updates
                mask_mod = mask_n_blocks == n
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
                    detector_indices[mask_mod].T == detectors_to_mask[:, 0], True, False
                ).T
                if detectors_to_mask.shape[-1] > 1:
                    # (2) Update mask to include masking of any further detector
                    for i in range(1, detectors_to_mask.shape[-1]):
                        mask_detectors_i = np.where(
                            detector_indices[mask_mod].T == detectors_to_mask[:, i],
                            True,
                            False,
                        ).T
                        mask_detectors = np.logical_or(mask_detectors_i, mask_detectors)
                # Keep mask=True from previous transforms with logical OR
                mask_detectors = np.logical_or(
                    input_sample["token_mask"][mask_mod], mask_detectors
                )
                # Update mask
                input_sample["token_mask"][mask_mod] = mask_detectors

        return input_sample


class MaskFrequencyRange(object):
    """
    Randomly mask tokens at the lower and/or upper frequency edges so that the network
    learns that f_min and f_max of the frequency range can vary.

    This transform does the following things:
    * Decides whether to apply masking to each element of the batch based on p_mask.
    * Decides whether to treat the detectors individually or apply the same mask to all detectors.
    * Decides whether to mask the upper or lower frequency end or both (potentially per detector).
    * Samples a boundary from [f_min, f_min_upper] and/or [f_max_lower, f_max] in UFD (potentially per detector).
    * Converts frequency values to tokens and creates a token mask removing the lower and/or upper
      frequency range (potentially per detector).
    """

    def __init__(
        self,
        domain: UniformFrequencyDomain | MultibandedFrequencyDomain,
        p_mask: float,
        f_min_upper: float,
        f_max_lower: float,
        p_same_all_detectors: float,
        p_lower_upper_both: Optional[list] = None,
        print_output: bool = True,
    ):
        """
        Parameters
        ----------
        domain: UniformFrequencyDomain | MultibandedFrequencyDomain
            Domain corresponding to the data being transformed.
        p_mask: float
            Probability of applying a mask to each element of the batch.
        f_min_upper: float
            Upper boundary of the lower masking region. The lower boundary is sampled from
            [f_min, f_min_upper] in UFD.
        f_max_lower: float
            Lower boundary of the upper masking region. The upper boundary is sampled from
            [f_max_lower, f_max] in UFD.
        p_same_all_detectors: float
            Probability of applying the same mask to all detectors.
        p_lower_upper_both: list[float]
            List of probabilities explaining with what probability we either mask at the lower, at the upper, or at both
            ends. Order: [p_lower, p_upper, p_both]
        print_output: bool
            Whether to write print statements to the console.
        """

        self.domain = domain
        self.p_mask = p_mask
        self.f_min_upper = f_min_upper
        self.f_max_lower = f_max_lower
        self.prevent_zero_information = (
            True if self.f_min_upper >= self.f_max_lower else False
        )
        self.p_same_all_detectors = p_same_all_detectors
        if p_lower_upper_both is None:
            p_lower_upper_both = np.array([0.4, 0.4, 0.2])
        self.p_lower_upper_both = p_lower_upper_both
        if not np.isclose(np.sum(self.p_lower_upper_both), 1.0, rtol=1e-6, atol=1e-12):
            raise ValueError(
                f"p_lower_upper_both {self.p_lower_upper_both} does not sum to 1. "
            )
        if print_output:
            print(
                f"Transform MaskFrequencyRange activated: \n"
                f"    - Probability of masking: {self.p_mask}\n"
                f"    - Lower boundary sampled from [{self.domain.f_min}, {self.f_min_upper}]\n"
                f"    - Upper boundary sampled from [{self.f_max_lower}, {self.domain.f_max}]\n"
                f"    - Probability to apply the same mask on all detectors: {self.p_same_all_detectors} "
            )
            if self.prevent_zero_information:
                print(
                    f"\n    - Preventing zero information is activated since [{self.domain.f_min}, {self.f_min_upper}]"
                    f"overlaps with [{self.f_max_lower}, {self.domain.f_max}] "
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
               contains information [f_min, f_max, detector_index]
            - 'token_mask', shape [batch_size, num_tokens]

        Returns
        ----------
        sample: Dict
            input_sample with modified value for key
            - 'token_mask', shape [batch_size, num_tokens]

        """
        num_tokens = input_sample["waveform"].shape[-2]
        detector_indices = input_sample["position"][..., 2]
        num_detectors = len(np.unique(detector_indices))
        num_tokens_per_detector = num_tokens // num_detectors

        # Mask in frequency domain, where we remove the upper, lower or both part(s),
        #     i.e. [f_min, f_lower], [f_upper, f_max], or both
        # - Decide whether to apply masking for each sample
        # - Decide whether to treat the detectors individually or apply the same mask to all detectors
        # - Decide whether to mask upper or lower range or both (potentially for each detector)
        # - Sample boundary from [f_min, f_min_upper] and/or [f_max_lower, f_max]
        #   in uniform frequency domain (potentially for each detector)
        # - Convert frequency values to token mask

        batch_size = (
            [*detector_indices.shape[:-1]] if detector_indices.shape[:-1] != () else [1]
        )
        # Decide whether to apply masking for each sample
        apply_cut = np.random.choice(
            [True, False], p=[self.p_mask, 1 - self.p_mask], size=batch_size
        )

        # Decide whether to treat the detectors individually or apply the same mask to all detectors
        same_cut_all_detectors = np.where(
            apply_cut,
            np.random.choice(
                [True, False],
                p=[self.p_same_all_detectors, 1 - self.p_same_all_detectors],
                size=batch_size,
            ),
            False,
        )
        batch_block_size = (
            [*detector_indices.shape[:-1], num_detectors]
            if detector_indices.shape[:-1] != ()
            else [1, num_detectors]
        )
        # (1) Different mask applied to every detector
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
        # Combine with masks (a) whether we apply masking and (b) whether we apply it to a single detector
        ones_vec = np.ones((1, num_detectors), dtype=bool)
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
        # Sample boundary from [f_min, f_min_upper] and/or [f_max_lower, f_max] in UFD for each detector
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
                f_values_base_domain[f_values_base_domain <= self.f_min_upper],
                replace=True,
                size=batch_block_size,
            ),
            -1,
        )
        f_upper_separate = np.where(
            mask_upper_separate_combined,
            np.random.choice(
                f_values_base_domain[f_values_base_domain >= self.f_max_lower],
                replace=True,
                size=batch_block_size,
            ),
            np.inf,
        )

        # Construct mask: f_lower >= f_min_per_token and f_upper <= f_max_per_token
        token_mask_separate_lower = (
            np.repeat(f_lower_separate, repeats=num_tokens_per_detector, axis=-1)
            >= input_sample["position"][..., 0]
        )
        token_mask_separate_upper = (
            np.repeat(f_upper_separate, repeats=num_tokens_per_detector, axis=-1)
            <= input_sample["position"][..., 1]
        )

        # Combine into one mask
        token_mask_separate = np.logical_or(
            token_mask_separate_lower, token_mask_separate_upper
        )
        if self.prevent_zero_information:
            # If all tokens are masked in one sample, only apply upper or lower mask
            replace_mask = np.where(
                np.sum(token_mask_separate, axis=-1) == num_tokens, True, False
            )
            repl_mask = np.repeat(
                replace_mask[..., np.newaxis], repeats=num_tokens, axis=-1
            )
            # Decide whether to choose lower or upper instead of both
            lower_upper_probs = self.p_lower_upper_both[:2] / np.sum(
                self.p_lower_upper_both[:2]
            )
            lower_upper_global = np.random.choice(
                ["lower", "upper"], p=lower_upper_probs, size=batch_size
            )
            mask_lower_separate_replace = np.where(
                lower_upper_global == "lower", True, False
            )
            mask_lower_sep_repl = np.repeat(
                mask_lower_separate_replace[..., np.newaxis],
                repeats=num_tokens,
                axis=-1,
            )
            # Create replace mask
            mask_combined_separate_replace = np.where(
                mask_lower_sep_repl,
                token_mask_separate_lower,
                token_mask_separate_upper,
            )
            # Combine with token_mask_separate
            token_mask_separate = np.where(
                repl_mask, mask_combined_separate_replace, token_mask_separate
            )

        # (2) Same mask applied to all detectors
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
        # Combine with masks (a) whether we apply masking and (b) whether we apply it to all detectors
        mask_lower_combined = np.logical_and.reduce(
            (mask_lower_same, apply_cut, same_cut_all_detectors)
        )
        mask_upper_combined = np.logical_and.reduce(
            (mask_upper_same, apply_cut, same_cut_all_detectors)
        )
        # Sample boundary from [f_min, f_min_upper] and/or [f_max_lower, f_max] in UFD
        f_lower_same = np.where(
            mask_lower_combined,
            np.random.choice(
                f_values_base_domain[f_values_base_domain <= self.f_min_upper],
                replace=True,
                size=batch_size,
            ),
            -1,
        )
        f_upper_same = np.where(
            mask_upper_combined,
            np.random.choice(
                f_values_base_domain[f_values_base_domain >= self.f_max_lower],
                replace=True,
                size=batch_size,
            ),
            np.inf,
        )
        # Construct mask: f_lower >= f_min_per_token and f_upper <= f_max_per_token
        # (Assume that all detectors have same f_min and f_max values)
        f_mins = input_sample["position"][..., 0:num_tokens_per_detector, 0]
        f_maxs = input_sample["position"][..., 0:num_tokens_per_detector, 1]
        token_mask_same_lower = f_lower_same[:, np.newaxis] >= f_mins
        token_mask_same_upper = f_upper_same[:, np.newaxis] <= f_maxs

        # Combine into one mask
        token_mask_same_one_detector = np.logical_or(
            token_mask_same_lower, token_mask_same_upper
        )
        if self.prevent_zero_information:
            # If all tokens are masked in one detector, only apply upper or lower mask
            replace_mask = np.where(
                np.sum(token_mask_same_one_detector, axis=-1)
                == num_tokens_per_detector,
                True,
                False,
            )
            repl_mask = np.repeat(
                replace_mask[..., np.newaxis], repeats=num_tokens_per_detector, axis=-1
            )
            # Decide whether to choose lower or upper instead of both
            lower_upper_probs = self.p_lower_upper_both[:2] / np.sum(
                self.p_lower_upper_both[:2]
            )
            lower_upper_global = np.random.choice(
                ["lower", "upper"], p=lower_upper_probs, size=batch_size
            )
            mask_lower_same_replace = np.where(
                lower_upper_global == "lower", True, False
            )
            mask_lower_same_repl = np.repeat(
                mask_lower_same_replace[..., np.newaxis],
                repeats=num_tokens_per_detector,
                axis=-1,
            )
            # Create replace mask
            mask_combined_same_replace = np.where(
                mask_lower_same_repl, token_mask_same_lower, token_mask_same_upper
            )
            # Combine with token_mask_same_one_detector
            token_mask_same_one_detector = np.where(
                repl_mask, mask_combined_same_replace, token_mask_same_one_detector
            )

        # Duplicate for number of detectors
        token_mask_same = np.tile(token_mask_same_one_detector, reps=num_detectors)

        # Modify mask
        if len(input_sample["token_mask"].shape) == 1:
            token_mask_separate = token_mask_separate.squeeze()
            token_mask_same = token_mask_same.squeeze()
        input_sample["token_mask"] = np.logical_or.reduce(
            (input_sample["token_mask"], token_mask_separate, token_mask_same)
        )

        return input_sample


class MaskFrequencyNotches(object):
    """
    Randomly mask tokens corresponding to a contiguous frequency notch per detector.

    This transform does the following things:
    * Decides whether to mask a frequency notch per detector based on p_per_detector.
    * Samples f_lower from [f_min, f_max - max_width].
    * Samples f_upper from [f_lower, f_lower + max_width].
    * Converts f_lower and f_upper to tokens and creates a token mask removing all tokens in [f_lower, f_upper].
    """

    def __init__(
        self,
        domain: UniformFrequencyDomain | MultibandedFrequencyDomain,
        p_per_detector: float,
        max_width: float,
        f_min: Optional[float] = None,
        f_max: Optional[float] = None,
        print_output: bool = True,
    ):
        """
        Parameters
        ----------
        domain: UniformFrequencyDomain | MultibandedFrequencyDomain
            Domain corresponding to the data being transformed.
        p_per_detector: float
            Probability of masking a frequency notch independently per detector.
        max_width: float
            Maximal width of the masked frequency notch.
        f_min: Optional[float]
            Minimal frequency value of the notch within which tokens can be masked.
            Defaults to the domain f_min; explicit values are clamped to the domain.
        f_max: Optional[float]
            Maximal frequency value of the notch within which tokens can be masked.
            Defaults to the domain f_max; explicit values are clamped to the domain.
        print_output: bool
            Whether to write print statements to the console.
        """
        self.domain = domain
        self.p_per_detector = p_per_detector
        self.notch_f_min = domain.f_min if f_min is None else max(f_min, domain.f_min)
        self.notch_f_max = domain.f_max if f_max is None else min(f_max, domain.f_max)
        self.notch_max_width = min(max_width, self.notch_f_max - self.notch_f_min)
        if print_output:
            print(
                f"Transform MaskFrequencyNotches activated:\n"
                f"    - Probability of masking an notch per detector: {self.p_per_detector}\n"
                f"    - Notch range sampled from [{self.notch_f_min}, {self.notch_f_max}]\n"
                f"    - Maximal width of notch: {self.notch_max_width}, but the effective notch can be larger "
                f"if {self.notch_f_min} or {self.notch_f_max} fall in the middle of a token."
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
               contains information [f_min, f_max, detector_index]
            - 'token_mask', shape [batch_size, num_tokens]

        Returns
        ----------
        sample: Dict
            input_sample with modified value for key
            - 'token_mask', shape [batch_size, num_tokens]

        """
        num_tokens = input_sample["waveform"].shape[-2]
        detector_indices = input_sample["position"][..., 2]
        num_detectors = len(np.unique(detector_indices))
        num_tokens_per_detector = num_tokens // num_detectors

        # Mask frequency notch per detector:
        # - Decide whether to apply a mask for each detector
        # - Sample f_lower and f_upper from the base domain frequencies
        # - Mask all tokens whose frequency range overlaps [f_lower, f_upper]

        batch_block_size = (
            [*detector_indices.shape[:-1], num_detectors]
            if detector_indices.shape[:-1] != ()
            else [1, num_detectors]
        )
        # Decide whether to mask a frequency notch for each detector
        apply_notch = np.random.choice(
            [True, False],
            p=[self.p_per_detector, 1 - self.p_per_detector],
            size=batch_block_size,
        )

        # Sample f_lower and f_upper from the base domain frequencies
        base_domain = getattr(self.domain, "base_domain", self.domain)
        if not isinstance(base_domain, UniformFrequencyDomain):
            raise ValueError(f"Unknown domain type: {self.domain}")
        f_values_base_domain = base_domain.sample_frequencies[
            base_domain.frequency_mask
        ]
        # f_lower from [notch_f_min, notch_f_max - notch_max_width]
        mask_f_vals_lower = np.logical_and(
            self.notch_f_min <= f_values_base_domain,
            f_values_base_domain <= self.notch_f_max - self.notch_max_width,
        )
        possible_f_vals_lower = f_values_base_domain[mask_f_vals_lower]
        f_lower = np.random.choice(
            possible_f_vals_lower, replace=True, size=batch_block_size
        )
        # f_upper from [f_lower, f_lower + notch_max_width]: draw a number of
        # grid steps rather than collecting per-row candidate arrays, whose counts
        # differ by one for non-dyadic delta_f (float rounding) and cannot be stacked.
        delta_f = base_domain.delta_f
        n_steps = int(np.floor(self.notch_max_width / delta_f + 1e-9))
        f_upper = (
            f_lower + np.random.randint(0, n_steps + 1, size=batch_block_size) * delta_f
        )

        # Mask the tokens overlapping [f_lower, f_upper] on the detectors drawn for
        # a notch. Per-detector values are repeated over that detector's tokens.
        f_mins = input_sample["position"][..., 0]
        f_maxs = input_sample["position"][..., 1]
        rep = dict(repeats=num_tokens_per_detector, axis=-1)
        token_mask = (
            np.repeat(apply_notch, **rep)
            & (np.repeat(f_lower, **rep) <= f_maxs)
            & (np.repeat(f_upper, **rep) >= f_mins)
        )

        # Modify mask
        if len(input_sample["token_mask"].shape) == 1:
            token_mask = token_mask.squeeze()
        input_sample["token_mask"] = np.logical_or(
            input_sample["token_mask"], token_mask
        )

        return input_sample


class MaskTokensForFrequencyRangeUpdate(object):
    """
    Inference-time token-level counterpart to MaskDataForFrequencyRangeUpdate.

    Whereas MaskDataForFrequencyRangeUpdate sets the strain to zero and the ASD to one
    outside [minimum_frequency, maximum_frequency] (operating on raw frequency bins),
    this transform sets token_mask=True for any token that falls outside the updated
    range (operating on the tokenized representation).

    Both minimum_frequency and maximum_frequency can be set globally (float) or
    per-detector (dict). Missing detectors in a per-detector dict fall back to the
    domain default.
    """

    def __init__(
        self,
        domain: UniformFrequencyDomain | MultibandedFrequencyDomain,
        detectors: list[str],
        minimum_frequency: Optional[float | dict] = None,
        maximum_frequency: Optional[float | dict] = None,
        psd_notch_dict: Optional[dict] = None,
        print_output: bool = True,
    ):
        """
        Parameters
        ----------
        domain:
            Domain corresponding to the data being transformed.
        detectors:
            List of detector names (e.g. ["H1", "L1"]).
        minimum_frequency: float | dict | None
            New lower frequency bound. Float applies to all detectors; dict specifies
            per-detector values. Detectors missing from the dict use domain.f_min.
        maximum_frequency: float | dict | None
            New upper frequency bound. Float applies to all detectors; dict specifies
            per-detector values. Detectors missing from the dict use domain.f_max.
        psd_notch_dict: dict | None
            Per-detector interior frequency intervals to mask, e.g.
            ``{H1: [[50, 60]], L1: [[50, 60]]}``.  Each value is either a
            single ``[f_lo, f_hi]`` or a list of such pairs.  Tokens whose
            frequency range overlaps with any notch interval are masked.
        print_output:
            Whether to write a summary to stdout on construction.
        """
        self.minimum_frequency = add_defaults_for_missing_detectors(
            object_to_update=minimum_frequency,
            update_value=domain.f_min,
            detectors=detectors,
        )
        self.maximum_frequency = add_defaults_for_missing_detectors(
            object_to_update=maximum_frequency,
            update_value=domain.f_max,
            detectors=detectors,
        )
        self.psd_notch_dict = psd_notch_dict
        self.print_output = print_output
        if print_output:
            print(
                f"Transform MaskTokensForFrequencyRangeUpdate activated:\n"
                f"    - minimum_frequency: {self.minimum_frequency}\n"
                f"    - maximum_frequency: {self.maximum_frequency}\n"
                + (
                    f"    - psd_notch_dict: {self.psd_notch_dict}\n"
                    if self.psd_notch_dict
                    else ""
                )
            )

    def __call__(self, input_sample: dict) -> dict:
        """
        Parameters
        ----------
        input_sample: dict
            Must contain:
            - 'position', shape [num_tokens, 3],
               last dim = [f_min, f_max, detector_index]
            - 'token_mask', shape [num_tokens]

        Returns
        -------
        dict with 'token_mask' updated: tokens outside the new frequency range are
        set to True (masked out).
        """
        sample = input_sample.copy()
        detector_indices = np.unique(sample["position"][..., 2])
        num_detectors = len(detector_indices)
        num_tokens_per_detector = sample["position"].shape[-2] // num_detectors

        f_min_per_token = sample["position"][..., 0]
        f_max_per_token = sample["position"][..., 1]
        # All detectors share the same frequency grid; use the first detector's tokens
        # as the reference for per-detector masking.
        f_min_per_token_single = f_min_per_token[:num_tokens_per_detector]
        f_max_per_token_single = f_max_per_token[:num_tokens_per_detector]

        mask = np.zeros_like(sample["token_mask"], dtype=bool)

        if self.minimum_frequency is not None:
            if isinstance(self.minimum_frequency, (float, int)):
                mask = np.logical_or(
                    mask,
                    f_min_per_token < self.minimum_frequency,
                )
            elif isinstance(self.minimum_frequency, dict):
                for b in detector_indices:
                    det = DETECTOR_DICT_INVERSE[b]
                    if det in self.minimum_frequency:
                        mask_min = np.where(
                            f_min_per_token_single < self.minimum_frequency[det],
                            True,
                            False,
                        )
                        mask_b = sample["position"][..., 2] == b
                        mask[mask_b] = np.logical_or(mask_min, mask[mask_b])
            else:
                raise TypeError(
                    f"minimum_frequency must be float, int, or dict, "
                    f"got {type(self.minimum_frequency)}."
                )

        if self.maximum_frequency is not None:
            if isinstance(self.maximum_frequency, (float, int)):
                mask = np.logical_or(
                    mask,
                    f_max_per_token > self.maximum_frequency,
                )
            elif isinstance(self.maximum_frequency, dict):
                for b in detector_indices:
                    det = DETECTOR_DICT_INVERSE[b]
                    if det in self.maximum_frequency:
                        mask_max = np.where(
                            f_max_per_token_single > self.maximum_frequency[det],
                            True,
                            False,
                        )
                        mask_b = sample["position"][..., 2] == b
                        mask[mask_b] = np.logical_or(mask_max, mask[mask_b])
            else:
                raise TypeError(
                    f"maximum_frequency must be float, int, or dict, "
                    f"got {type(self.maximum_frequency)}."
                )

        if self.psd_notch_dict is not None:
            for b in detector_indices:
                det = DETECTOR_DICT_INVERSE[b]
                if det not in self.psd_notch_dict:
                    continue
                notch = self.psd_notch_dict[det]
                # Support single [f_lo, f_hi] or list of [[f_lo, f_hi], ...].
                if not isinstance(notch[0], (list, tuple)):
                    notch = [notch]
                mask_b = sample["position"][..., 2] == b
                for f_lo, f_hi in notch:
                    mask_notch = (f_max_per_token_single >= f_lo) & (
                        f_min_per_token_single <= f_hi
                    )
                    mask[mask_b] = np.logical_or(mask_notch, mask[mask_b])

        sample["token_mask"] = np.logical_or(mask, sample["token_mask"])
        return sample


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
