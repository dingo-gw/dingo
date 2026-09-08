from typing import Optional

import numpy as np

from dingo.gw.domains import UniformFrequencyDomain, MultibandedFrequencyDomain
from dingo.gw.gwutils import add_defaults_for_missing_detectors


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
                         last dim = [f_min, f_max, detector_index], where the
                         detector index is the detector's position in the
                         training detector list
    - 'token_mask': [..., num_tokens] bool, False = keep token
                         (PyTorch transformer convention: True = masked out).
    """

    def __init__(
        self,
        domain: UniformFrequencyDomain | MultibandedFrequencyDomain,
        detectors: list[str],
        num_tokens_per_block: Optional[int] = None,
        token_size: Optional[int] = None,
        drop_last_token: bool = False,
        training_detectors: Optional[list[str]] = None,
        print_output: bool = True,
    ):
        """
        Parameters
        ----------
        domain:
            Domain carrying f_min, f_max, delta_f, sample_frequencies.
        detectors:
            Detectors in the order of the waveform's detector blocks.
        num_tokens_per_block:
            Number of tokens per detector. Mutually exclusive with token_size.
        token_size:
            Number of frequency bins per token. Mutually exclusive with
            num_tokens_per_block.
        drop_last_token:
            If True and the bins do not divide evenly, drop the trailing incomplete
            token. If False, pad it with zeros.
        training_detectors:
            Detectors the network was trained with; the detector index of a token is
            the position of its detector in this list. Defaults to ``detectors``
            (training); at inference ``detectors`` may be a subset in any order.
        print_output:
            Write a summary to stdout on construction.
        """
        if (num_tokens_per_block is None) == (token_size is None):
            raise ValueError(
                "Specify exactly one of num_tokens_per_block or token_size."
            )
        if training_detectors is None:
            training_detectors = detectors
        unknown = [d for d in detectors if d not in training_detectors]
        if unknown:
            raise ValueError(
                f"Detectors {unknown} are not among the training detectors "
                f"{list(training_detectors)}."
            )
        self.detectors = list(detectors)
        self.detector_indices = np.array(
            [list(training_detectors).index(d) for d in detectors]
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
            - 'waveform': array of shape [..., num_detectors, num_channels, num_bins],
                          detector blocks in the order of ``self.detectors``

        Returns
        -------
        dict with keys 'waveform', 'position', 'token_mask' (see class docstring).
        """
        sample = input_sample.copy()
        strain = sample["waveform"]
        *batch_dims, num_detectors, num_channels, _ = strain.shape
        if num_detectors != len(self.detectors):
            raise ValueError(
                f"Expected {len(self.detectors)} detector blocks {self.detectors}, "
                f"got {num_detectors}."
            )

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
        token_detector = np.repeat(
            self.detector_indices.astype(strain.dtype), self.num_tokens_per_detector
        )
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
    Randomly mask whole detectors.

    For each sample, first draw how many detectors to mask from ``p_num_masked``,
    then draw which ones from ``p_detector`` without replacement.
    """

    def __init__(
        self,
        detectors: list[str],
        p_num_masked: list | None = None,
        p_detector: dict | None = None,
        print_output: bool = True,
    ):
        """
        Parameters
        ----------
        detectors: list[str]
            Training detectors; the detector index of a token is its position in
            this list.
        p_num_masked: list[float]
            Categorical distribution over the number of masked detectors, 0 to
            len(detectors) - 1. Example for three detectors: [0.6, 0.3, 0.1] = 60%
            mask none, 30% mask one, 10% mask two. Default: uniform.
        p_detector: dict
            Categorical distribution over which detector to mask, keyed by detector
            name. Example: {'H1': 0.3, 'L1': 0.3, 'V1': 0.4}. Default: uniform.
        print_output: bool
            Whether to write print statements to the console.
        """
        num_detectors = len(detectors)
        if p_num_masked is None:
            p_num_masked = [1 / num_detectors] * num_detectors
        if len(p_num_masked) != num_detectors:
            raise ValueError(
                f"p_num_masked {p_num_masked} needs one entry per number of masked "
                f"detectors, 0 to {num_detectors - 1}."
            )
        if not np.isclose(np.sum(p_num_masked), 1.0, rtol=1e-6, atol=1e-12):
            raise ValueError(f"p_num_masked {p_num_masked} does not sum to 1.")
        if p_detector is None:
            p_detector = {d: 1 / num_detectors for d in detectors}
        if set(p_detector) != set(detectors):
            raise ValueError(
                f"p_detector keys {sorted(p_detector)} do not match the detectors "
                f"{list(detectors)}."
            )
        if not np.isclose(
            np.sum(list(p_detector.values())), 1.0, rtol=1e-6, atol=1e-12
        ):
            raise ValueError(f"p_detector {p_detector} does not sum to 1.")
        self.detectors = list(detectors)
        self.p_num_masked = p_num_masked
        self.p_detector = p_detector
        # Same distribution indexed by the token detector index.
        self._p_detector_by_index = np.array([p_detector[d] for d in detectors])

        if print_output:
            print(
                f"Transform MaskDetectors activated: \n"
                f"    - Probabilities for masking 0, ..., {num_detectors - 1} detectors "
                f"are {self.p_num_masked}.\n"
                f"    - Probabilities for specific detectors are {self.p_detector}."
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
        detectors = np.unique(detector_indices)
        num_detectors = len(detectors)
        p_detector = self._p_detector_by_index[detectors.astype(int)]

        # Decide how many detectors to mask (either none, or one less than the number of detectors present)
        # for each element in batch_size
        mask_n_blocks = np.random.choice(
            [i for i in range(num_detectors)],
            p=self.p_num_masked,
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
                    p=p_detector,
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

    For each sample, with probability p_mask, cut the lower end, the upper end, or
    both (p_lower_upper_both). With probability p_same_all_detectors the same cut is
    applied to every detector; otherwise each detector gets its own draw. The lower
    boundary is drawn from the base-domain frequencies in [f_min, f_min_upper], the
    upper boundary from [f_max_lower, f_max], and every token overlapping the cut
    region is masked. A sample that would lose every token keeps only its lower or
    its upper cut instead.
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
        self.p_same_all_detectors = p_same_all_detectors
        if p_lower_upper_both is None:
            p_lower_upper_both = [0.4, 0.4, 0.2]
        self.p_lower_upper_both = np.asarray(p_lower_upper_both, dtype=float)
        if not np.isclose(np.sum(self.p_lower_upper_both), 1.0, rtol=1e-6, atol=1e-12):
            raise ValueError(
                f"p_lower_upper_both {self.p_lower_upper_both} does not sum to 1. "
            )
        # Boundaries are drawn from the base-domain grid (for a multibanded domain,
        # the uniform grid before decimation).
        base_domain = getattr(domain, "base_domain", domain)
        f_values = base_domain.sample_frequencies[base_domain.frequency_mask]
        self._lower_candidates = f_values[f_values <= f_min_upper]
        self._upper_candidates = f_values[f_values >= f_max_lower]
        if len(self._lower_candidates) == 0 or len(self._upper_candidates) == 0:
            raise ValueError(
                f"No base-domain frequencies in [{domain.f_min}, {f_min_upper}] or in "
                f"[{f_max_lower}, {domain.f_max}] to draw a cut boundary from."
            )
        if print_output:
            print(
                f"Transform MaskFrequencyRange activated: \n"
                f"    - Probability of masking: {self.p_mask}\n"
                f"    - Lower boundary sampled from [{self.domain.f_min}, {self.f_min_upper}]\n"
                f"    - Upper boundary sampled from [{self.f_max_lower}, {self.domain.f_max}]\n"
                f"    - Probability to apply the same mask on all detectors: {self.p_same_all_detectors} "
            )

    def __call__(self, input_sample: dict) -> dict:
        """
        Parameters
        ----------
        input_sample: dict
            Tokenized sample with 'position' and 'token_mask' (see StrainTokenization),
            with or without a leading batch dimension.

        Returns
        -------
        dict
            input_sample with the cut tokens set to True in 'token_mask'.
        """
        position = input_sample["position"]
        token_mask = input_sample["token_mask"]
        unbatched = token_mask.ndim == 1
        if unbatched:
            position, token_mask = position[None], token_mask[None]
        num_batch, num_tokens = token_mask.shape
        num_detectors = len(np.unique(position[..., 2]))
        num_tokens_per_detector = num_tokens // num_detectors
        p = self.p_lower_upper_both

        apply_cut = np.random.choice(
            [True, False], p=[self.p_mask, 1 - self.p_mask], size=[num_batch]
        )
        same_cut = np.where(
            apply_cut,
            np.random.choice(
                [True, False],
                p=[self.p_same_all_detectors, 1 - self.p_same_all_detectors],
                size=[num_batch],
            ),
            False,
        )[:, None]
        # Draw one cut per detector and one cut per sample, then pick per sample
        # according to same_cut. Both sets are always drawn, in this order, so the
        # random stream matches the earlier two-branch implementation.
        per_detector = [num_batch, num_detectors]
        ends = ["lower", "upper", "both"]
        which_sep = np.random.choice(ends, p=p, size=per_detector)
        f_lower_sep = np.random.choice(self._lower_candidates, size=per_detector)
        f_upper_sep = np.random.choice(self._upper_candidates, size=per_detector)
        which_same = np.random.choice(ends, p=p, size=[num_batch])
        f_lower_same = np.random.choice(self._lower_candidates, size=[num_batch])
        f_upper_same = np.random.choice(self._upper_candidates, size=[num_batch])
        which = np.where(same_cut, which_same[:, None], which_sep)
        f_lower = np.where(same_cut, f_lower_same[:, None], f_lower_sep)
        f_upper = np.where(same_cut, f_upper_same[:, None], f_upper_sep)
        cut_lower = apply_cut[:, None] & (which != "upper")
        cut_upper = apply_cut[:, None] & (which != "lower")

        # Mask the tokens overlapping [f_min, f_lower] or [f_upper, f_max] on each
        # detector; per-detector values are repeated over that detector's tokens.
        rep = dict(repeats=num_tokens_per_detector, axis=-1)
        mask_lower = (
            np.repeat(np.where(cut_lower, f_lower, -1), **rep) >= position[..., 0]
        )
        mask_upper = (
            np.repeat(np.where(cut_upper, f_upper, np.inf), **rep) <= position[..., 1]
        )
        mask = mask_lower | mask_upper

        # A sample that lost every token keeps only its lower or its upper cut.
        all_masked = mask.all(axis=-1)
        if all_masked.any():
            keep_lower = np.random.choice(
                [True, False], p=p[:2] / p[:2].sum(), size=all_masked.sum()
            )
            mask[all_masked] = np.where(
                keep_lower[:, None], mask_lower[all_masked], mask_upper[all_masked]
            )

        token_mask = token_mask | mask
        input_sample["token_mask"] = token_mask[0] if unbatched else token_mask
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


class NormalizePosition(object):
    """
    Rescale the frequency columns of the token positions from Hz to [0, 1] over a
    fixed reference interval [f_min, f_max] (the training domain's bounds). Must run
    after every mask transform, since those compare positions against frequencies
    in Hz. The detector column is left unchanged.
    """

    def __init__(self, f_min: float, f_max: float):
        self.f_min = f_min
        self.f_max = f_max

    def __call__(self, input_sample: dict) -> dict:
        """
        Parameters
        ----------
        input_sample: dict
            with key 'position', shape [..., num_tokens, 3], last dim
            [f_min, f_max, detector_index] in Hz.

        Returns
        -------
        dict
            input_sample with a new 'position' array whose first two columns are
            rescaled to [0, 1].
        """
        position = input_sample["position"]
        scaled = (position[..., :2] - self.f_min) / (self.f_max - self.f_min)
        input_sample["position"] = np.concatenate((scaled, position[..., 2:]), axis=-1)
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
        training_detectors: Optional[list[str]] = None,
        print_output: bool = True,
    ):
        """
        Parameters
        ----------
        domain:
            Domain corresponding to the data being transformed.
        detectors:
            Detectors present in the data (e.g. ["H1", "L1"]).
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
        training_detectors:
            Detectors the network was trained with; token detector indices are
            positions in this list. Defaults to ``detectors``.
        print_output:
            Whether to write a summary to stdout on construction.
        """
        self.training_detectors = list(
            detectors if training_detectors is None else training_detectors
        )
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
                    det = self.training_detectors[int(b)]
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
                    det = self.training_detectors[int(b)]
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
                det = self.training_detectors[int(b)]
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
