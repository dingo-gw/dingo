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


class MaskRandomTokens:
    """
    Randomly mask tokens by setting entries in ``token_mask`` to True.

    Applied after StrainTokenization to implement token-level masking during
    training and validation. Each token is masked independently with probability
    ``mask_probability``.

    Operates on numpy arrays and must therefore be placed before ToTorch in the
    transform chain.
    """

    def __init__(self, mask_probability: float):
        """
        Parameters
        ----------
        mask_probability:
            Probability in [0, 1) that each token is masked out.
        """
        if not 0.0 <= mask_probability < 1.0:
            raise ValueError("mask_probability must be in [0, 1).")
        self.mask_probability = mask_probability

    def __call__(self, sample: dict) -> dict:
        sample = sample.copy()
        mask = sample["token_mask"]
        random_mask = np.random.random(mask.shape) < self.mask_probability
        sample["token_mask"] = mask | random_mask
        return sample


class MaskDetectors:
    """
    Randomly mask entire detectors by setting all their tokens in ``token_mask``
    to True.

    Applied after StrainTokenization. Each detector is masked independently with
    probability ``mask_probability``. The detector assignment is read from
    ``position[..., 2]``, so no explicit knowledge of the detector list is required.

    Operates on numpy arrays and must therefore be placed before ToTorch in the
    transform chain.
    """

    def __init__(self, mask_probability: float):
        """
        Parameters
        ----------
        mask_probability:
            Probability in [0, 1) that each detector is masked out.
        """
        if not 0.0 <= mask_probability < 1.0:
            raise ValueError("mask_probability must be in [0, 1).")
        self.mask_probability = mask_probability

    def __call__(self, sample: dict) -> dict:
        sample = sample.copy()
        position = sample["position"]  # [..., num_tokens, 3]
        mask = sample["token_mask"].copy()  # [..., num_tokens] bool

        detector_indices = position[..., 2]  # [..., num_tokens]
        unique_detectors = np.unique(detector_indices)
        batch_shape = mask.shape[:-1]

        for det_idx in unique_detectors:
            token_is_this_det = detector_indices == det_idx  # [..., num_tokens]
            if batch_shape:
                drop_this = (np.random.random(batch_shape) < self.mask_probability)[
                    ..., np.newaxis
                ]  # [..., 1] bool
            else:
                drop_this = np.random.random() < self.mask_probability  # bool scalar
            mask = mask | (token_is_this_det & drop_this)

        sample["token_mask"] = mask
        return sample


class MaskRandomFrequencyRange:
    """
    Randomly mask tokens outside a sampled frequency range by setting their
    ``token_mask`` entries to True.

    The token-level analogue of ``CropMaskStrainRandom``. At each call, lower and upper
    frequency boundaries are sampled at token granularity from ``position[..., 0/1]``,
    and all tokens outside the resulting range are masked.

    Operates on numpy arrays and must therefore be placed before ToTorch in the
    transform chain.
    """

    def __init__(
        self,
        f_min_upper: Optional[float] = None,
        f_max_lower: Optional[float] = None,
        mask_probability: float = 1.0,
        independent_detectors: bool = True,
        independent_lower_upper: bool = True,
    ):
        """
        Parameters
        ----------
        f_min_upper:
            New f_min is sampled uniformly in token space over tokens with
            f_min ≤ f_min_upper. Defaults to no lower-boundary masking.
        f_max_lower:
            New f_max is sampled uniformly in token space over tokens with
            f_max ≥ f_max_lower. Defaults to no upper-boundary masking.
        mask_probability:
            Probability in [0, 1] that masking is applied to a given boundary.
        independent_detectors:
            If True, frequency boundaries are sampled independently per detector.
        independent_lower_upper:
            If True, ``mask_probability`` is applied independently to the lower
            and upper boundaries. If False, both boundaries are masked or kept
            together.
        """
        if not 0.0 <= mask_probability <= 1.0:
            raise ValueError("mask_probability must be in [0, 1].")
        if f_min_upper is not None and f_max_lower is not None:
            if f_min_upper >= f_max_lower:
                raise ValueError("f_min_upper must be less than f_max_lower.")
        self.f_min_upper = f_min_upper
        self.f_max_lower = f_max_lower
        self.mask_probability = mask_probability
        self.independent_detectors = independent_detectors
        self.independent_lower_upper = independent_lower_upper

    def _get_token_index_bounds(
        self, token_f_mins: np.ndarray, token_f_maxs: np.ndarray
    ):
        """
        Return ``(i_lower_max, i_upper_min)``: the inclusive token-index limits for
        sampling the lower and upper frequency boundaries.

        ``i_lower_max``: the highest token index allowed for the lower boundary
        (controlled by ``f_min_upper``). ``i_lower`` is sampled from
        ``[0, i_lower_max]``.

        ``i_upper_min``: the lowest token index allowed for the upper boundary
        (controlled by ``f_max_lower``). ``i_upper`` is sampled from
        ``[i_upper_min, T-1]``.
        """
        T = len(token_f_mins)
        if self.f_min_upper is not None:
            valid = np.where(token_f_mins <= self.f_min_upper)[0]
            i_lower_max = int(valid[-1]) if len(valid) > 0 else 0
        else:
            i_lower_max = 0  # no lower-boundary masking

        if self.f_max_lower is not None:
            valid = np.where(token_f_maxs >= self.f_max_lower)[0]
            i_upper_min = int(valid[0]) if len(valid) > 0 else T - 1
        else:
            i_upper_min = T - 1  # no upper-boundary masking

        return i_lower_max, i_upper_min

    def __call__(self, sample: dict) -> dict:
        sample = sample.copy()
        position = sample["position"]
        mask = sample["token_mask"].copy()

        unique_detectors = np.unique(position[..., 2])
        num_dets = len(unique_detectors)
        batch_shape = mask.shape[:-1]
        num_tokens = mask.shape[-1]
        T = num_tokens // num_dets

        if batch_shape:
            tok_f_mins = position.reshape(-1, num_tokens, 3)[0, :T, 0]
            tok_f_maxs = position.reshape(-1, num_tokens, 3)[0, :T, 1]
        else:
            tok_f_mins = position[:T, 0]
            tok_f_maxs = position[:T, 1]
        i_lower_max, i_upper_min = self._get_token_index_bounds(tok_f_mins, tok_f_maxs)

        within_det = np.tile(np.arange(T), num_dets)
        det_block = np.repeat(np.arange(num_dets), T)

        det_dim = (num_dets,) if self.independent_detectors else ()
        shape = batch_shape + det_dim

        if shape:
            i_lower = np.random.randint(0, i_lower_max + 1, size=shape)
            i_upper = np.random.randint(i_upper_min, T, size=shape)
        else:
            i_lower = int(np.random.randint(0, i_lower_max + 1))
            i_upper = int(np.random.randint(i_upper_min, T))

        if self.mask_probability < 1.0:
            if shape:
                apply = np.random.uniform(size=shape) <= self.mask_probability
                i_lower = np.where(apply, i_lower, 0)
                if self.independent_lower_upper:
                    apply = np.random.uniform(size=shape) <= self.mask_probability
                i_upper = np.where(apply, i_upper, T - 1)
            else:
                apply = np.random.uniform() <= self.mask_probability
                if not apply:
                    i_lower = 0
                if self.independent_lower_upper:
                    apply = np.random.uniform() <= self.mask_probability
                if not apply:
                    i_upper = T - 1

        if self.independent_detectors:
            i_lower_tok = i_lower[..., det_block]
            i_upper_tok = i_upper[..., det_block]
        else:
            i_lower_tok = i_lower[..., np.newaxis] if batch_shape else i_lower
            i_upper_tok = i_upper[..., np.newaxis] if batch_shape else i_upper

        sample["token_mask"] = (
            mask | (within_det < i_lower_tok) | (within_det > i_upper_tok)
        )
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
