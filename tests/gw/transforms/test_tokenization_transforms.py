import numpy as np
import pytest

from dingo.gw.domains import UniformFrequencyDomain, MultibandedFrequencyDomain
from dingo.gw.transforms import (
    StrainTokenization,
    MaskRandomTokens,
    MaskDetectors,
    MaskFrequencyEdges,
    DETECTOR_DICT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_ufd(f_min=20.0, f_max=1024.0, T=8.0):
    return UniformFrequencyDomain(f_min=f_min, f_max=f_max, delta_f=1.0 / T)


def make_mfd(nodes=None, f_max=1038.0, T=8.0):
    if nodes is None:
        nodes = [20.0, 34.0, 46.0, 62.0, 78.0, 1038.0]
    base = UniformFrequencyDomain(f_min=nodes[0], f_max=f_max, delta_f=1.0 / T)
    return MultibandedFrequencyDomain(
        nodes=nodes, delta_f_initial=1.0 / T, base_domain=base
    )


def make_sample(domain, batch_size, num_channels=3):
    """Build a minimal {'waveform': ..., 'asds': ...} dict.

    H1 waveform is all zeros; L1 real channel is set to [1, 2, ..., num_f]
    so we can track ordering through the reshape.
    """
    num_f = domain.frequency_mask_length
    detectors = ["H1", "L1"]

    if batch_size is None:
        # No batch dimension: shape [num_blocks, num_channels, num_f]
        waveform_h1 = np.zeros([1, num_channels, num_f])
        waveform_l1 = np.ones([1, num_channels, num_f])
        waveform_l1[0, 0, :] = np.arange(1, num_f + 1)
        waveform = np.concatenate([waveform_h1, waveform_l1], axis=0)
        asds = {d: np.random.rand(num_f) for d in detectors}
    else:
        # Batched: shape [batch, num_blocks, num_channels, num_f]
        waveform_h1 = np.zeros([batch_size, 1, num_channels, num_f])
        waveform_l1 = np.ones([batch_size, 1, num_channels, num_f])
        waveform_l1[0, 0, 0, :] = np.arange(1, num_f + 1)
        waveform = np.concatenate([waveform_h1, waveform_l1], axis=1)
        asds = {d: np.random.rand(batch_size, num_f) for d in detectors}

    return {"waveform": waveform, "asds": asds}


def _check_position_and_mask(out, domain, num_tokens_per_block, num_blocks):
    """Shared checks for position and token_mask outputs."""
    num_tokens = num_tokens_per_block * num_blocks

    # position shape
    assert out["position"].shape[-2:] == (num_tokens, 3)

    # First token of each detector starts at domain.f_min
    for block in range(num_blocks):
        first_tok = block * num_tokens_per_block
        assert np.all(out["position"][..., first_tok, 0] == domain.f_min)

    # Last token of each detector reaches at least domain.f_max - delta_f
    if isinstance(domain, MultibandedFrequencyDomain):
        f_max_threshold = domain.f_max - domain.delta_f[-1]
    else:
        f_max_threshold = domain.f_max - domain.delta_f
    for block in range(num_blocks):
        last_tok = block * num_tokens_per_block + num_tokens_per_block - 1
        assert np.all(out["position"][..., last_tok, 1] >= f_max_threshold)

    # f_min increases monotonically within each detector's tokens
    for block in range(num_blocks):
        tok_slice = slice(
            block * num_tokens_per_block, (block + 1) * num_tokens_per_block
        )
        f_mins = out["position"][..., tok_slice, 0]
        # Take first batch element if batched
        f_mins_1d = f_mins.reshape(-1, num_tokens_per_block)[0]
        assert np.all(
            np.diff(f_mins_1d) > 0
        ), "f_min is not monotonically increasing within a detector"

    # Each detector's tokens share the same detector index, and indices differ across detectors
    unique_det_indices = set()
    for block in range(num_blocks):
        tok_slice = slice(
            block * num_tokens_per_block, (block + 1) * num_tokens_per_block
        )
        det_vals = np.unique(out["position"][..., tok_slice, 2])
        assert (
            len(det_vals) == 1
        ), "Tokens of a single detector have mixed detector indices"
        unique_det_indices.add(det_vals[0])
    assert (
        len(unique_det_indices) == num_blocks
    ), "Detector indices are not unique across blocks"

    # token_mask: shape and default all-False
    assert out["token_mask"].shape[-1] == num_tokens
    assert not out[
        "token_mask"
    ].any(), "Default mask should keep all tokens (all False)"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ufd_batch():
    domain = make_ufd()
    # frequency_mask_length = 8032 (f_min=20, f_max=1024, delta_f=0.125).
    # 8032 % 40 != 0, so these fixtures exercise the zero-padding path.
    # 8032 has no convenient power-of-2 divisor, so an exact-division UFD case
    # would require an artificially chosen domain; the MFD fixtures cover that.
    num_tokens_per_block = 40
    return domain, num_tokens_per_block, 2, make_sample(domain, batch_size=100)


@pytest.fixture
def ufd_no_batch():
    domain = make_ufd()
    num_tokens_per_block = 40  # non-divisible; see ufd_batch comment
    return domain, num_tokens_per_block, 2, make_sample(domain, batch_size=None)


@pytest.fixture
def ufd_single_batch():
    domain = make_ufd()
    num_tokens_per_block = 40  # non-divisible; see ufd_batch comment
    return domain, num_tokens_per_block, 2, make_sample(domain, batch_size=1)


@pytest.fixture
def mfd_batch():
    # 43 tokens fits exactly for these nodes with T=8
    domain = make_mfd()
    num_tokens_per_block = 43
    return domain, num_tokens_per_block, 2, make_sample(domain, batch_size=100)


@pytest.fixture
def mfd_drop_last_token():
    # Extend f_max slightly so the last token is incomplete
    domain = make_mfd(f_max=1040.0)
    num_tokens_per_block = 43
    return domain, num_tokens_per_block, 2, make_sample(domain, batch_size=100)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

SETUPS = ["ufd_batch", "ufd_no_batch", "ufd_single_batch", "mfd_batch"]


@pytest.mark.parametrize("setup", SETUPS)
def test_strain_tokenization_num_tokens(request, setup):
    """Basic tokenization using num_tokens_per_block: shapes and content."""
    domain, num_tokens_per_block, num_blocks, sample = request.getfixturevalue(setup)

    transform = StrainTokenization(
        domain, num_tokens_per_block=num_tokens_per_block, print_output=False
    )
    out = transform(sample)

    # Output waveform shape
    assert out["waveform"].shape[-2] == num_tokens_per_block * num_blocks
    num_features = out["waveform"].shape[-1]

    # L1 sits in the second half of the token sequence; real is the first 1/3 of features.
    # The linearly increasing values 1..num_f should be recoverable in order.
    num_f = domain.frequency_mask_length
    if out["waveform"].ndim == 2:
        l1_real = out["waveform"][num_tokens_per_block:, : num_features // 3]
    else:
        l1_real = out["waveform"][0, num_tokens_per_block:, : num_features // 3]
    assert np.all(l1_real.flatten()[:num_f] == np.arange(1, num_f + 1))

    _check_position_and_mask(out, domain, num_tokens_per_block, num_blocks)


@pytest.mark.parametrize("setup", SETUPS)
def test_strain_tokenization_token_size(request, setup):
    """Equivalent result when specifying token_size instead of num_tokens_per_block."""
    domain, num_tokens_per_block, num_blocks, sample = request.getfixturevalue(setup)
    token_size = int(np.ceil(domain.frequency_mask_length / num_tokens_per_block))

    transform = StrainTokenization(domain, token_size=token_size, print_output=False)
    out = transform(sample)

    assert out["waveform"].shape[-2] == num_tokens_per_block * num_blocks
    _check_position_and_mask(out, domain, num_tokens_per_block, num_blocks)


@pytest.mark.parametrize("setup", SETUPS + ["mfd_drop_last_token"])
def test_strain_tokenization_drop_last_token(request, setup):
    """drop_last_token removes the trailing incomplete token."""
    domain, num_tokens_per_block, num_blocks, sample = request.getfixturevalue(setup)
    token_size = int(np.ceil(domain.frequency_mask_length / num_tokens_per_block))
    remainder = domain.frequency_mask_length % num_tokens_per_block
    expected = (num_tokens_per_block - (1 if remainder else 0)) * num_blocks

    transform = StrainTokenization(
        domain, token_size=token_size, drop_last_token=True, print_output=False
    )
    out = transform(sample)

    assert out["waveform"].shape[-2] == expected


def test_token_bin_content():
    """Each token contains exactly the right frequency bins in order.

    Uses the MFD fixture (exact division, no padding) so token k of L1 contains
    bins [k*P, ..., (k+1)*P - 1] with values [k*P+1, ..., (k+1)*P].
    """
    domain = make_mfd()
    num_tokens_per_block = 43
    num_channels = 3
    P = domain.frequency_mask_length // num_tokens_per_block  # bins per token

    sample = make_sample(domain, batch_size=100, num_channels=num_channels)
    transform = StrainTokenization(
        domain, num_tokens_per_block=num_tokens_per_block, print_output=False
    )
    out = transform(sample)

    num_features = out["waveform"].shape[-1]
    real_width = num_features // num_channels  # = P

    # L1 tokens occupy indices [num_tokens_per_block, 2*num_tokens_per_block)
    # Real channel is the first `real_width` features within each token
    l1_real = out["waveform"][0, num_tokens_per_block:, :real_width]  # [T, P]

    for k in range(num_tokens_per_block):
        expected = np.arange(k * P + 1, (k + 1) * P + 1, dtype=float)
        assert np.allclose(
            l1_real[k], expected
        ), f"Token {k} has wrong bin values: got {l1_real[k]}, expected {expected}"


def test_three_detectors():
    """Detector-index assignment and token ordering with three detectors (H1, L1, V1).

    Uses MFD with exact division so no zero-padding obscures the value checks.
    """
    domain = make_mfd()  # frequency_mask_length=688, 688/16=43 exactly
    num_f = domain.frequency_mask_length
    num_channels = 3
    batch_size = 4
    detectors = ["H1", "L1", "V1"]

    waveforms = []
    for i, det in enumerate(detectors):
        w = np.full([batch_size, 1, num_channels, num_f], float(i))
        waveforms.append(w)
    waveform = np.concatenate(waveforms, axis=1)  # [B, 3, C, F]
    asds = {d: np.ones([batch_size, num_f]) for d in detectors}
    sample = {"waveform": waveform, "asds": asds}

    num_tokens_per_block = 43
    transform = StrainTokenization(
        domain, num_tokens_per_block=num_tokens_per_block, print_output=False
    )
    out = transform(sample)

    T = num_tokens_per_block
    for block_idx, det in enumerate(detectors):
        tok_slice = slice(block_idx * T, (block_idx + 1) * T)

        # All waveform values in this block's tokens equal float(block_idx)
        assert np.all(
            out["waveform"][0, tok_slice, :] == float(block_idx)
        ), f"Wrong waveform values for detector {det}"
        # Detector index in position matches DETECTOR_DICT
        det_indices = out["position"][0, tok_slice, 2]
        assert np.all(
            det_indices == DETECTOR_DICT[det]
        ), f"Wrong detector index for {det}: got {det_indices[0]}, expected {DETECTOR_DICT[det]}"


def test_output_dtype():
    """Output arrays preserve input dtype; token_mask is always bool."""
    domain = make_ufd()

    for dtype in (np.float32, np.float64):
        sample = make_sample(domain, batch_size=8)
        sample["waveform"] = sample["waveform"].astype(dtype)

        transform = StrainTokenization(
            domain, num_tokens_per_block=40, print_output=False
        )
        out = transform(sample)

        assert out["waveform"].dtype == dtype, f"waveform dtype changed from {dtype}"
        assert out["position"].dtype == dtype, f"position dtype changed from {dtype}"
        assert out["token_mask"].dtype == bool


def test_mutual_exclusivity():
    """Passing both or neither of num_tokens_per_block / token_size raises ValueError."""
    domain = make_ufd()
    with pytest.raises(ValueError):
        StrainTokenization(domain, print_output=False)
    with pytest.raises(ValueError):
        StrainTokenization(
            domain, num_tokens_per_block=10, token_size=20, print_output=False
        )


def test_mfd_incompatible_nodes():
    """MFD node inside a token should raise ValueError."""
    # nodes=[20, 34, ...]: with token_size=200, a node will land inside a token
    domain = make_mfd()
    with pytest.raises(ValueError):
        StrainTokenization(domain, token_size=200, print_output=False)


# ---------------------------------------------------------------------------
# MaskRandomTokens tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "setup",
    [
        "ufd_batch",
        "ufd_no_batch",
        "ufd_single_batch",
        "mfd_batch",
    ],
)
def test_MaskRandomTokens(request, setup):
    domain, num_tokens_per_block, num_blocks, sample = request.getfixturevalue(setup)
    # Drop random tokens
    # Initialize StrainTokenization transform
    token_transformation = StrainTokenization(
        domain,
        num_tokens_per_block=num_tokens_per_block,
    )
    mask_dict = {
        "p_mask": 0.2,
        "max_num_tokens": num_tokens_per_block,
    }
    mask_trafo = MaskRandomTokens(
        p_mask=mask_dict["p_mask"],
        max_num_tokens=mask_dict["max_num_tokens"],
    )

    # Evaluate transforms
    out = token_transformation(sample)
    out = mask_trafo(out)

    # Check that not more than max_num_tokens are masked per sample
    num_removed_tokens = np.sum(out["token_mask"], axis=-1)
    assert np.all(num_removed_tokens <= mask_dict["max_num_tokens"])

    # Only check probabilities if we can average over the batch dimension
    if len(out["position"].shape) > 2 and out["position"].shape[0] > 1:
        # Check that p_mask is correct
        prob_mask = np.mean(np.where(num_removed_tokens > 0.0, 1.0, 0.0))
        assert np.isclose(prob_mask, mask_dict["p_mask"], atol=0.1, rtol=0.1)

        # Check that dropped tokens are uniformly distributed
        hist_removed_tokens = np.sum(out["token_mask"], axis=0)
        mean_removed_tokens = np.mean(hist_removed_tokens)
        assert np.all(
            np.isclose(mean_removed_tokens, hist_removed_tokens, atol=5, rtol=5)
        )


def test_MaskRandomTokens_p_mask_zero():
    """With p_mask=0 no tokens should ever be masked."""
    domain = make_ufd()
    sample = make_sample(domain, batch_size=100)
    out = StrainTokenization(domain, num_tokens_per_block=40)(sample)
    out = MaskRandomTokens(p_mask=0.0, max_num_tokens=40)(out)
    assert not out["token_mask"].any()


def test_MaskRandomTokens_preserves_existing_mask():
    """Pre-existing True entries in token_mask must remain True after the transform."""
    domain = make_ufd()
    sample = make_sample(domain, batch_size=100)
    out = StrainTokenization(domain, num_tokens_per_block=40)(sample)
    out["token_mask"][:, 0] = True
    out = MaskRandomTokens(p_mask=0.0, max_num_tokens=40)(out)
    assert np.all(out["token_mask"][:, 0])


# ---------------------------------------------------------------------------
# MaskDetectors tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "setup",
    [
        "ufd_batch",
        "ufd_no_batch",
        "ufd_single_batch",
        "mfd_batch",
    ],
)
def test_MaskDetectors(request, setup):
    domain, num_tokens_per_block, num_blocks, sample = request.getfixturevalue(setup)

    # Initialize StrainTokenization transform
    token_transformation = StrainTokenization(
        domain,
        num_tokens_per_block=num_tokens_per_block,
    )
    # Always mask exactly H1, never L1
    mask_transformation = MaskDetectors(
        num_blocks=num_blocks,
        p_mask_012_detectors=[0.0, 1.0],
        p_mask_hlv={"H1": 1.0, "L1": 0.0},
    )

    out = token_transformation(sample)
    out = mask_transformation(out)

    # Check that mask has expected shape
    assert out["token_mask"].shape[-1] == num_tokens_per_block * num_blocks
    # Check that mask only contains True for tokens of exactly one detector
    assert np.all(np.sum(out["token_mask"], axis=-1) == num_tokens_per_block)

    trafo_dict = {
        "p_mask_012_detectors": [0.3, 0.7],
        "p_mask_hlv": {"H1": 0.4, "L1": 0.6},
    }

    mask_transformation = MaskDetectors(
        num_blocks=num_blocks,
        p_mask_012_detectors=trafo_dict["p_mask_012_detectors"],
        p_mask_hlv=trafo_dict["p_mask_hlv"],
    )
    out = token_transformation(sample)
    out = mask_transformation(out)

    # Masked count is always 0 or num_tokens_per_block (all-or-nothing per detector)
    count_masked_tokens = np.sum(out["token_mask"], axis=-1)
    assert np.all(np.isin(count_masked_tokens, [0, num_tokens_per_block]))

    # Only run probability checks if we can average over the batch dimension
    if len(out["position"].shape) > 2 and out["position"].shape[0] > 1:
        prob_mask_1_detector = np.mean(np.where(count_masked_tokens > 0, 1, 0))
        assert np.isclose(
            prob_mask_1_detector,
            trafo_dict["p_mask_012_detectors"][1],
            atol=0.1,
            rtol=0.1,
        )

        # Check per-detector masking probabilities
        detectors = [det for det in out["asds"].keys()]
        for b in range(num_blocks):
            b_min, b_max = b * num_tokens_per_block, (b + 1) * num_tokens_per_block
            vals = out["token_mask"][..., b_min:b_max]
            count_masked = np.sum(vals, axis=-1)
            assert np.all(np.isin(count_masked, [0, num_tokens_per_block]))
            prob_mask_detector = np.mean(np.where(count_masked > 0, 1, 0))
            prob_expected = (
                trafo_dict["p_mask_012_detectors"][1]
                * trafo_dict["p_mask_hlv"][detectors[b]]
            )
            assert np.isclose(prob_mask_detector, prob_expected, atol=0.1, rtol=0.1)


def test_MaskDetectors_never_masks():
    """With p_mask_012_detectors=[1.0, 0.0] nothing should ever be masked."""
    domain = make_ufd()
    sample = make_sample(domain, batch_size=100)
    out = StrainTokenization(domain, num_tokens_per_block=40)(sample)
    out = MaskDetectors(
        num_blocks=2,
        p_mask_012_detectors=[1.0, 0.0],
        p_mask_hlv={"H1": 0.5, "L1": 0.5},
    )(out)
    assert not out["token_mask"].any()


def test_MaskDetectors_preserves_existing_mask():
    """Pre-existing True entries in token_mask must remain True after the transform."""
    domain = make_ufd()
    sample = make_sample(domain, batch_size=100)
    out = StrainTokenization(domain, num_tokens_per_block=40)(sample)
    out["token_mask"][:, 0] = True
    out = MaskDetectors(
        num_blocks=2,
        p_mask_012_detectors=[1.0, 0.0],
        p_mask_hlv={"H1": 0.5, "L1": 0.5},
    )(out)
    assert np.all(out["token_mask"][:, 0])


def test_MaskDetectors_two_of_three():
    """With 3 detectors, masking 2 simultaneously exercises the multi-detector branch."""
    domain = make_ufd()
    num_tokens_per_block = 40
    # Build a 3-detector sample manually
    num_f = domain.frequency_mask_length
    batch_size = 100
    waveform = np.concatenate([np.zeros([batch_size, 1, 3, num_f])] * 3, axis=1)
    asds = {d: np.random.rand(batch_size, num_f) for d in ["H1", "L1", "V1"]}
    sample = {"waveform": waveform, "asds": asds}

    out = StrainTokenization(domain, num_tokens_per_block=num_tokens_per_block)(sample)
    # Always mask exactly 2 detectors, always H1 and L1
    out = MaskDetectors(
        num_blocks=3,
        p_mask_012_detectors=[0.0, 0.0, 1.0],
        p_mask_hlv={"H1": 0.5, "L1": 0.5, "V1": 0.0},
    )(out)
    # Exactly 2 * num_tokens_per_block tokens should be masked per sample
    assert np.all(np.sum(out["token_mask"], axis=-1) == 2 * num_tokens_per_block)


# ---------------------------------------------------------------------------
# MaskFrequencyEdges tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("setup", SETUPS)
def test_MaskFrequencyEdges(request, setup):
    domain, num_tokens_per_block, num_blocks, sample = request.getfixturevalue(setup)

    token_transformation = StrainTokenization(
        domain,
        num_tokens_per_block=num_tokens_per_block,
        print_output=False,
    )
    mask_dict = {
        "p_mask": 0.2,
        "f_max_lower": 100.0,
        "f_min_upper": 800.0,
        "p_lower_upper_both": [0.4, 0.4, 0.2],
        "p_same_all_detectors": 0.7,
    }
    mask_transformation = MaskFrequencyEdges(
        domain=domain,
        p_mask=mask_dict["p_mask"],
        f_max_lower=mask_dict["f_max_lower"],
        f_min_upper=mask_dict["f_min_upper"],
        p_lower_upper_both=mask_dict["p_lower_upper_both"],
        p_same_all_detectors=mask_dict["p_same_all_detectors"],
        print_output=False,
    )
    out = token_transformation(sample)
    out = mask_transformation(out)

    # Masked tokens must be at the lower end (f_min < f_max_lower) or
    # upper end (f_max > f_min_upper) of the frequency range.
    dropped_f_mins = out["position"][..., 0][out["token_mask"]]
    dropped_f_maxs = out["position"][..., 1][out["token_mask"]]
    assert np.all(
        np.logical_or(
            dropped_f_mins < mask_dict["f_max_lower"],
            dropped_f_maxs > mask_dict["f_min_upper"],
        )
    )

    # Only run probability checks if we can average over the batch dimension
    if len(out["position"].shape) > 2 and out["position"].shape[0] > 1:
        num_masked_tokens = np.sum(out["token_mask"], axis=-1)
        prob_mask = np.mean(np.where(num_masked_tokens > 0.0, 1.0, 0.0))
        assert np.isclose(prob_mask, mask_dict["p_mask"], atol=0.1, rtol=0.1)

        # Check p_lower_upper_both
        mask_lower = (
            out["position"][..., :num_tokens_per_block, 0] <= mask_dict["f_max_lower"]
        )
        mask_upper = (
            out["position"][..., :num_tokens_per_block, 1] >= mask_dict["f_min_upper"]
        )
        lower_blocks = []
        upper_blocks = []
        both_blocks = []
        for b in range(num_blocks):
            b_min, b_max = b * num_tokens_per_block, (b + 1) * num_tokens_per_block
            vals = out["token_mask"][:, b_min:b_max]
            num_masked_lower = np.sum(np.where(mask_lower, vals, False), axis=-1)
            num_masked_upper = np.sum(np.where(mask_upper, vals, False), axis=-1)
            lower = np.where(num_masked_lower > 0.0, 1.0, 0.0)
            upper = np.where(num_masked_upper > 0.0, 1.0, 0.0)
            both = np.logical_and(lower, upper)
            lower = np.where(np.logical_and(lower, ~both), 1.0, 0.0)
            upper = np.where(np.logical_and(upper, ~both), 1.0, 0.0)
            assert np.isclose(
                np.mean(lower),
                mask_dict["p_mask"] * mask_dict["p_lower_upper_both"][0],
                atol=0.1,
                rtol=0.1,
            )
            assert np.isclose(
                np.mean(upper),
                mask_dict["p_mask"] * mask_dict["p_lower_upper_both"][1],
                atol=0.1,
                rtol=0.1,
            )
            assert np.isclose(
                np.mean(both),
                mask_dict["p_mask"] * mask_dict["p_lower_upper_both"][2],
                atol=0.1,
                rtol=0.1,
            )
            lower_blocks.append(lower)
            upper_blocks.append(upper)
            both_blocks.append(both)

        # Check p_same_all_detectors
        all_lower = np.logical_and(*lower_blocks)
        all_upper = np.logical_and(*upper_blocks)
        all_both = np.logical_and(*both_blocks)
        p_mask_all_detectors = np.mean(
            np.logical_or.reduce((all_lower, all_upper, all_both))
        )
        assert np.isclose(
            p_mask_all_detectors,
            mask_dict["p_mask"] * mask_dict["p_same_all_detectors"],
            atol=0.1,
            rtol=0.1,
        )

        # Verify masked-token counts decrease toward higher frequencies at the lower
        # cut end, and increase toward higher frequencies at the upper cut end.
        edge_mask_lower = mask_lower[..., :-1] & ~mask_lower[..., 1:]
        mask_lower_strict = mask_lower.copy()
        mask_lower_strict[..., :-1][edge_mask_lower] = False
        edge_mask_upper = ~mask_upper[..., :-1] & mask_upper[..., 1:]
        mask_upper_strict = mask_upper.copy()
        mask_upper_strict[..., 1:][edge_mask_upper] = False
        masked_lower_blocks, masked_upper_blocks = [], []
        for b in range(num_blocks):
            b_min, b_max = b * num_tokens_per_block, (b + 1) * num_tokens_per_block
            vals = out["token_mask"][:, b_min:b_max]
            masked_lower_blocks.append(np.where(mask_lower_strict, vals, False))
            masked_upper_blocks.append(np.where(mask_upper_strict, vals, False))
        num_tokens_masked_lower = np.apply_over_axes(
            np.sum, np.array(masked_lower_blocks), [0, 1]
        ).squeeze()
        num_tokens_masked_upper = np.apply_over_axes(
            np.sum, np.array(masked_upper_blocks), [0, 1]
        ).squeeze()
        assert np.all(num_tokens_masked_lower[1:] <= num_tokens_masked_lower[:-1])
        assert np.all(num_tokens_masked_upper[1:] >= num_tokens_masked_upper[:-1])

    # Check that we never mask all tokens when f_max_lower > f_min_upper
    mask_dict_overlap = {
        "p_mask": 1.0,
        "f_max_lower": 900.0,
        "f_min_upper": 100.0,
        "p_lower_upper_both": [0.1, 0.1, 0.8],
        "p_same_all_detectors": 0.7,
    }
    mask_transformation_overlap = MaskFrequencyEdges(
        domain=domain,
        p_mask=mask_dict_overlap["p_mask"],
        f_max_lower=mask_dict_overlap["f_max_lower"],
        f_min_upper=mask_dict_overlap["f_min_upper"],
        p_lower_upper_both=mask_dict_overlap["p_lower_upper_both"],
        p_same_all_detectors=mask_dict_overlap["p_same_all_detectors"],
        print_output=False,
    )
    out = token_transformation(sample)
    out = mask_transformation_overlap(out)

    num_masked_tokens = np.sum(out["token_mask"], axis=-1)
    assert np.all(num_masked_tokens < num_tokens_per_block * num_blocks)
    dropped_f_mins = out["position"][..., 0][out["token_mask"]]
    dropped_f_maxs = out["position"][..., 1][out["token_mask"]]
    assert np.all(
        np.logical_or(
            dropped_f_mins < mask_dict_overlap["f_max_lower"],
            dropped_f_maxs > mask_dict_overlap["f_min_upper"],
        )
    )


def test_MaskFrequencyEdges_p_mask_zero():
    """With p_mask=0 no tokens should ever be masked."""
    domain = make_ufd()
    sample = make_sample(domain, batch_size=100)
    out = StrainTokenization(domain, num_tokens_per_block=40, print_output=False)(
        sample
    )
    out = MaskFrequencyEdges(
        domain=domain,
        p_mask=0.0,
        f_max_lower=100.0,
        f_min_upper=800.0,
        p_same_all_detectors=0.5,
        print_output=False,
    )(out)
    assert not out["token_mask"].any()


def test_MaskFrequencyEdges_preserves_existing_mask():
    """Pre-existing True entries in token_mask must remain True after the transform."""
    domain = make_ufd()
    sample = make_sample(domain, batch_size=100)
    out = StrainTokenization(domain, num_tokens_per_block=40, print_output=False)(
        sample
    )
    out["token_mask"][:, 0] = True
    out = MaskFrequencyEdges(
        domain=domain,
        p_mask=0.0,
        f_max_lower=100.0,
        f_min_upper=800.0,
        p_same_all_detectors=0.5,
        print_output=False,
    )(out)
    assert np.all(out["token_mask"][:, 0])


def test_MaskFrequencyEdges_p_lower_upper_both_not_summing_to_one():
    """p_lower_upper_both that does not sum to 1 raises ValueError."""
    domain = make_ufd()
    with pytest.raises(ValueError, match="does not sum to 1"):
        MaskFrequencyEdges(
            domain=domain,
            p_mask=0.2,
            f_max_lower=100.0,
            f_min_upper=800.0,
            p_same_all_detectors=0.5,
            p_lower_upper_both=[0.3, 0.3, 0.3],
            print_output=False,
        )
