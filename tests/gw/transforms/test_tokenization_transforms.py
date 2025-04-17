import numpy as np
import pytest

from dingo.gw.domains import UniformFrequencyDomain, MultibandedFrequencyDomain
from dingo.gw.transforms import (
    StrainTokenization,
    DropDetectors,
    DropFrequenciesToUpdateRange,
    DropFrequencyInterval,
    DropRandomTokens,
)
from gw.transforms import NormalizePosition


@pytest.fixture
def strain_tokenization_setup():
    num_tokens_per_block = 40  # needs to be larger than for MFD
    f_min = 20.0
    f_max = 1024.0
    T = 8.0
    domain = UniformFrequencyDomain(f_min, f_max, delta_f=1 / T)
    num_f = domain.frequency_mask_length

    batch_size = 100
    waveform_h1 = np.zeros([batch_size, 1, 3, num_f])
    waveform_l1 = np.ones([batch_size, 1, 3, num_f])
    # Set real part of second detector to linearly increasing values
    waveform_l1[0, 0, 0, :] *= np.arange(1, num_f + 1)
    waveform = np.concatenate([waveform_h1, waveform_l1], axis=-3)
    num_blocks = waveform.shape[-3]
    asds = {
        "H1": np.random.random([batch_size, num_f]),
        "L1": np.random.random([batch_size, num_f]),
    }

    sample = {"waveform": waveform, "asds": asds}

    return domain, num_tokens_per_block, num_blocks, sample


@pytest.fixture
def strain_tokenization_setup_no_batch():
    num_tokens_per_block = 40  # needs to be larger than for MFD
    f_min = 20.0
    f_max = 1024.0
    T = 8.0
    domain = UniformFrequencyDomain(f_min, f_max, delta_f=1 / T)
    num_f = domain.frequency_mask_length

    waveform_h1 = np.zeros([1, 3, num_f])
    waveform_l1 = np.ones([1, 3, num_f])
    # Set real part of second detector to linearly increasing values
    waveform_l1[0, 0, :] *= np.arange(1, num_f + 1)
    waveform = np.concatenate([waveform_h1, waveform_l1], axis=-3)
    num_blocks = waveform.shape[-3]
    asds = {"H1": np.random.random([num_f]), "L1": np.random.random([num_f])}

    sample = {"waveform": waveform, "asds": asds}

    return domain, num_tokens_per_block, num_blocks, sample


@pytest.fixture
def strain_tokenization_setup_single_batch():
    num_tokens_per_block = 40  # needs to be larger than for MFD
    f_min = 20.0
    f_max = 1024.0
    T = 8.0
    domain = UniformFrequencyDomain(f_min, f_max, delta_f=1 / T)
    num_f = domain.frequency_mask_length

    waveform_h1 = np.zeros([1, 3, num_f])
    waveform_l1 = np.ones([1, 3, num_f])
    # Set real part of second detector to linearly increasing values
    waveform_l1[0, 0, :] *= np.arange(1, num_f + 1)
    waveform = np.concatenate([waveform_h1, waveform_l1], axis=-3)
    num_blocks = waveform.shape[-3]
    asds = {"H1": np.random.random([1, num_f]), "L1": np.random.random([1, num_f])}

    sample = {"waveform": np.expand_dims(waveform, axis=0), "asds": asds}

    return domain, num_tokens_per_block, num_blocks, sample


@pytest.fixture
def strain_tokenization_setup_mfd():
    num_tokens_per_block = 20
    nodes = [20.0, 34.0, 46.0, 62.0, 78.0, 990.0]
    f_min = 20.0
    f_max = 990.0
    T = 8.0
    base_domain = UniformFrequencyDomain(f_min=f_min, f_max=f_max, delta_f=1 / T)
    domain = MultibandedFrequencyDomain(
        nodes=nodes, delta_f_initial=1 / T, base_domain=base_domain
    )
    num_f = domain.frequency_mask_length

    batch_size = 100
    waveform_h1 = np.zeros([batch_size, 1, 3, num_f])
    waveform_l1 = np.ones([batch_size, 1, 3, num_f])
    # Set real part of second detector to linearly increasing values
    waveform_l1[0, 0, 0, :] *= np.arange(1, num_f + 1)
    waveform = np.concatenate([waveform_h1, waveform_l1], axis=-3)
    num_blocks = waveform.shape[-3]
    asds = {
        "H1": np.random.random([batch_size, num_f]),
        "L1": np.random.random([batch_size, num_f]),
    }

    sample = {"waveform": waveform, "asds": asds}

    return domain, num_tokens_per_block, num_blocks, sample


@pytest.mark.parametrize(
    "setup",
    [
        "strain_tokenization_setup",
        "strain_tokenization_setup_no_batch",
        "strain_tokenization_setup_single_batch",
        "strain_tokenization_setup_mfd",
    ],
)
def test_StrainTokenization(request, setup):
    domain, num_tokens_per_block, num_blocks, sample = request.getfixturevalue(setup)

    # Initialize StrainTokenization transform
    token_transformation = StrainTokenization(
        domain,
        num_tokens_per_block=num_tokens_per_block,
    )

    # Evaluate StrainTokenization transform
    out = token_transformation(sample)

    # -------------- Checks regarding the waveform --------------
    # Check that waveform has expected num_tokens
    assert out["waveform"].shape[-2] == num_tokens_per_block * num_blocks
    # Check that linearly increasing values got mapped to expected position:
    num_features = out["waveform"].shape[-1]
    # We expect L1 in second half of num_tokens and real in first third of num_features
    if len(out["waveform"].shape) == 2:
        vals = out["waveform"][num_tokens_per_block:, : int(num_features / 3)]
    else:
        vals = out["waveform"][0, num_tokens_per_block:, : int(num_features / 3)]
    assert np.all(
        vals.flatten()[: domain.frequency_mask_length]
        == np.arange(1, domain.frequency_mask_length + 1)
    )

    # -------------- Checks regarding the position --------------
    # Check that position has expected shape
    assert out["position"].shape[-2:] == (num_tokens_per_block * num_blocks, 3)
    # Check that f_min of all detectors are num_tokens_per_block apart
    assert all(
        [
            np.all(out["position"][..., n, 0] == domain.f_min)
            for n in range(0, num_blocks * num_tokens_per_block, num_tokens_per_block)
        ]
    )
    # Check that f_max of all detectors are num_tokens_per_block apart
    inds_f_max = [
        out["position"][
            ..., n * num_tokens_per_block : num_tokens_per_block * (n + 1), 1
        ].argmax()
        + n * num_tokens_per_block
        for n in range(0, num_blocks)
    ]
    assert inds_f_max == [
        i
        for i in range(
            num_tokens_per_block - 1,
            num_blocks * num_tokens_per_block,
            num_tokens_per_block,
        )
    ]
    # Check that f_max of each detector is larger equal domain.f_max
    assert all([np.all(out["position"][..., i, 1] >= domain.f_max) for i in inds_f_max])
    # Check that block information contains correct number of blocks
    assert len(np.unique(out["position"][..., 2])) == num_blocks
    # Check that block encodings are stacked as one block after the other
    assert all(
        [
            len(
                np.unique(
                    out["position"][
                        ...,
                        n * num_tokens_per_block : num_tokens_per_block * (n + 1),
                        2,
                    ]
                )
            )
            == 1
            for n in range(0, num_blocks)
        ]
    )
    # Check that mask has expected shape
    assert out["drop_token_mask"].shape[-1] == num_tokens_per_block * num_blocks
    # Check that mask default value is False for all tokens (i.e., drop no tokens)
    assert out["drop_token_mask"].all() == False


@pytest.mark.parametrize(
    "setup",
    [
        "strain_tokenization_setup",
        "strain_tokenization_setup_no_batch",
        "strain_tokenization_setup_single_batch",
        "strain_tokenization_setup_mfd",
    ],
)
def test_StrainTokenization_token_size(request, setup):
    # transform should also work when specifying a token_size (instead of num_tokens_per_block)
    domain, num_tokens_per_block, num_blocks, sample = request.getfixturevalue(setup)
    token_size = int(np.ceil(domain.frequency_mask_length / num_tokens_per_block))

    # Initialize StrainTokenization transform
    token_transformation = StrainTokenization(
        domain,
        token_size=token_size,
    )

    # Evaluate StrainTokenization transform
    out = token_transformation(sample)

    # -------------- Checks regarding the waveform --------------
    # Check that waveform has expected num_tokens
    assert out["waveform"].shape[-2] == num_tokens_per_block * num_blocks
    # Check that linearly increasing values got mapped to expected position:
    num_features = out["waveform"].shape[-1]
    # We expect L1 in second half of num_tokens and real in first third of num_features
    if len(out["waveform"].shape) == 2:
        vals = out["waveform"][num_tokens_per_block:, : int(num_features / 3)]
    else:
        vals = out["waveform"][0, num_tokens_per_block:, : int(num_features / 3)]
    assert np.all(
        vals.flatten()[: domain.frequency_mask_length]
        == np.arange(1, domain.frequency_mask_length + 1)
    )

    # -------------- Checks regarding the position --------------
    # Check that position has expected shape
    assert out["position"].shape[-2:] == (num_tokens_per_block * num_blocks, 3)
    # Check that f_min of all detectors are num_tokens_per_block apart
    assert all(
        [
            np.all(out["position"][..., n, 0] == domain.f_min)
            for n in range(0, num_blocks * num_tokens_per_block, num_tokens_per_block)
        ]
    )
    # Check that f_max of all detectors are num_tokens_per_block apart
    inds_f_max = [
        out["position"][
            ..., n * num_tokens_per_block : num_tokens_per_block * (n + 1), 1
        ].argmax()
        + n * num_tokens_per_block
        for n in range(0, num_blocks)
    ]
    assert inds_f_max == [
        i
        for i in range(
            num_tokens_per_block - 1,
            num_blocks * num_tokens_per_block,
            num_tokens_per_block,
        )
    ]
    # Check that f_max of each detector is larger equal domain.f_max
    assert all([np.all(out["position"][..., i, 1] >= domain.f_max) for i in inds_f_max])
    # Check that block information contains correct number of blocks
    assert len(np.unique(out["position"][..., 2])) == num_blocks
    # Check that block encodings are stacked as one block after the other
    assert all(
        [
            len(
                np.unique(
                    out["position"][
                        ...,
                        n * num_tokens_per_block : num_tokens_per_block * (n + 1),
                        2,
                    ]
                )
            )
            == 1
            for n in range(0, num_blocks)
        ]
    )
    # Check that mask has expected shape
    assert out["drop_token_mask"].shape[-1] == num_tokens_per_block * num_blocks
    # Check that mask default value is False for all tokens (i.e., drop no tokens)
    assert out["drop_token_mask"].all() == False


@pytest.mark.parametrize(
    "setup",
    [
        "strain_tokenization_setup",
        "strain_tokenization_setup_no_batch",
        "strain_tokenization_setup_single_batch",
        "strain_tokenization_setup_mfd",
    ],
)
def test_DropDetectors(request, setup):
    domain, num_tokens_per_block, num_blocks, sample = request.getfixturevalue(setup)

    # Initialize StrainTokenization transform
    token_transformation = StrainTokenization(
        domain,
        num_tokens_per_block=num_tokens_per_block,
    )
    # Initialize DropDetectors transform
    drop_transformation = DropDetectors(
        num_blocks=num_blocks,
        p_drop_012_detectors=[0.0, 1.0],
        p_drop_hlv={"H1": 1.0, "L1": 0.0},
    )

    # Evaluate StrainTokenization transform
    out = token_transformation(sample)
    # Evaluate DropDetectors transform
    out = drop_transformation(out)

    # Check that mask has expected shape
    assert out["drop_token_mask"].shape[-1] == num_tokens_per_block * num_blocks
    # Check that mask only contains True for tokens of one detector
    assert np.all(np.sum(out["drop_token_mask"], axis=-1) == num_tokens_per_block)

    trafo_dict = {
        "p_drop_012_detectors": [0.3, 0.7],
        "p_drop_hlv": {"H1": 0.4, "L1": 0.6},
    }

    # Initialize DropDetectors transform
    drop_transformation = DropDetectors(
        num_blocks=num_blocks,
        p_drop_012_detectors=trafo_dict["p_drop_012_detectors"],
        p_drop_hlv=trafo_dict["p_drop_hlv"],
    )
    out = token_transformation(sample)
    out = drop_transformation(out)

    # Check whether probability for dropping one detector aligns with p_drop_012_detectors[1]
    count_dropped_tokens = np.sum(out["drop_token_mask"], axis=-1)
    assert np.all(np.isin(count_dropped_tokens, [0, num_tokens_per_block]))

    # Only run tests involving probabilities if we can average over the batch dimension
    if len(out["position"].shape) > 2 and out["position"].shape[0] > 1:
        prob_drop_1_detector = np.mean(np.where(count_dropped_tokens > 0, 1, 0))
        assert np.isclose(
            prob_drop_1_detector,
            trafo_dict["p_drop_012_detectors"][1],
            atol=0.1,
            rtol=0.1,
        )

        # Check whether probabilities for individual detectors are consistent with p_drop_hlv
        detectors = [det for det in out["asds"].keys()]
        for b in range(num_blocks):
            b_min, b_max = b * num_tokens_per_block, (b + 1) * num_tokens_per_block
            vals = out["drop_token_mask"][..., b_min:b_max]
            # Check that either 0 or num_tokens_per_block values are dropped
            count_dropped_tokens = np.sum(vals, axis=-1)
            assert np.all(np.isin(count_dropped_tokens, [0, num_tokens_per_block]))
            prob_drop_detector = np.mean(np.where(count_dropped_tokens > 0, 1, 0))
            prob_expected = (
                trafo_dict["p_drop_012_detectors"][1]
                * trafo_dict["p_drop_hlv"][detectors[b]]
            )
            assert np.isclose(prob_drop_detector, prob_expected, atol=0.1, rtol=0.1)


@pytest.mark.parametrize(
    "setup",
    [
        "strain_tokenization_setup",
        "strain_tokenization_setup_no_batch",
        "strain_tokenization_setup_single_batch",
        "strain_tokenization_setup_mfd",
    ],
)
def test_DropFrequenciesToUpdateRange(request, setup):
    domain, num_tokens_per_block, num_blocks, sample = request.getfixturevalue(setup)
    # (1) Test cuts in frequency domain

    # Initialize StrainTokenization transform
    token_transformation = StrainTokenization(
        domain,
        num_tokens_per_block=num_tokens_per_block,
    )
    drop_dict = {
        "p_cut": 0.2,
        "f_max_lower_cut": 100.0,
        "f_min_upper_cut": 800.0,
        "p_lower_upper_both": [0.4, 0.4, 0.2],
        "p_same_cut_all_detectors": 0.7,
    }
    drop_transformation = DropFrequenciesToUpdateRange(
        domain=domain,
        p_cut=drop_dict["p_cut"],
        f_max_lower_cut=drop_dict["f_max_lower_cut"],
        f_min_upper_cut=drop_dict["f_min_upper_cut"],
        p_lower_upper_both=drop_dict["p_lower_upper_both"],
        p_same_cut_all_detectors=drop_dict["p_same_cut_all_detectors"],
    )
    # Evaluate transforms
    out = token_transformation(sample)
    out = drop_transformation(out)

    # Check that dropped tokens are either at frequencies lower than f_max_lower_cut and larger than f_min_upper_cut
    dropped_f_mins = out["position"][..., 0][out["drop_token_mask"]]
    dropped_f_maxs = out["position"][..., 1][out["drop_token_mask"]]
    assert np.all(
        np.logical_or(
            dropped_f_mins < drop_dict["f_max_lower_cut"],
            dropped_f_maxs > drop_dict["f_min_upper_cut"],
        )
    )

    # Only run tests involving probabilities if we can average over the batch dimension
    if len(out["position"].shape) > 2 and out["position"].shape[0] > 1:
        # Check that p_cut is correct
        num_removed_tokens = np.sum(out["drop_token_mask"], axis=-1)
        prob_cut = np.mean(np.where(num_removed_tokens > 0.0, 1.0, 0.0))
        # Very noisy because we are just averaging over 100 samples
        assert np.isclose(prob_cut, drop_dict["p_cut"], atol=0.1, rtol=0.1)

        # Check that p_upper_lower_both is correct
        # Find token indices which correspond to f_max_lower_cut and f_min_upper_cut
        mask_lower = (
            out["position"][..., :num_tokens_per_block, 0]
            <= drop_dict["f_max_lower_cut"]
        )
        mask_upper = (
            out["position"][..., :num_tokens_per_block, 1]
            >= drop_dict["f_min_upper_cut"]
        )
        lower_blocks = []
        upper_blocks = []
        both_blocks = []
        for b in range(num_blocks):
            b_min, b_max = b * num_tokens_per_block, (b + 1) * num_tokens_per_block
            vals = out["drop_token_mask"][:, b_min:b_max]
            num_masked_lower = np.sum(np.where(mask_lower, vals, False), axis=-1)
            num_masked_upper = np.sum(np.where(mask_upper, vals, False), axis=-1)
            lower = np.where(num_masked_lower > 0.0, 1.0, 0.0)
            upper = np.where(num_masked_upper > 0.0, 1.0, 0.0)
            both = np.logical_and(lower, upper)
            lower = np.where(np.logical_and(lower, ~both), 1.0, 0.0)
            upper = np.where(np.logical_and(upper, ~both), 1.0, 0.0)
            assert np.isclose(
                np.mean(lower),
                drop_dict["p_cut"] * drop_dict["p_lower_upper_both"][0],
                atol=0.1,
                rtol=0.1,
            )
            assert np.isclose(
                np.mean(upper),
                drop_dict["p_cut"] * drop_dict["p_lower_upper_both"][1],
                atol=0.1,
                rtol=0.1,
            )
            assert np.isclose(
                np.mean(both),
                drop_dict["p_cut"] * drop_dict["p_lower_upper_both"][2],
                atol=0.1,
                rtol=0.1,
            )

            lower_blocks.append(lower)
            upper_blocks.append(upper)
            both_blocks.append(both)

        # Check that p_cut_all_detectors is correct
        all_lower = np.logical_and(*lower_blocks)
        all_upper = np.logical_and(*upper_blocks)
        all_both = np.logical_and(*both_blocks)
        p_cut_all_detectors = np.mean(
            np.logical_or.reduce((all_lower, all_upper, all_both))
        )
        assert np.isclose(
            p_cut_all_detectors,
            drop_dict["p_cut"] * drop_dict["p_same_cut_all_detectors"],
            atol=0.1,
            rtol=0.1,
        )

        # Check that we sample the cut frequencies uniformly in UFD
        # Make sure to only consider bins that are completely in [f_min, f_max_lower] and [f_min_upper, f_max]
        # => remove tokens at border
        edge_mask_lower = mask_lower[..., :-1] & ~mask_lower[..., 1:]
        mask_lower_strict = mask_lower.copy()
        mask_lower_strict[..., :-1][edge_mask_lower] = False
        edge_mask_upper = ~mask_upper[..., :-1] & mask_upper[..., 1:]
        mask_upper_strict = mask_upper.copy()
        mask_upper_strict[..., 1:][edge_mask_upper] = False
        # Combine detectors as well as lower & both and upper & both to get better stats
        masked_lower_blocks, masked_upper_blocks = [], []
        edge_mask_lower_blocks, edge_mask_upper_blocks = [], []
        for b in range(num_blocks):
            b_min, b_max = b * num_tokens_per_block, (b + 1) * num_tokens_per_block
            vals = out["drop_token_mask"][:, b_min:b_max]
            masked_lower_blocks.append(np.where(mask_lower_strict, vals, False))
            masked_upper_blocks.append(np.where(mask_upper_strict, vals, False))
            edge_mask_lower_blocks.append(
                masked_lower_blocks[-1][..., :-1] & ~masked_lower_blocks[-1][..., 1:]
            )
            edge_mask_upper_blocks.append(
                ~masked_upper_blocks[-1][..., :-1] & masked_upper_blocks[-1][..., 1:]
            )
        num_tokens_masked_lower = np.apply_over_axes(
            np.sum, np.array(masked_lower_blocks), [0, 1]
        ).squeeze()  # (num_tokens)
        num_tokens_masked_upper = np.apply_over_axes(
            np.sum, np.array(masked_upper_blocks), [0, 1]
        ).squeeze()  # (num_tokens)

        # Since we mask from f_min to a random f_lower and from a random f_upper to f_max, we expect the count of masked
        # tokens to decrease at the lower end and increase at the upper end.
        assert np.all(num_tokens_masked_lower[1:] <= num_tokens_masked_lower[:-1])
        assert np.all(num_tokens_masked_upper[1:] >= num_tokens_masked_upper[:-1])

        if isinstance(domain, UniformFrequencyDomain):
            # We sample f_max_lower and f_min_upper in UFD, so we expect the masked edge tokens to be uniformly
            # distributed.
            num_cuts_lower = np.apply_over_axes(
                np.sum, np.array(edge_mask_lower_blocks), [0, 1]
            ).squeeze()  # (num_tokens)
            num_cuts_upper = np.apply_over_axes(
                np.sum, np.array(edge_mask_upper_blocks), [0, 1]
            ).squeeze()  # (num_tokens)
            non_zero_lower = num_cuts_lower[num_cuts_lower > 0.0]
            non_zero_upper = num_cuts_upper[num_cuts_upper > 0.0]
            if not non_zero_lower.size == 0:
                assert np.isclose(
                    np.mean(non_zero_lower), non_zero_lower, atol=5, rtol=5
                ).all()
            if not non_zero_upper.size == 0:
                assert np.isclose(
                    np.mean(non_zero_upper), non_zero_upper, atol=5, rtol=5
                ).all()

        elif isinstance(domain, MultibandedFrequencyDomain):
            # We expect tokens completely within [f_min, f_max_lower] and [f_min_upper, f_max]
            # AND with the same compression factor (i.e., tokens between the same nodes) to be masked with equal
            # probability
            tokens_in_first_band = np.where(
                out["position"][0, :num_tokens_per_block, 1] < domain.nodes[1],
                num_tokens_masked_lower,
                0.0,
            )
            # FIX ME: I cannot test this at the moment because the nodes of the MFD domain do not coincide with the
            # beginning of a token
            # This means that some tokens contain strain values separated by different delta_f!
            # tokens_in_second_band =
            print(
                "TODO: Implement test for distribution of frequency cut values in MFD"
            )


@pytest.mark.parametrize(
    "setup",
    [
        "strain_tokenization_setup",
        "strain_tokenization_setup_no_batch",
        "strain_tokenization_setup_single_batch",
        "strain_tokenization_setup_mfd",
    ],
)
def test_DropFrequencyInterval(request, setup):
    domain, num_tokens_per_block, num_blocks, sample = request.getfixturevalue(setup)
    # (2) Test masking frequency interval

    # Initialize StrainTokenization transform
    token_transformation = StrainTokenization(
        domain,
        num_tokens_per_block=num_tokens_per_block,
    )
    # Test mask_glitch
    drop_dict = {
        "p_glitch_per_detector": 0.4,
        "f_min": 100.0,
        "f_max": 500.0,
        "max_width": 100.0,
    }
    drop_transformation = DropFrequencyInterval(
        domain=domain,
        p_per_detector=drop_dict["p_glitch_per_detector"],
        f_min=drop_dict["f_min"],
        f_max=drop_dict["f_max"],
        max_width=drop_dict["max_width"],
    )
    # Evaluate transforms
    out = token_transformation(sample)
    out = drop_transformation(out)

    # Check that no tokens are masked outside [f_min, f_max]
    # It can happen that drop_dict['f_min'] falls in the middle of a token. Such a token might be dropped, resulting in
    # the f_min of the token being lower than drop_dict['f_min']. The same can happen for drop_dict['f_max'].
    # To exclude this case, we select f_max (f_min) of the dropped tokens when comparing to drop_dict['f_min']
    # (drop_dict['f_max'])
    dropped_f_mins = out["position"][..., 1][out["drop_token_mask"]]
    dropped_f_maxs = out["position"][..., 0][out["drop_token_mask"]]
    assert np.all(
        np.logical_and(
            dropped_f_mins >= drop_dict["f_min"], dropped_f_maxs <= drop_dict["f_max"]
        )
    )

    # Check that masked ranges are not wider than max_width
    mask_has_true = []
    for b in range(num_blocks):
        b_min, b_max = b * num_tokens_per_block, (b + 1) * num_tokens_per_block
        vals = out["drop_token_mask"][..., b_min:b_max]
        first_true_idx = np.argmax(vals, axis=-1)
        last_true_idx = vals.shape[-1] - 1 - np.argmax(vals[..., ::-1], axis=-1)

        # Initialize edge mask
        edge_mask_lower = np.zeros_like(vals, dtype=bool)
        edge_mask_upper = np.zeros_like(vals, dtype=bool)
        if len(out["position"].shape) > 2:
            batch_indices = np.arange(vals.shape[0])
            edge_mask_lower[batch_indices, first_true_idx] = True
            edge_mask_upper[batch_indices, last_true_idx] = True
        else:
            edge_mask_lower[first_true_idx] = True
            edge_mask_upper[last_true_idx] = True

        # If a row has no True values, clear accidentally masked values at the beginning or end
        has_true = np.any(vals, axis=-1)
        edge_mask_lower[~has_true] = False
        edge_mask_upper[~has_true] = False

        # Depending on the position of the tokens and the values of drop_dict['f_min'] and drop_dict['f_max'], it can
        # happen that we mask a larger frequency range than drop_dict['max_width']. This is the case when
        # drop_dict['f_min'] is located in the middle/at the upper end of a token and drop_dict['f_max'] is located in
        # the middle/at the lower end of a token.
        # Similar to the case of testing that tokens fall in the correct range [drop_dict['f_min'], drop_dict['f_max']],
        # we select f_max (f_min) of the dropped tokens for the lower edge (upper edge)
        dropped_f_mins_lower = out["position"][..., b_min:b_max, 1][edge_mask_lower]
        dropped_f_maxs_upper = out["position"][..., b_min:b_max, 0][edge_mask_upper]
        # If we only dropped one token, dropped_f_mins_lower (based on f_max) is larger than dropped_f_maxs_upper
        # (based on f_min). We mask these tokens during the check
        diff_f = dropped_f_maxs_upper - dropped_f_mins_lower
        assert np.all(np.where(diff_f > 0.0, diff_f <= drop_dict["max_width"], True))

        # Save has_true for next check
        mask_has_true.append(has_true)

    # Only run tests involving probabilities if we can average over the batch dimension
    if len(out["position"].shape) > 2 and out["position"].shape[0] > 1:
        # Check that p_glitch_per_detector is correct
        masked_blocks = np.concatenate(mask_has_true)
        prob_glitch = np.mean(masked_blocks)
        assert np.isclose(
            prob_glitch, drop_dict["p_glitch_per_detector"], atol=0.1, rtol=0.1
        )

        # Check that f_lower and f_upper are sampled uniformly in UFD between f_min and f_max
        # Similarly complicated as in test above, TODO


@pytest.mark.parametrize(
    "setup",
    [
        "strain_tokenization_setup",
        "strain_tokenization_setup_no_batch",
        "strain_tokenization_setup_single_batch",
        "strain_tokenization_setup_mfd",
    ],
)
def test_DropRandomTokens(request, setup):
    domain, num_tokens_per_block, num_blocks, sample = request.getfixturevalue(setup)
    # Drop random tokens
    # Initialize StrainTokenization transform
    token_transformation = StrainTokenization(
        domain,
        num_tokens_per_block=num_tokens_per_block,
    )
    drop_dict = {
        "p_drop": 0.2,
        "max_num_tokens": num_tokens_per_block,
    }
    drop_trafo = DropRandomTokens(
        p_drop=drop_dict["p_drop"],
        max_num_tokens=drop_dict["max_num_tokens"],
    )

    # Evaluate transforms
    out = token_transformation(sample)
    out = drop_trafo(out)

    # Check that not more than max_num_tokens are masked per sample
    num_removed_tokens = np.sum(out["drop_token_mask"], axis=-1)
    assert np.all(num_removed_tokens <= drop_dict["max_num_tokens"])

    # Only check probabilities if we can average over the batch dimension
    if len(out["position"].shape) > 2 and out["position"].shape[0] > 1:
        # Check that p_drop is correct
        prob_drop = np.mean(np.where(num_removed_tokens > 0.0, 1.0, 0.0))
        assert np.isclose(prob_drop, drop_dict["p_drop"], atol=0.1, rtol=0.1)

        # Check that dropped tokens are uniformly distributed
        hist_removed_tokens = np.sum(out["drop_token_mask"], axis=0)
        mean_removed_tokens = np.mean(hist_removed_tokens)
        assert np.all(
            np.isclose(mean_removed_tokens, hist_removed_tokens, atol=5, rtol=5)
        )


@pytest.mark.parametrize(
    "setup",
    [
        "strain_tokenization_setup",
        "strain_tokenization_setup_no_batch",
        "strain_tokenization_setup_single_batch",
        "strain_tokenization_setup_mfd",
    ],
)
def test_NormalizePosition(request, setup):
    domain, num_tokens_per_block, num_blocks, sample = request.getfixturevalue(setup)

    # Initialize StrainTokenization transform
    token_transformation = StrainTokenization(
        domain,
        num_tokens_per_block=num_tokens_per_block,
    )
    trafo = NormalizePosition()
    # Evaluate transforms
    out = token_transformation(sample)
    original_positions = out["position"].copy()
    out = trafo(out)

    # Check that blocks remain the same
    assert np.all(out["position"][..., 2] == original_positions[..., 2])

    # Check that f_min and f_max are correctly normalized
    f_min = np.min(original_positions[..., 0])
    f_max = np.max(original_positions[..., 1])
    f_min_norm = (original_positions[..., 0] - f_min) / (f_max - f_min)
    f_max_norm = (original_positions[..., 1] - f_min) / (f_max - f_min)
    assert np.all(f_min_norm == out["position"][..., 0])
    assert np.all(f_max_norm == out["position"][..., 1])
