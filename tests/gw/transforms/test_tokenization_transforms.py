import numpy as np
import pytest

from dingo.gw.domains import FrequencyDomain, MultibandedFrequencyDomain
from dingo.gw.transforms import StrainTokenization, DropDetectors


@pytest.fixture
def strain_tokenization_setup():
    num_tokens_per_block = 20
    f_min = 20.0
    f_max = 1024.0
    T = 8.0
    domain = FrequencyDomain(f_min, f_max, delta_f=1 / T)
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
    num_tokens_per_block = 20
    f_min = 20.0
    f_max = 1024.0
    T = 8.0
    domain = FrequencyDomain(f_min, f_max, delta_f=1 / T)
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
def strain_tokenization_setup_mfd():
    num_tokens_per_block = 20
    nodes = [20.0, 34.0, 46.0, 62.0, 78.0, 990.0]
    f_min = 20.0
    f_max = 990.0
    T = 8.0
    base_domain = FrequencyDomain(f_min=f_min, f_max=f_max, delta_f=1 / T)
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
        "strain_tokenization_setup_mfd",
    ],
)
def test_StrainTokenization(request, setup):
    domain, num_tokens_per_block, num_blocks, sample = request.getfixturevalue(setup)

    # Initialize StrainTokenization transform
    token_transformation = StrainTokenization(
        domain,
        num_tokens_per_block=num_tokens_per_block,
        normalize_frequency=False,
        single_tokenizer=True,
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
        "strain_tokenization_setup_mfd",
    ],
)
def test_StrainTokenization_token_size(request, setup):
    # transform should also work when specifying a token_size (instead of num_tokens_per_block)
    domain, num_tokens_per_block, num_blocks, sample = request.getfixturevalue(setup)
    token_size = int(np.ceil(domain.frequency_mask_length / num_tokens_per_block))

    # Initialize StrainTokenization transform
    token_transformation = StrainTokenization(
        domain, token_size=token_size, normalize_frequency=False, single_tokenizer=True
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
        "strain_tokenization_setup_mfd",
    ],
)
def test_DroDetectors(request, setup):
    domain, num_tokens_per_block, num_blocks, sample = request.getfixturevalue(setup)

    # Initialize StrainTokenization transform
    token_transformation = StrainTokenization(
        domain,
        num_tokens_per_block=num_tokens_per_block,
        normalize_frequency=False,
        single_tokenizer=True,
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
    assert np.all(np.sum(out["drop_token_mask"], axis=1) == num_tokens_per_block)


# TODO: write tests for DropFrequencyRange
