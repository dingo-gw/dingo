import pytest
import numpy as np
from scipy.stats import binom

from dingo.gw.domains import FrequencyDomain
from dingo.gw.transforms import CropMaskStrainRandom


TOLERANCE = 1e-4  # probability with which we allow confidence tests to fail


@pytest.fixture
def cropping_setup():
    domain = FrequencyDomain(f_min=20, f_max=1024, delta_f=1 / 4)
    example_batch = dict(
        waveform=np.random.normal(size=(1000, 2, 3, len(domain) - domain.min_idx))
    )
    return domain, example_batch


def test_cropping_init_errors(cropping_setup):
    """Check that CropMaskStrainRandom raises errors for invalid initialization inputs."""
    domain, _ = cropping_setup
    with pytest.raises(ValueError):
        _ = CropMaskStrainRandom(domain, f_min_upper=domain.f_min - 2)
    with pytest.raises(ValueError):
        _ = CropMaskStrainRandom(domain, f_max_lower=domain.f_max + 2)
    with pytest.raises(ValueError):
        _ = CropMaskStrainRandom(domain, f_min_upper=32, f_max_lower=30)


def test_cropping_frequency_calibration(cropping_setup):
    """Test that the crop frequencies are calibrated w.r.t. the strain domain."""
    domain, example_batch = cropping_setup
    strain_in = example_batch["waveform"]

    # test cropping from below
    f_min = 32
    idx_f_min = np.where(domain()[domain.min_idx :] == f_min)[0][0]
    cropping_transform = CropMaskStrainRandom(
        domain, f_min_upper=f_min, deterministic=True
    )
    strain_out = cropping_transform(example_batch)["waveform"]
    # all values below 32Hz should be zero, all values above should be non-zero
    assert (strain_out[..., :idx_f_min] == 0).all()
    assert (strain_out[..., :idx_f_min] != strain_in[..., :idx_f_min]).all()
    assert (strain_out[..., idx_f_min:] != 0).all()
    assert (strain_out[..., idx_f_min:] == strain_in[..., idx_f_min:]).all()

    # test cropping from above
    f_max = 64
    idx_f_max = np.where(domain()[domain.min_idx :] == f_max)[0][0]
    cropping_transform = CropMaskStrainRandom(
        domain, f_max_lower=f_max, deterministic=True
    )
    strain_out = cropping_transform(example_batch)["waveform"]
    # all values above 64Hz should be zero, all values below should be non-zero
    assert (strain_out[..., idx_f_max + 1 :] == 0).all()
    assert (strain_out[..., idx_f_max + 1 :] != strain_in[..., idx_f_max + 1 :]).all()
    assert (strain_out[..., : idx_f_max + 1] != 0).all()
    assert (strain_out[..., : idx_f_max + 1] == strain_in[..., : idx_f_max + 1]).all()


def test_cropping_bounds(cropping_setup):
    domain, example_batch = cropping_setup
    frequencies = domain()[domain.min_idx :]
    n = 1000
    f_min_upper = domain.f_min + 2
    f_max_lower = domain.f_max - 4
    cropping_transform = CropMaskStrainRandom(
        domain,
        f_min_upper=f_min_upper,
        f_max_lower=f_max_lower,
        independent_detectors=True,
    )
    x = frequencies[None, None, :].repeat(n, 0).repeat(3, 1)
    out = cropping_transform(dict(waveform=x))["waveform"][:, 0]
    # check boundaries
    upper = np.max(out, axis=1)
    out[out == 0] = np.inf
    lower = np.min(out, axis=1)
    assert lower.min() == domain.f_min
    assert upper.max() == domain.f_max
    assert lower.max() == f_min_upper
    assert upper.min() == f_max_lower


def test_cropping_probability(cropping_setup):
    """Test that cropping probability is applied correctly."""
    domain, example_batch = cropping_setup
    frequencies = domain()[domain.min_idx :]
    strain_in = example_batch["waveform"]
    f_min = 24
    f_max = 512
    cropping_kwargs = dict(
        domain=domain,
        f_min_upper=f_min,
        f_max_lower=f_max,
        independent_detectors=True,
    )

    for p in [0.0, 0.2, 0.5, 0.8, 1.0]:
        for independent_lower_upper in [False, True]:
            cropping_transform = CropMaskStrainRandom(
                cropping_probability=p,
                independent_lower_upper=independent_lower_upper,
                **cropping_kwargs,
            )
            strain_out = cropping_transform(example_batch)["waveform"]
            is_cropped_lower = ((strain_out[..., 0]).all(axis=-1) == 0).flatten()
            is_cropped_upper = ((strain_out[..., -1]).all(axis=-1) == 0).flatten()
            n = len(is_cropped_lower)

            # probability that strain_out[..., 0] is 0 is given by p * q,
            # where q takes into account that even if the cropping transform is applied,
            # there is a chance that the cropping boundary falls on the first bin
            p_lower = p * (1 - 1 / (frequencies <= f_min).sum())
            p_upper = p * (1 - 1 / (frequencies >= f_max).sum())
            # check that correct fraction is cropped
            assert is_cropped_lower.sum() >= binom(n, p_lower).ppf(TOLERANCE)
            assert is_cropped_lower.sum() <= binom(n, p_lower).ppf(1 - TOLERANCE)
            assert is_cropped_upper.sum() >= binom(n, p_upper).ppf(TOLERANCE)
            assert is_cropped_upper.sum() <= binom(n, p_upper).ppf(1 - TOLERANCE)

            # check that cropping from below and from above is correlated iff
            # independent_detectors is True
            if 0 < p < 1:
                a = is_cropped_lower
                b = is_cropped_upper
                covariance = ((a - a.mean()) * (b - b.mean())).mean()
                correlation = covariance / (a.std() * b.std())
                # print(p, a.std(), b.std(), covariance, correlation, (a == b).mean())
                if independent_lower_upper:
                    assert np.abs(correlation) < 0.1
                elif 0.1 < p < 0.9:
                    assert np.abs(correlation) > 0.1


def test_cropping_bounds_independence(cropping_setup):
    """Check that bounds are independent along batch and optionally detector axes."""
    domain, example_batch = cropping_setup
    strain_in = example_batch["waveform"]
    example_batch_single = dict(waveform=strain_in[0])

    for independent_detectors in [False, True]:
        cropping_transform = CropMaskStrainRandom(
            domain,
            f_min_upper=32,
            f_max_lower=512,
            cropping_probability=0.8,
            independent_detectors=independent_detectors,
            independent_lower_upper=False,
        )

        # check batched version
        strain_out = cropping_transform(example_batch)["waveform"]
        mask = strain_in != strain_out
        # there should be no variation along channel axis
        assert mask.std(axis=-2).max() == 0
        # there should be variation along detector axis if independent_detectors is True
        assert (mask.std(axis=-3).max() > 0) == independent_detectors
        # there should be variation along batch
        assert mask.std(axis=-4).max() > 0

        # check non-batched version
        strain_out = cropping_transform(example_batch_single)["waveform"]
        mask = strain_in[0] != strain_out
        # there should be no variation along channel axis
        assert mask.std(axis=-2).max() == 0
        # there should be variation along detector axis if independent_detectors is True
        assert (mask.std(axis=-3).max() > 0) == independent_detectors


# TODO once multibanded frequency domain is merged into main
# def test_cropping_multibanded():
#     pass
