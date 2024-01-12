import numpy as np

from dingo.gw.transforms import StrainTokenization


def test_StrainTokenization():
    num_tokens = 20
    f_min = 20.0
    f_max = 1024.0
    T = 8.0
    num_f = int((f_max - f_min) * T) + 1

    token_transformation = StrainTokenization(num_tokens, f_min, f_max, df=1 / T)
    waveform = np.random.random_sample([2, 3, num_f])
    asds = {"H1": np.random.random_sample(num_f), "L1": np.random.random_sample(num_f)}
    sample = {"waveform": waveform, "asds": asds}

    out = token_transformation(sample)
    # Check that waveform has expected shape
    assert out["waveform"].shape[:-1] == (
        waveform.shape[0],
        waveform.shape[1],
        num_tokens,
    )
    # Check that token parameters have expected shape
    assert out["f_min_per_token"].shape == out["f_max_per_token"].shape
    assert len(out["f_min_per_token"]) == num_tokens
    # Check that token parameters match with initial f_min & f_max
    assert out["f_min_per_token"].min() == f_min
    assert out["f_max_per_token"].max() >= f_max
