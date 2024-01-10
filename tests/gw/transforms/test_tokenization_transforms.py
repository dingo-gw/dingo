import numpy as np

from dingo.gw.transforms import StrainTokenization


def test_StrainTokenization():
    num_tokens = 20
    f_min = 20.
    f_max = 1024.
    T = 8.
    num_f = int((f_max - f_min)*T)

    token_transformation = StrainTokenization(num_tokens, f_min, f_max, df=1/T)
    waveform = np.random.random_sample([2, 3, num_f])
    print(waveform.shape)
    sample = {"waveform": waveform}

    out = token_transformation(sample)
    tok_params = out["tokenization_parameters"]
    # Check that waveform has expected shape
    assert out["waveform"].shape[:-1] == (waveform.shape[0], waveform.shape[1], num_tokens)
    # Check that token parameters have expected shape
    assert tok_params["f_min_per_token"].shape == tok_params["f_max_per_token"].shape
    assert len(tok_params["f_min_per_token"]) == num_tokens
    assert len(tok_params["num_bins_per_token"]) == len(tok_params["f_max_per_token"]) - 1
    # Check that token parameters match with initial f_min & f_max
    assert tok_params["f_min_per_token"].min() == f_min
    assert tok_params["f_max_per_token"].max() >= f_max


if __name__ == "__main__":
    test_StrainTokenization()
