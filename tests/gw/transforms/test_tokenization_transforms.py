import numpy as np

from dingo.gw.domains import FrequencyDomain
from dingo.gw.transforms import StrainTokenization


def test_StrainTokenization():
    num_tokens = 20
    f_min = 20.0
    f_max = 1024.0
    T = 8.0
    num_f = int((f_max - f_min) * T) + 1
    domain = FrequencyDomain(f_min, f_max, delta_f=1/T)
    single_tokenizer = False
    norm_freq = False
    token_transformation = StrainTokenization(
        domain, num_tokens, normalize_frequency=norm_freq, single_tokenizer=single_tokenizer
    )

    waveform = np.random.random_sample([2, 3, num_f])
    asds = {"H1": np.random.random_sample(num_f), "L1": np.random.random_sample(num_f)}
    num_blocks = len(asds.keys())
    sample = {"waveform": waveform, "asds": asds}

    out = token_transformation(sample)
    print(out["position"].shape, out["blocks"].shape)
    # Check that first dimensions of waveform have expected shape
    assert out["waveform"].shape[:-1] == (num_tokens, num_blocks)
    # Check that position have expected shape
    assert out["position"].shape == (num_tokens, num_blocks, 2)
    # Check that position values match with initial f_min & f_max
    assert out["position"].min() == f_min
    assert out["position"].max() >= f_max
    # Check that block information has expected shape
    assert out["blocks"].shape == (num_blocks,)

