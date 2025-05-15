import numpy as np
import pytest

from dingo.gw.domains import UniformFrequencyDomain, MultibandedFrequencyDomain
from dingo.gw.transforms import MaskDataForFrequencyRangeUpdate


@pytest.fixture
def strain_tokenization_setup():
    f_min = 20.0
    f_max = 1024.0
    T = 8.0
    domain = UniformFrequencyDomain(f_min, f_max, delta_f=1 / T)
    num_f = len(domain.sample_frequencies)

    batch_size = 100
    waveform = {
        "H1": np.ones([batch_size, num_f], dtype=np.complex64),
        "L1": np.ones([batch_size, num_f], dtype=np.complex64),
        "V1": np.ones([batch_size, num_f], dtype=np.complex64),
    }
    asds = {
        "H1": np.random.random([batch_size, num_f]),
        "L1": np.random.random([batch_size, num_f]),
        "V1": np.random.random([batch_size, num_f]),
    }

    sample = {"waveform": waveform, "asds": asds}

    return domain, sample, batch_size


@pytest.fixture
def strain_tokenization_setup_no_batch():
    f_min = 20.0
    f_max = 1024.0
    T = 8.0
    domain = UniformFrequencyDomain(f_min, f_max, delta_f=1 / T)
    num_f = len(domain.sample_frequencies)

    waveform = {
        "H1": np.ones([num_f], dtype=np.complex64),
        "L1": np.ones([num_f], dtype=np.complex64),
        "V1": np.ones([num_f], dtype=np.complex64),
    }
    asds = {
        "H1": np.random.random([num_f]),
        "L1": np.random.random([num_f]),
        "V1": np.random.random([num_f]),
    }

    sample = {"waveform": waveform, "asds": asds}

    return domain, sample, 0


@pytest.fixture
def strain_tokenization_setup_single_batch():
    f_min = 20.0
    f_max = 1024.0
    T = 8.0
    domain = UniformFrequencyDomain(f_min, f_max, delta_f=1 / T)
    num_f = len(domain.sample_frequencies)

    batch_size = 1
    waveform = {
        "H1": np.ones([batch_size, num_f], dtype=np.complex64),
        "L1": np.ones([batch_size, num_f], dtype=np.complex64),
        "V1": np.ones([batch_size, num_f], dtype=np.complex64),
    }
    asds = {
        "H1": np.random.random([batch_size, num_f]),
        "L1": np.random.random([batch_size, num_f]),
        "V1": np.random.random([batch_size, num_f]),
    }

    sample = {"waveform": waveform, "asds": asds}

    return domain, sample, batch_size


@pytest.fixture
def strain_tokenization_setup_mfd():
    nodes = [20.0, 34.0, 46.0, 62.0, 78.0, 1038.0]
    f_min = 20.0
    f_max = 1038.0
    T = 8.0
    base_domain = UniformFrequencyDomain(f_min=f_min, f_max=f_max, delta_f=1 / T)
    domain = MultibandedFrequencyDomain(
        nodes=nodes, delta_f_initial=1 / T, base_domain=base_domain
    )
    num_f = domain.frequency_mask_length

    batch_size = 100
    waveform = {
        "H1": np.ones([batch_size, num_f], dtype=np.complex64),
        "L1": np.ones([batch_size, num_f], dtype=np.complex64),
        "V1": np.ones([batch_size, num_f], dtype=np.complex64),
    }
    asds = {
        "H1": np.random.random([batch_size, num_f]),
        "L1": np.random.random([batch_size, num_f]),
        "V1": np.random.random([batch_size, num_f]),
    }

    sample = {"waveform": waveform, "asds": asds}

    return domain, sample, batch_size


@pytest.mark.parametrize(
    "setup",
    [
        "strain_tokenization_setup",
        "strain_tokenization_setup_no_batch",
        "strain_tokenization_setup_single_batch",
        "strain_tokenization_setup_mfd",
    ],
)
def test_MaskDataForFrequencyRangeUpdate(request, setup):
    domain, sample, batch_size = request.getfixturevalue(setup)

    frequency_update = {
        "minimum_frequency": {"H1": 30.0, "L1": 20, "V1": 30.0},
        "maximum_frequency": 1000.0,
        "suppress": {"V1": [50.0, 51.0]},
    }

    # Initialize transform
    trafo = MaskDataForFrequencyRangeUpdate(
        domain=domain,
        minimum_frequency=frequency_update["minimum_frequency"],
        maximum_frequency=frequency_update["maximum_frequency"],
        suppress_range=frequency_update["suppress"],
        ifos=[d for d in sample["waveform"].keys()],
    )
    # Pass data through transform
    out = trafo(sample)

    # Check minimum_frequencies
    mask = np.logical_and(
        domain.frequency_mask,
        domain.sample_frequencies < frequency_update["minimum_frequency"]["H1"],
    )
    if batch_size > 0:
        mask = np.repeat(mask[np.newaxis, :], axis=0, repeats=batch_size)
    assert np.all(out["waveform"]["H1"][mask] == 0.0)
    assert np.all(out["asds"]["H1"][mask] == 1.0)
    mask = np.logical_and(
        domain.frequency_mask,
        domain.sample_frequencies < frequency_update["minimum_frequency"]["L1"],
    )
    if batch_size > 0:
        mask = np.repeat(mask[np.newaxis, :], axis=0, repeats=batch_size)
    assert np.all(out["waveform"]["L1"][mask] == 0.0)
    assert np.all(out["asds"]["L1"][mask] == 1.0)
    mask = np.logical_and(
        domain.frequency_mask,
        domain.sample_frequencies < frequency_update["minimum_frequency"]["V1"],
    )
    if batch_size > 0:
        mask = np.repeat(mask[np.newaxis, :], axis=0, repeats=batch_size)
    assert np.all(out["waveform"]["V1"][mask] == 0.0)
    assert np.all(out["asds"]["V1"][mask] == 1.0)

    # Check maximum frequencies
    mask = np.logical_and(
        domain.frequency_mask,
        domain.sample_frequencies > frequency_update["maximum_frequency"],
    )
    if batch_size > 0:
        mask = np.repeat(mask[np.newaxis, :], axis=0, repeats=batch_size)
    assert np.all([np.all(v[mask] == 0.0) for v in sample["waveform"].values()])
    assert np.all([np.all(v[mask] == 1.0) for v in sample["asds"].values()])

    # Check suppress
    mask = np.logical_and(
        domain.sample_frequencies >= frequency_update["suppress"]["V1"][0],
        domain.sample_frequencies <= frequency_update["suppress"]["V1"][1],
    )
    mask = np.logical_and(domain.frequency_mask, mask)
    if batch_size > 0:
        mask = np.repeat(mask[np.newaxis, :], axis=0, repeats=batch_size)
    assert np.all(out["waveform"]["V1"][mask] == 0.0)
    assert np.all(out["asds"]["V1"][mask] == 1.0)
