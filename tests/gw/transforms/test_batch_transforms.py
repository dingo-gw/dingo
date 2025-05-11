import numpy as np
import torchvision
import pytest

from bilby.gw.detector import InterferometerList
from bilby.core.prior import Uniform

from dingo.gw.transforms import (
    SampleExtrinsicParameters,
    GetDetectorTimes,
    GNPECoalescenceTimes,
    ProjectOntoDetectors,
    SampleNoiseASD,
    WhitenAndScaleStrain,
    AddWhiteNoiseComplex,
    SelectStandardizeRepackageParameters,
    RepackageStrainsAndASDS,
    UnpackDict,
)
from dingo.gw.prior import default_extrinsic_dict
from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.noise.asd_dataset import ASDDataset


@pytest.fixture
def standardization_dict():
    return {
        "mean": {
            "chirp_mass": 10.0,
            "mass_ratio": 0.5,
        },
        "std": {
            "chirp_mass": 1.0,
            "mass_ratio": 0.1,
        },
    }


@pytest.fixture
def domain():
    return UniformFrequencyDomain(0, 1024, 0.125, 1.0)


def input_sample_batched(batch_size, input_domain):
    rand = np.random.rand(batch_size, len(input_domain)) + 1j * np.random.rand(
        batch_size, len(input_domain)
    )
    d = {
        "parameters": {
            "chirp_mass": np.random.rand(batch_size),
            "mass_ratio": np.random.rand(batch_size),
            "luminosity_distance": np.ones(batch_size) * 100.0,
            "geocent_time": np.zeros(batch_size),
        },
        "waveform": {
            "h_plus": rand,
            "h_cross": rand,
        },
    }

    return d


@pytest.fixture
def input_sample_unbatched(domain):
    rand = np.random.rand(len(domain)) + 1j * np.random.rand(len(domain)
    )
    d = {
        "parameters": {
            "chirp_mass": 12.3,
            "mass_ratio": 0.4,
            "luminosity_distance": 100.0,
            "geocent_time": 0.0,
        },
        "waveform": {
            "h_plus": rand,
            "h_cross": rand
        },
    }

    return d


@pytest.fixture
def asd_dataset(domain):
    asd = np.random.rand(5, len(domain))
    asds = {"H1": asd, "L1": asd}
    settings = {"domain_dict": domain.domain_dict}
    dictionary = {"asds": asds, "settings": settings}

    return ASDDataset(dictionary=dictionary)


@pytest.fixture
def transform_list(standardization_dict, asd_dataset, domain):
    ifo_list = InterferometerList(["H1", "L1"])
    ref_time = 1126259462.391
    transforms = [
        SampleExtrinsicParameters(default_extrinsic_dict),
        GetDetectorTimes(ifo_list, ref_time),
        GNPECoalescenceTimes(ifo_list, Uniform(minimum=-0.001, maximum=0.001), True),
        ProjectOntoDetectors(ifo_list, domain, ref_time),
        SampleNoiseASD(asd_dataset),
        WhitenAndScaleStrain(domain.noise_std),
        AddWhiteNoiseComplex(),
        SelectStandardizeRepackageParameters(
            {"inference_parameters": ["chirp_mass", "mass_ratio"]}, standardization_dict
        ),
        RepackageStrainsAndASDS(
            [ifo.name for ifo in ifo_list], first_index=domain.min_idx
        ),
        UnpackDict(["inference_parameters", "waveform"]),
    ]

    return transforms


# test that the transforms work for batched and for unbatched data
batch_sizes = [1, 2]
@pytest.mark.parametrize("batch_size", batch_sizes)
def test_batched_training_transforms(transform_list, batch_size, domain):
    input_sample = input_sample_batched(batch_size, domain)
    transforms = torchvision.transforms.Compose(transform_list)
    output = transforms(input_sample)
    assert output[0].shape[0] == batch_size
    assert output[1].shape[0] == batch_size

def test_unbatched_training_transforms(transform_list, input_sample_unbatched):
    transforms = torchvision.transforms.Compose(transform_list)
    output = transforms(input_sample_unbatched)
    assert output[0].ndim == 1
    assert output[1].ndim == 3