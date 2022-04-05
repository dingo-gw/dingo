import pytest
import numpy as np
import torch
from bilby.core.prior import PriorDict
from bilby.gw.detector import InterferometerList

from dingo.gw.transforms import GNPEShiftDetectorTimes


@pytest.fixture
def gnpe_time_setup():
    kernel = "bilby.core.prior.Uniform(minimum=-0.001, maximum=0.001)"
    ifo_list = InterferometerList(["H1", "L1"])
    time_prior = PriorDict(
        {
            "geocent_time": "bilby.core.prior.Uniform(minimum=-0.1, maximum=0.1)",
            "H1_time": "bilby.core.prior.Uniform(minimum=-0.1, maximum=0.1)",
            "L1_time": "bilby.core.prior.Uniform(minimum=-0.1, maximum=0.1)",
        }
    )
    return kernel, ifo_list, time_prior


def test_gnpe_time_training(gnpe_time_setup):
    kernel, ifo_list, time_prior = gnpe_time_setup
    transform = GNPEShiftDetectorTimes(ifo_list, kernel, exact_global_equivariance=True)

    # During training, the sample is assumed to be *not* batched, just consisting of an
    # array of floats.
    extrinsic_parameters = time_prior.sample()
    sample = {"extrinsic_parameters": extrinsic_parameters}
    sample_new = transform(sample)
    extrinsic_parameters_new = sample_new["extrinsic_parameters"]

    # The first time proxy should be dropped for exact equivariance
    assert "H1_time_proxy" not in extrinsic_parameters_new
    assert "L1_time_proxy" in extrinsic_parameters_new

    for v in extrinsic_parameters_new.values():
        assert type(v) == float  # Correct type, not an array
    for k in ["H1_time", "L1_time"]:
        # Detector coalescence times should be close to 0 for data simplification.
        assert np.abs(extrinsic_parameters_new[k]) <= 0.001

    # Check particular sum rules.
    assert extrinsic_parameters_new["geocent_time"] == (
        extrinsic_parameters["geocent_time"]
        - extrinsic_parameters["H1_time"]
        + extrinsic_parameters_new["H1_time"]
    )
    assert extrinsic_parameters_new["L1_time_proxy"] == (
        extrinsic_parameters["L1_time"]
        - extrinsic_parameters_new["L1_time"]
        - extrinsic_parameters["H1_time"]
        + extrinsic_parameters_new["H1_time"]
    )


def test_gnpe_time_inference(gnpe_time_setup):
    kernel, ifo_list, time_prior = gnpe_time_setup
    transform = GNPEShiftDetectorTimes(
        ifo_list, kernel, exact_global_equivariance=True, inference=True
    )

    # For inference, the sample is assumed to be a torch tensor and batched.
    batch_size = 10
    extrinsic_parameters = time_prior.sample(batch_size)
    extrinsic_parameters.pop("geocent_time")  # Should not be included at this point.
    extrinsic_parameters = {k: torch.tensor(v) for k, v in extrinsic_parameters.items()}
    sample = {"extrinsic_parameters": extrinsic_parameters}
    sample_new = transform(sample)
    extrinsic_parameters_new = sample_new["extrinsic_parameters"]

    # The first time proxy should be dropped for exact equivariance
    assert "H1_time_proxy" not in extrinsic_parameters_new
    assert "L1_time_proxy" in extrinsic_parameters_new

    for v in extrinsic_parameters_new.values():
        assert type(v) == torch.Tensor and len(v) == batch_size

    # Check particular quantities differ by <= epsilon.
    assert torch.allclose(
        extrinsic_parameters_new["H1_time"],
        -extrinsic_parameters["H1_time"],
        atol=0.001,
    )
    assert torch.allclose(
        extrinsic_parameters_new["L1_time"],
        -extrinsic_parameters["L1_time"],
        atol=0.001,
    )

    # Check particular equalities.
    assert torch.equal(
        extrinsic_parameters_new["L1_time_proxy"],
        -extrinsic_parameters_new["L1_time"] + extrinsic_parameters_new["H1_time"],
    )
    assert torch.equal(
        extrinsic_parameters_new["geocent_time"], extrinsic_parameters_new["H1_time"]
    )
