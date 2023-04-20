import pytest
import numpy as np
import torch
from bilby.core.prior import PriorDict
from bilby.gw.detector import InterferometerList

from dingo.gw.transforms import GNPECoalescenceTimes


@pytest.fixture
def gnpe_time_setup():
    kernel = "bilby.core.prior.Uniform(minimum=-0.001, maximum=0.001)"
    ifo_list = InterferometerList(["H1", "L1", "V1"])
    time_prior = PriorDict(
        {
            "geocent_time": "bilby.core.prior.Uniform(minimum=-0.1, maximum=0.1)",
            "H1_time": "bilby.core.prior.Uniform(minimum=-0.1, maximum=0.1)",
            "L1_time": "bilby.core.prior.Uniform(minimum=-0.1, maximum=0.1)",
            "V1_time": "bilby.core.prior.Uniform(minimum=-0.1, maximum=0.1)",
        }
    )
    return kernel, ifo_list, time_prior


def test_gnpe_time_training(gnpe_time_setup):
    kernel, ifo_list, time_prior = gnpe_time_setup
    ifos = [ifo.name for ifo in ifo_list]
    transform = GNPECoalescenceTimes(ifo_list, kernel, exact_global_equivariance=True)

    assert transform.context_parameters == [
        f"{ifo}_time_proxy_relative" for ifo in ifos[1:]
    ]

    # During training, the sample is assumed to be *not* batched, just consisting of an
    # array of floats.
    extrinsic_parameters = time_prior.sample()
    sample = {"extrinsic_parameters": extrinsic_parameters}
    sample_new = transform(sample)
    extrinsic_parameters_new = sample_new["extrinsic_parameters"]

    # All proxies should be included in new extrinsic parameters.
    assert {f"{ifo}_time_proxy" for ifo in ifos} <= extrinsic_parameters_new.keys()
    # Context parameters need to be in new extrinsic parameters.
    assert set(transform.context_parameters) <= extrinsic_parameters_new.keys()

    for v in extrinsic_parameters_new.values():
        assert type(v) == float  # Correct type, not an array
    for ifo in ifos:
        # Detector coalescence times should be close to 0 for data simplification.
        assert np.abs(extrinsic_parameters_new[f"{ifo}_time"]) <= 0.001

    # Check particular sum rules.
    assert np.isclose(
        extrinsic_parameters_new["geocent_time"],
        extrinsic_parameters["geocent_time"]
        - extrinsic_parameters[f"{ifos[0]}_time"]
        + extrinsic_parameters_new[f"{ifos[0]}_time"],
    )
    for ifo in ifos[1:]:
        assert np.isclose(
            extrinsic_parameters_new[f"{ifo}_time_proxy_relative"],
            extrinsic_parameters[f"{ifo}_time"]
            - extrinsic_parameters_new[f"{ifo}_time"]
            - extrinsic_parameters[f"{ifos[0]}_time"]
            + extrinsic_parameters_new[f"{ifos[0]}_time"],
        )


def test_gnpe_time_inference(gnpe_time_setup):
    kernel, ifo_list, time_prior = gnpe_time_setup
    ifos = [ifo.name for ifo in ifo_list]
    transform = GNPECoalescenceTimes(
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

    # All proxies should be included in new extrinsic parameters.
    assert {f"{ifo}_time_proxy" for ifo in ifos} <= extrinsic_parameters_new.keys()
    # Context parameters need to be in new extrinsic parameters.
    assert set(transform.context_parameters) <= extrinsic_parameters_new.keys()

    for v in extrinsic_parameters_new.values():
        assert type(v) == torch.Tensor and len(v) == batch_size

    # Check particular quantities differ by <= epsilon.
    for ifo in ifos:
        assert torch.allclose(
            extrinsic_parameters_new[f"{ifo}_time"],
            -extrinsic_parameters[f"{ifo}_time"],
            atol=0.001,
        )

    # Check particular equalities.
    for ifo in ifos[1:]:
        assert torch.equal(
            extrinsic_parameters_new[f"{ifo}_time_proxy_relative"],
            -extrinsic_parameters_new[f"{ifo}_time"]
            + extrinsic_parameters_new[f"{ifos[0]}_time"],
        )
    assert torch.equal(
        extrinsic_parameters_new["geocent_time"],
        extrinsic_parameters_new[f"{ifos[0]}_time"],
    )
