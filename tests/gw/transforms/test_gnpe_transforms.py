import lal
import pytest
import numpy as np
import torch
from bilby.core.prior import PriorDict
from bilby.gw.detector import InterferometerList

from dingo.gw.domains import FrequencyDomain
from dingo.gw.transforms import GNPECoalescenceTimes, GNPEChirp


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


@pytest.fixture
def gnpe_chirp_setup():
    # Use a constant kernel for the tests.
    kernel = "1.0"
    prior = PriorDict(
        {"chirp_mass": "bilby.core.prior.Uniform(minimum=20, maximum=30)"}
    )
    p = {"f_min": 20.0, "f_max": 1024.0, "delta_f": 1.0 / 4.0}
    domain = FrequencyDomain(**p)
    return GNPEChirp(kernel, domain), prior


def test_gnpe_time_training(gnpe_time_setup):
    kernel, ifo_list, time_prior = gnpe_time_setup
    transform = GNPECoalescenceTimes(ifo_list, kernel, exact_global_equivariance=True)

    assert transform.proxy_list == ["L1_time_proxy"]

    # During training, the sample is assumed to be *not* batched, just consisting of an
    # array of floats.
    extrinsic_parameters = time_prior.sample()
    sample = {"extrinsic_parameters": extrinsic_parameters}
    sample_new = transform(sample)
    extrinsic_parameters_new = sample_new["extrinsic_parameters"]

    # The first time proxy should be dropped for exact equivariance.
    assert "H1_time_proxy" not in extrinsic_parameters_new
    assert "L1_time_proxy" in extrinsic_parameters_new

    for v in extrinsic_parameters_new.values():
        assert type(v) == float  # Correct type, not an array
    for k in ["H1_time", "L1_time"]:
        # Detector coalescence times should be close to 0 for data simplification.
        assert np.abs(extrinsic_parameters_new[k]) <= 0.001

    # Check particular sum rules.
    assert np.isclose(
        extrinsic_parameters_new["geocent_time"],
        extrinsic_parameters["geocent_time"]
        - extrinsic_parameters["H1_time"]
        + extrinsic_parameters_new["H1_time"],
    )
    assert np.isclose(
        extrinsic_parameters_new["L1_time_proxy"],
        extrinsic_parameters["L1_time"]
        - extrinsic_parameters_new["L1_time"]
        - extrinsic_parameters["H1_time"]
        + extrinsic_parameters_new["H1_time"],
    )


def test_gnpe_time_inference(gnpe_time_setup):
    kernel, ifo_list, time_prior = gnpe_time_setup
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

    # The first time proxy should be dropped for exact equivariance.
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


def test_gnpe_chirp_training(gnpe_chirp_setup):
    transform, prior = gnpe_chirp_setup
    sample = {"parameters": prior.sample(), "extrinsic_parameters": {}, "waveform": {}}
    chirp_mass = sample["parameters"]["chirp_mass"]
    domain = transform.domain
    f = domain.sample_frequencies
    f = f.astype(np.float64)  # Check effect of changing precision.
    mf = chirp_mass * f
    mf[0] = 1.0
    phase = (3 / 128) * (np.pi * lal.GMSUN_SI / lal.C_SI ** 3 * mf) ** (-5 / 3)
    # Each detector gets the same waveform. The transform also uses the same fiducial
    # waveform for each detector.
    sample["waveform"]["H1"] = np.exp(-1j * phase).astype(np.complex64)
    sample["waveform"]["L1"] = sample["waveform"]["H1"]
    new_sample = transform(sample)
    assert isinstance(new_sample["waveform"], dict)
    for pol, data in new_sample["waveform"].items():
        assert data.shape == sample["waveform"][pol].shape
        assert isinstance(data, np.ndarray)

        # Check that the transform removes the initial phase.
        #
        # For low frequencies, single-precision floats are not adequate to achieve
        # phase accuracy. Since above f_min, things seem to work fine, we limit our
        # test to this regime. Even above f_min, the tolerance must be raised to 1e-4.
        #
        # Oddly, the error all occurs in the imaginary part of the data.
        # TODO: Figure out why. Maybe it's because the imaginary part is close to zero?
        assert np.allclose(data[domain.min_idx :], 1.0, atol=1e-4)

    assert new_sample["extrinsic_parameters"]["chirp_mass_proxy"] == chirp_mass


def test_gnpe_chirp_inference(gnpe_chirp_setup):
    transform, prior = gnpe_chirp_setup
    sample = {"parameters": {}, "extrinsic_parameters": {}, "waveform": {}}
    batch_size = 5
    num_detectors = 2
    chirp_mass = prior.sample(batch_size)["chirp_mass"]
    chirp_mass = torch.tensor(chirp_mass)  # torch.float64
    sample["extrinsic_parameters"]["chirp_mass"] = chirp_mass.type(torch.float32)
    domain = transform.domain
    f = domain.sample_frequencies_torch[domain.min_idx :]
    mf = torch.outer(chirp_mass, f)
    phase = (3 / 128) * (np.pi * lal.GMSUN_SI / lal.C_SI ** 3 * mf) ** (-5 / 3)
    waveform = torch.exp(-1j * phase)
    waveform = torch.stack((waveform.real, waveform.imag), dim=1)
    waveform = waveform[:, None, :, :]
    waveform = waveform.expand(-1, num_detectors, -1, -1)
    sample["waveform"] = waveform.type(torch.float32)

    new_sample = transform(sample)
    assert isinstance(new_sample["waveform"], torch.Tensor)
    assert new_sample["waveform"].dtype == torch.float32
    assert new_sample["waveform"].shape == (
        batch_size,
        num_detectors,
        2,
        len(domain) - domain.min_idx,
    )
    # Verify real and complex parts of transformed waveform are 1, 0, respectively.
    assert torch.allclose(new_sample["waveform"][..., 0, :], torch.tensor(1.0))
    # TODO: Figure out why imaginary part has worse accuracy.
    assert torch.allclose(
        new_sample["waveform"][..., 1, :], torch.tensor(0.0), atol=1e-4
    )
    assert torch.equal(
        new_sample["extrinsic_parameters"]["chirp_mass_proxy"],
        sample["extrinsic_parameters"]["chirp_mass"],
    )
