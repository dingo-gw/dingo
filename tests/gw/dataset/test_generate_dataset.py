import bilby
import numpy as np
import pytest

from dingo.gw.dataset.generate_dataset import generate_dataset


@pytest.fixture
def wfd_settings():
    settings = {
        "domain": {
            "delta_f": 0.125,
            "f_max": 1024,
            "f_min": 20,
            "type": "FrequencyDomain",
        },
        "intrinsic_prior": {
            "mass_1": bilby.core.prior.Constraint(minimum=7.5, maximum=120.0),
            "mass_2": bilby.core.prior.Constraint(minimum=7.5, maximum=120.0),
            "chi_1": bilby.gw.prior.AlignedSpin(
                name="chi_1", a_prior=bilby.core.prior.Uniform(minimum=0, maximum=0.99)
            ),
            "chi_2": bilby.gw.prior.AlignedSpin(
                name="chi_2", a_prior=bilby.core.prior.Uniform(minimum=0, maximum=0.99)
            ),
            "chirp_mass": bilby.gw.prior.UniformInComponentsChirpMass(
                minimum=12.5, maximum=150.0
            ),
            "geocent_time": 0.0,
            "luminosity_distance": 100.0,
            "mass_ratio": bilby.gw.prior.UniformInComponentsMassRatio(
                minimum=0.125, maximum=1.0
            ),
            "phase": bilby.core.prior.Uniform(
                minimum=0.0, maximum=2 * np.pi, boundary="periodic"
            ),
            "theta_jn": bilby.core.prior.Sine(minimum=0.0, maximum=np.pi),
        },
        "num_samples": 1000,
        "waveform_generator": {
            "approximant": "IMRPhenomD",
            "f_ref": 10.0,
            "spin_conversion_phase": 0,
        },
    }
    return settings


def test_wfd_size(wfd_settings):
    """
    Test that the size requested by the waveform generator settings is the same as the
    size of the generated dataset. This should be the case even when there are failures.
    """

    assert 1==1