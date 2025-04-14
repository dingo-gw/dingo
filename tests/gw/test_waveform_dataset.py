import os
import tempfile
import uuid
from pathlib import Path
from typing import Generator
import yaml

import numpy as np
import pytest
import bilby

from dingo.gw.dataset.generate_dataset import _generate_dataset_main
from dingo.gw.dataset.waveform_dataset import WaveformDataset
from dingo.gw.domains import Domain
from dingo.gw.dataset import generate_dataset

SETTINGS_YAML_SMALL = """\
# settings for domain of waveforms
domain:
  type: UniformFrequencyDomain
  f_min: 10.0
  f_max: 1024.0
  delta_f: 1.0

# settings for waveform generator
waveform_generator:
  approximant: IMRPhenomPv2
  f_ref: 20.0

# settings for intrinsic prior over parameters
intrinsic_prior:
  # prior for non-fixed parameters
  mass_1: bilby.core.prior.Constraint(minimum=10.0, maximum=80.0)
  mass_2: bilby.core.prior.Constraint(minimum=10.0, maximum=80.0)
  mass_ratio: bilby.core.prior.Uniform(minimum=0.125, maximum=1.0)
  chirp_mass: bilby.core.prior.Uniform(minimum=25.0, maximum=100.0)
  phase: default
  a_1: bilby.core.prior.Uniform(minimum=0.0, maximum=0.88)
  a_2: bilby.core.prior.Uniform(minimum=0.0, maximum=0.88)
  tilt_1: default
  tilt_2: default
  phi_12: default
  phi_jl: default
  theta_jn: default
  # reference values for fixed (extrinsic) parameters
  luminosity_distance: 100.0 # Mpc
  geocent_time: 0.0 # s

num_samples: 50

compression:
  svd:
    num_training_samples: 10
    size: 5
"""


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """
    Fixture to provide a temporary directory path using pathlib.Path.

    Returns
    -------
    A pathlib.Path object pointing to the temporary directory.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture(scope="session")
def generate_waveform_dataset_small(temp_dir: Path) -> Path:
    """
    Create a small waveform dataset on-the-fly
    in a temporary directory.
    """

    # Create temp directory and settings file
    path = temp_dir / "tmp_test" / str(uuid.uuid4())
    path.mkdir(parents=True)
    settings_path = path / "settings.yaml"
    out_file = path / "waveform_dataset.hdf5"
    num_processes = 4

    with open(settings_path, "w") as fp:
        fp.writelines(SETTINGS_YAML_SMALL)

    _generate_dataset_main(str(settings_path), str(out_file), num_processes)

    return path


@pytest.mark.slow
def test_load_waveform_dataset(generate_waveform_dataset_small):
    wfd_path = generate_waveform_dataset_small

    path = f"{wfd_path}/waveform_dataset.hdf5"
    wfd = WaveformDataset(file_name=path, precision="single")

    assert len(wfd) > 0
    el = wfd[0]
    assert isinstance(el, dict)
    assert set(el.keys()) == {"parameters", "waveform"}
    assert isinstance(el["parameters"], dict)
    assert isinstance(el["waveform"], dict)
    assert isinstance(el["waveform"]["h_plus"], np.ndarray)
    assert isinstance(el["waveform"]["h_cross"], np.ndarray)

    # Check the associated domain
    assert len(wfd.domain) > 0
    assert isinstance(wfd.domain, Domain)
    assert isinstance(wfd.domain.domain_dict, dict)

    # Check the associated data settings
    data_settings_keys = {
        "domain",
        "waveform_generator",
        "intrinsic_prior",
        "num_samples",
        "compression",
    }
    assert set(wfd.settings.keys()) == data_settings_keys

    """Check that using a list of indices also works."""
    ind_list = [0, 1, 2]
    el_list = wfd.__getitems__(ind_list)
    assert isinstance(el_list, list)
    assert len(el_list) == len(ind_list)
    assert set(el_list[0].keys()) == {"parameters", "waveform"}
    assert isinstance(el_list[0]["parameters"], dict)
    assert isinstance(el_list[0]["waveform"], dict)
    assert isinstance(el_list[0]["waveform"]["h_plus"], np.ndarray)
    assert isinstance(el_list[0]["waveform"]["h_cross"], np.ndarray)


@pytest.mark.slow
def test_truncation_of_waveform(generate_waveform_dataset_small):
    """Check truncation of wfd."""
    wfd_path = generate_waveform_dataset_small

    path = f"{wfd_path}/waveform_dataset.hdf5"
    wfd = WaveformDataset(file_name=path, precision="single")
    el = wfd[0]

    f_min = wfd.domain.f_min
    f_max = wfd.domain.f_max
    delta_f = wfd.domain._delta_f

    # check that truncation works as intended when setting new range
    f_min_new = 20
    f_max_new = 100
    wfd2 = WaveformDataset(
        file_name=path, domain_update={"f_min": f_min_new, "f_max": f_max_new}
    )
    assert len(wfd2.domain) == len(wfd2.domain())
    # check that new domain settings are correctly adapted
    assert wfd2.domain.f_min == f_min_new
    assert wfd2.domain.f_max == f_max_new
    assert wfd2.domain._delta_f == wfd.domain._delta_f
    # check that truncation works as intended
    for pol in ["h_cross", "h_plus"]:
        # f_min_new to f_max_new check
        a = el["waveform"][pol][int(f_min_new / delta_f) : int(f_max_new / delta_f) + 1]
        b = wfd2[0]["waveform"][pol][int(f_min_new / delta_f) :]
        scale_factor = np.max(np.abs(a))
        assert len(a) == f_max_new / delta_f + 1 - f_min_new / delta_f
        assert np.allclose(b / scale_factor, a / scale_factor)
        assert not np.allclose(b / scale_factor, np.roll(a, 1) / scale_factor)

        # f_min to f_min_new check
        a = el["waveform"][pol][int(f_min / delta_f) : int(f_min_new / delta_f)]
        b = wfd2[0]["waveform"][pol][int(f_min / delta_f) : int(f_min_new / delta_f)]
        assert not np.allclose(b / scale_factor, a / scale_factor)

        # below f_min_new check
        assert np.all(wfd2[0]["waveform"][pol][: int(f_min_new)] == 0.0)
    assert len(wfd2.domain) == f_max_new / delta_f + 1
    assert len(wfd2.domain) == len(wfd2.domain())


@pytest.mark.slow
def test_load_waveform_dataset_with_leave_polarizations_on_disk(
    generate_waveform_dataset_small,
):
    wfd_path = generate_waveform_dataset_small

    path = f"{wfd_path}/waveform_dataset.hdf5"
    wfd = WaveformDataset(
        file_name=path,
        precision="single",
        leave_waveforms_on_disk=True,
    )

    assert len(wfd) > 0
    el = wfd[0]
    assert isinstance(el, dict)
    assert set(el.keys()) == {"parameters", "waveform"}
    assert isinstance(el["parameters"], dict)
    assert isinstance(el["waveform"], dict)
    assert isinstance(el["waveform"]["h_plus"], np.ndarray)
    assert isinstance(el["waveform"]["h_cross"], np.ndarray)

    """Check that using a list of indices also works."""
    ind_list = [0, 1, 2, 3, 4, 5]
    el_list = wfd.__getitems__(ind_list)
    assert isinstance(el_list, list)
    assert len(el_list) == len(ind_list)
    assert set(el_list[0].keys()) == {"parameters", "waveform"}
    assert isinstance(el_list[0]["parameters"], dict)
    assert isinstance(el_list[0]["waveform"], dict)
    assert isinstance(el_list[0]["waveform"]["h_plus"], np.ndarray)
    assert isinstance(el_list[0]["waveform"]["h_cross"], np.ndarray)

    """Check that using a randomly ordered list of indices provides the expected ordering."""
    ind_list_rand = np.random.permutation(ind_list).tolist()
    el_list_rand = wfd.__getitems__(ind_list_rand)
    np.testing.assert_equal([el_list[i] for i in ind_list_rand], el_list_rand)

    """Check that waveforms and parameters are loaded from disk equivalently to loading the full dataset."""
    wfd_full = WaveformDataset(
        file_name=path,
        precision="single",
        leave_waveforms_on_disk=False,
    )
    el_list_full = wfd_full.__getitems__(ind_list)
    np.testing.assert_equal(el_list, el_list_full)


@pytest.fixture
def wfd_settings():
    settings = yaml.safe_load(SETTINGS_YAML_SMALL)
    return settings


class BinaryPrior(bilby.prior.Prior):
    def __init__(self):
        super().__init__(name="failing_prior")

    def sample(self, size):
        return np.random.choice([30, -1], size=size, p=[0.8, 0.2])


@pytest.fixture
def binary_prior():
    return BinaryPrior()


def test_wfd_size(wfd_settings: str, binary_prior: BinaryPrior):
    """
    Test that the size requested by the waveform generator settings is the same as the
    size of the generated dataset. This should be the case even when there are failures
    in the waveform generation
    """
    # changing the waveform generator settings to create a prior which will create
    # failing waveforms for a fraction of the prior. I.e. can't generate negative
    # chirp masses so the waveform generator will fail
    wfd_settings["intrinsic_prior"]["chirp_mass"] = binary_prior
    del wfd_settings["intrinsic_prior"]["mass_1"]
    del wfd_settings["intrinsic_prior"]["mass_2"]
    del wfd_settings["compression"]
    wfd = generate_dataset(wfd_settings, 1)
    assert len(wfd) == wfd_settings["num_samples"]
