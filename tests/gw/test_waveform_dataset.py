import os
import stat
import uuid

import numpy as np
import pytest

from dingo.gw.domains import Domain
from dingo.gw.waveform_dataset import WaveformDataset
from dingo.gw.waveform_dataset_generation.create_waveform_generation_bash_script import parse_args, generate_workflow


SETTINGS_YAML_SMALL = """\
# settings for domain of waveforms
domain_settings:
  name: UniformFrequencyDomain
  kwargs:
    f_min: 10.0
    f_max: 1024.0
    delta_f: 1.0

# settings for waveform generator
waveform_generator_settings:
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

reference_frequency: 20.0  # Hz

waveform_dataset_generation_settings:
  # Number of waveforms to generate for building the SVD basis
  num_wfs_basis: 10
  # Number of waveforms to generate for the waveform dataset
  num_wfs_dataset: 50
  # Truncate the SVD basis at this size. No truncation if zero.
  rb_max: 5
"""


@pytest.fixture
def generate_waveform_dataset_small(venv_dir='venv'):
    """
    Create a small waveform dataset on-the-fly
    in a temporary directory.

    venv_dir:
        The name of the folder that contains the dingo
        virtual environment relative to the 'dingo-devel' root.
    """
    # Figure out the path to the 'dingo-devel' root directory
    s = os.getcwd()
    root = s.split('dingo-devel')[0] + 'dingo-devel'
    venv_path = os.path.join(root, venv_dir)

    # Create temp directory and settings file
    path = os.path.join('/tmp', str(uuid.uuid4()))
    os.mkdir(path)
    with open(os.path.join(path, 'settings.yaml'), 'w') as fp:
        fp.writelines(SETTINGS_YAML_SMALL)

    # Mock up command line arguments for script generator
    args_in = ['--waveforms_directory', path,
               '--env_path', venv_path,
               '--num_threads', '4']
    args = parse_args(args_in)
    out_script = './waveform_generation_script.sh'
    if os.path.exists(out_script):
        os.remove(out_script)
    generate_workflow(args)

    # Now execute the generated workflow
    os.chmod(out_script, stat.S_IREAD | stat.S_IWRITE | stat.S_IXUSR)
    res = os.system(f'{out_script} > ./waveform_generation_script.log 2>&1')
    if not os.WIFEXITED(res):
        raise RuntimeError(f'waveform_generation_script.sh returned a '
                           f'nonzero exit code: {res}')

    return path


def test_load_waveform_dataset(generate_waveform_dataset_small):
    wfd_path = generate_waveform_dataset_small

    path = f'{wfd_path}/waveform_dataset.hdf5'
    wd = WaveformDataset(path)

    assert len(wd) > 0
    el = wd[0]
    assert isinstance(el, dict)
    assert set(el.keys()) == {'parameters', 'waveform'}
    assert isinstance(el['parameters'], dict)
    assert isinstance(el['waveform'], dict)
    assert isinstance(el['waveform']['h_plus'], np.ndarray)
    assert isinstance(el['waveform']['h_cross'], np.ndarray)

    # Check the associated domain
    assert len(wd.domain) > 0
    assert isinstance(wd.domain, Domain)
    assert isinstance(wd.domain.domain_dict, dict)

    # Check the associated data settings
    data_settings_keys = {'domain_settings', 'waveform_generator_settings',
                          'intrinsic_prior', 'reference_frequency',
                          'waveform_dataset_generation_settings'}
    assert set(wd.data_settings.keys()) == data_settings_keys
