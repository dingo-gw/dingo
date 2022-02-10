import os
import uuid

import numpy as np
import pytest

from dingo.gw.domains import Domain
from dingo.gw.dataset.waveform_dataset import WaveformDataset
from dingo.gw.dataset.generate_dataset_dag import create_args_string

SETTINGS_YAML_SMALL = """\
# settings for domain of waveforms
domain:
  type: FrequencyDomain
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
    # s = os.getcwd()
    # root = s.split('dingo-devel')[0]  # + 'dingo-devel'
    # venv_path = os.path.join(root, venv_dir)
    # generate_waveform_path = os.path.join(venv_path, 'bin/dingo_generate_waveforms')

    # Create temp directory and settings file
    path = os.path.join('./tmp_test', str(uuid.uuid4()))
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'settings.yaml'), 'w') as fp:
        fp.writelines(SETTINGS_YAML_SMALL)

    # Mock up command line arguments for script
    args_in = {'settings_file': os.path.join(path, 'settings.yaml'),
               'out_file': os.path.join(path, 'waveform_dataset.hdf5'),
               'num_processes': '4'}
    args_string = create_args_string(args_in)
    res = os.system('dingo_generate_dataset ' + args_string)
    if not os.WIFEXITED(res):
        raise RuntimeError(f'dingo_generate_waveforms returned a '
                           f'nonzero exit code: {res}')

    # args = parse_args(args_in)
    # out_script = './waveform_generation_script.sh'
    # if os.path.exists(out_script):
    #     os.remove(out_script)
    # generate_workflow(args)
    #
    # # Now execute the generated workflow
    # os.chmod(out_script, stat.S_IREAD | stat.S_IWRITE | stat.S_IXUSR)
    # res = os.system(f'{out_script} > ./waveform_generation_script.log 2>&1')
    # if not os.WIFEXITED(res):
    #     raise RuntimeError(f'waveform_generation_script.sh returned a '
    #                        f'nonzero exit code: {res}')

    return path

@pytest.mark.slow
def test_load_waveform_dataset(generate_waveform_dataset_small):
    wfd_path = generate_waveform_dataset_small

    path = f'{wfd_path}/waveform_dataset.hdf5'
    wd = WaveformDataset(file_name=path, precision='single')

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
    data_settings_keys = {'domain', 'waveform_generator',
                          'intrinsic_prior', 'num_samples',
                          'compression'}
    assert set(wd.settings.keys()) == data_settings_keys


    """Check truncation of wd. Ideally, this should be an individual test, 
    but since the setup takes so long it is added here."""

    f_min = wd.domain.f_min
    f_max = wd.domain.f_max
    delta_f = wd.domain._delta_f

    # check that truncation works as intended when setting new range
    f_min_new = 20
    f_max_new = 100
    wd2 = WaveformDataset(path, domain_update={'f_min': f_min_new, 'f_max': f_max_new})
    assert len(wd2.domain) == len(wd2.domain())
    # check that new domain settings are correctly adapted
    assert wd2.domain.f_min == f_min_new
    assert wd2.domain.f_max == f_max_new
    assert wd2.domain._delta_f == wd.domain._delta_f
    # check that truncation works as intended
    for pol in ['h_cross', 'h_plus']:
        # f_min_new to f_max_new check
        a = el['waveform'][pol][int(f_min_new/delta_f):int(f_max_new/delta_f)+1]
        b = wd2[0]['waveform'][pol][int(f_min_new/delta_f):]
        scale_factor = np.max(np.abs(a))
        assert len(a) == f_max_new / delta_f + 1 - f_min_new / delta_f
        assert np.allclose(b / scale_factor, a / scale_factor)
        assert not np.allclose(b / scale_factor, np.roll(a, 1) / scale_factor)

        # f_min to f_min_new check
        a = el['waveform'][pol][int(f_min / delta_f):int(f_min_new / delta_f)]
        b = wd2[0]['waveform'][pol][int(f_min / delta_f):int(f_min_new / delta_f)]
        assert not np.allclose(b / scale_factor, a / scale_factor)

        # below f_min_new check
        assert np.all(wd2[0]['waveform'][pol][:int(f_min_new)] == 0.0)
    assert len(wd2.domain) == f_max_new / delta_f + 1
    assert len(wd2.domain) == len(wd2.domain())