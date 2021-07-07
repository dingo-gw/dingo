from dingo.gw.domains import UniformFrequencyDomain, TimeDomain
from dingo.gw.waveform_generator import WaveformGenerator
import pytest
import numpy as np

@pytest.fixture
def uniform_fd_domain():
    p = {'f_min': 20.0, 'f_max': 4096.0, 'delta_f': 1.0/4.0, 'window_factor': 1.0}
    return UniformFrequencyDomain(**p)

@pytest.fixture
def aligned_spin_wf_parameters():
    parameters = {'chirp_mass': 34.0, 'mass_ratio': 0.35, 'chi_1': 0.2, 'chi_2': 0.1,
                  'theta_jn': 1.57, 'f_ref': 20.0, 'phase': 0.0, 'luminosity_distance': 1.0}
    approximant = 'IMRPhenomPv2'
    return parameters, approximant


def test_waveform_generator_FD(uniform_fd_domain, aligned_spin_wf_parameters):
    domain = uniform_fd_domain
    parameters, approximant = aligned_spin_wf_parameters

    wf_gen = WaveformGenerator(approximant, domain)
    hp, hc = wf_gen.generate_hplus_hcross(parameters)
    assert len(hp) == len(domain)
    assert domain()[domain.frequency_mask][0] == domain.f_min
