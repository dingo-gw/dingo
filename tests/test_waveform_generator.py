from dingo.gw.domains import UniformFrequencyDomain, TimeDomain
from dingo.gw.waveform_generator import WaveformGenerator, StandardizedDistribution
import pytest
import numpy as np
import torch
import torch.distributions


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
    wf_dict = wf_gen.generate_hplus_hcross(parameters)

    assert len(wf_dict['h_plus']) == len(domain)
    assert domain()[domain.frequency_mask][0] == domain.f_min


def test_standardized_distribution():
    """Check standardization of samples from a multi-normal distribution."""
    mean_ = torch.tensor([3.0, 2.0, 8.0])
    std_ = torch.tensor([2.0, 4.0, 7.0])
    base_dist = torch.distributions.Normal(mean_, std_)

    std_dist = StandardizedDistribution(base_dist, mean_, std_)
    Y2 = std_dist.sample((100000,))

    tol = 0.01
    assert np.all(np.abs(torch.mean(Y2, dim=0).numpy()) < tol)
    assert np.all(np.abs(torch.std(Y2, dim=0).numpy()) - np.ones(3) < tol)
