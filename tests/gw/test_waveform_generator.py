from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.waveform_generator import WaveformGenerator, StandardizeParameters
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

@pytest.fixture
def precessing_spin_wf_parameters():
    parameters = {'chirp_mass': 34.0, 'mass_ratio': 0.35,
                  'a_1': 0.5, 'a_2': 0.2, 'tilt_1': 2*np.pi/3.0, 'tilt_2': np.pi/4.0,
                  'phi_12': np.pi/4.0, 'phi_jl': np.pi/3.0,
                  'theta_jn': 1.57, 'f_ref': 100.0, 'phase': 0.0, 'luminosity_distance': 1.0}
    approximant = 'IMRPhenomPv2'
    return parameters, approximant

@pytest.fixture(params=["aligned_spin_wf_parameters", "precessing_spin_wf_parameters"])
def wf_parameters(request):
    return request.getfixturevalue(request.param)


def test_waveform_generator_FD(uniform_fd_domain, wf_parameters):
    """Basic check that a waveform can be generated without error and
    it is consistent with the domain."""
    domain = uniform_fd_domain
    parameters, approximant = wf_parameters

    wf_gen = WaveformGenerator(approximant, domain)
    wf_dict = wf_gen.generate_hplus_hcross(parameters)

    assert len(wf_dict['h_plus']) == len(domain)
    assert domain()[domain.frequency_mask][0] == domain.f_min


def test_standardize_parameters_on_distribution():
    """Check standardization of samples from a multi-normal distribution."""
    mean_ = torch.tensor([3.0, 2.0, 8.0])
    std_ = torch.tensor([2.0, 4.0, 7.0])
    n_samples = 100000
    parameters = torch.distributions.Normal(mean_, std_).sample((n_samples,)).numpy()
    samples = {'parameters': parameters, 'waveform': None}
    tr = StandardizeParameters(mean_.numpy(), std_.numpy())
    samples_tr = tr(samples)
    parameters_tr = samples_tr['parameters']

    tol = 0.01
    assert np.all(np.abs(np.mean(parameters_tr, axis=0)) < tol)
    assert np.all(np.abs(np.std(parameters_tr, axis=0)) - np.ones(3) < tol)
