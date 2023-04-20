import pytest
import numpy as np
import pandas as pd
from dingo.gw.conversion import cartesian_spins, pe_spins, change_spin_conversion_phase


@pytest.fixture
def param_setup():
    f_ref = 20
    params = {
        "mass_ratio": 0.9834472418378746,
        "chirp_mass": 56.75982559976776,
        "luminosity_distance": 1000.0,
        "theta_jn": 1.913533991012538,
        "phase": 0.27093919407580236,
        "a_1": 0.23877963026418533,
        "a_2": 0.04520077119798103,
        "tilt_1": 1.5215996866276775,
        "tilt_2": 1.323159785592912,
        "phi_12": 0.8296989755334262,
        "phi_jl": 3.5628496275567447,
        "geocent_time": 0.0,
    }
    return f_ref, params


def test_spin_conversions(param_setup):
    f_ref, params = param_setup
    params_cart = cartesian_spins(params, f_ref)
    params_pe = pe_spins(params_cart, f_ref)
    params_pe_wrong = pe_spins(params_cart, f_ref + 10)

    assert params.keys() == params_pe.keys()
    assert np.sum([np.abs(v - params_pe[k]) for k, v in params.items()]) < 1e-10
    assert np.sum([np.abs(v - params_pe_wrong[k]) for k, v in params.items()]) > 1e-3


def test_change_of_spin_conversion_phase(param_setup):
    f_ref, params = param_setup
    params_pd = pd.DataFrame({k: np.array([v, v / 1.2]) for k, v in params.items()})

    # 1) check that only theta_jn and phi_jl are impacted
    params_pd_new = change_spin_conversion_phase(params_pd, f_ref, None, 0)
    diff = {
        k: np.sum(np.abs(params_pd_new[k] - params_pd[k])) for k in params_pd.keys()
    }
    for k in ["theta_jn", "phi_jl"]:
        assert diff[k] > 1e-3
    for k in [kk for kk in diff.keys() if kk not in ["theta_jn", "phi_jl"]]:
        assert diff[k] < 1e-10

    # 2) check that changing phases back and forth is consistent
    params_pd_new = params_pd.copy()
    params_pd_new = change_spin_conversion_phase(params_pd_new, f_ref, None, 0)
    params_pd_new = change_spin_conversion_phase(params_pd_new, f_ref, 0, 1)
    params_pd_new = change_spin_conversion_phase(params_pd_new, f_ref, 1, None)
    diff = {
        k: np.sum(np.abs(params_pd_new[k] - params_pd[k])) for k in params_pd.keys()
    }
    for v in diff.values():
        assert v < 1e-10
