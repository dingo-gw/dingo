import bilby
import numpy as np
import pandas as pd
import pytest
from bilby.gw import GravitationalWaveTransient
from bilby.gw import WaveformGenerator as BilbyWaveformGenerator
from bilby.gw.detector import InterferometerList

from dingo.gw.domains import build_domain
from dingo.gw.injection import Injection
from dingo.gw.likelihood import StationaryGaussianGWLikelihood
from dingo.gw.prior import (
    build_prior_with_defaults,
    default_intrinsic_dict,
    default_extrinsic_dict,
)
from dingo.gw.waveform_generator import NewInterfaceWaveformGenerator


@pytest.fixture(params=["IMRPhenomXPHM", "SEOBNRv4PHM", "SEOBNRv5PHM"])
def approximant(request):
    return request.param


@pytest.fixture
def bbh_parameters():
    return {
        "mass_1": 60.29442201204798,
        "mass_2": 25.460299253933126,
        "phase": 2.346269257440926,
        "a_1": 0.07104636316747037,
        "a_2": 0.7853578509086726,
        "tilt_1": 1.8173336549500292,
        "tilt_2": 0.4380213394743055,
        "phi_12": 5.892609139936818,
        "phi_jl": 1.6975651971466297,
        "theta_jn": 1.0724395559873239,
        "luminosity_distance": 1000.0,
        "geocent_time": 0.0,
        "ra": 1.0,
        "dec": 2.0,
        "psi": 2.5,
    }


@pytest.fixture  # (params=["IMRPhenomXPHM", "SEOBNRv4PHM", "SEOBNRv5PHM"])
def injection(approximant):
    # approximant = request.param
    wfg_kwargs = {
        "approximant": approximant,
        "f_ref": 20.0,
        "spin_conversion_phase": None,
    }
    if approximant == "SEOBNRv4PHM":
        wfg_kwargs["f_start"] = 12.0
    elif approximant == "SEOBNRv5PHM":
        wfg_kwargs["f_start"] = 12.0
        wfg_kwargs["new_interface"] = True

    wfg_domain = build_domain(
        {
            "type": "FrequencyDomain",
            "f_min": 20,
            "f_max": 1024,
            "delta_f": 0.125,
            "window_factor": 1.0,
        }
    )
    data_domain = build_domain(
        {
            "type": "FrequencyDomain",
            "f_min": 20.0,
            "f_max": 1024.0,
            "delta_f": 0.125,
            "window_factor": 0.9374713897717841,
        }
    )
    ifo_list = ["H1", "L1"]
    t_ref = 1126259462.4
    injection = Injection(
        prior=None,
        wfg_kwargs=wfg_kwargs,
        wfg_domain=wfg_domain,
        data_domain=data_domain,
        ifo_list=ifo_list,
        t_ref=t_ref,
    )

    # Use default ASD from Bilby interferometers
    ifos = bilby.gw.detector.InterferometerList(ifo_list)
    injection.asd = {
        ifo.name: ifo.power_spectral_density.get_amplitude_spectral_density_array(
            data_domain.sample_frequencies
        )
        for ifo in ifos
    }
    injection.asd = {
        ifo: data_domain.update_data(asd, low_value=1e-20)
        for ifo, asd in injection.asd.items()
    }

    return injection


@pytest.fixture
def data(injection, bbh_parameters):
    return injection.injection(bbh_parameters)


@pytest.fixture
def prior():
    prior_dict = default_intrinsic_dict | default_extrinsic_dict
    return build_prior_with_defaults(prior_dict)


@pytest.fixture
def bilby_likelihood(injection, data):
    post_trigger_duration = 2.0

    domain = injection.waveform_generator.domain

    waveform_arguments = {
        "waveform_approximant": injection.waveform_generator.approximant_str,
        "reference_frequency": injection.waveform_generator.f_ref,
        "minimum_frequency": injection.waveform_generator.f_start
        if injection.waveform_generator.f_start
        else domain.f_min,
    }

    # Setup is based on https://lscsoft.docs.ligo.org/bilby/compact-binary-coalescence-parameter-estimation.html

    if isinstance(injection.waveform_generator, NewInterfaceWaveformGenerator):
        frequency_domain_source_function = bilby.gw.source.gwsignal_binary_black_hole
    else:
        frequency_domain_source_function = bilby.gw.source.lal_binary_black_hole

    bilby_wfg = BilbyWaveformGenerator(
        duration=domain.duration,
        sampling_frequency=domain.sampling_rate,
        frequency_domain_source_model=frequency_domain_source_function,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
    )

    ifo_list = []
    for ifo in injection.ifo_list:
        ifo = bilby.gw.detector.get_empty_interferometer(ifo.name)
        ifo.duration = domain.duration
        ifo.sampling_frequency = domain.sampling_rate
        ifo.set_strain_data_from_frequency_domain_strain(
            injection.data_domain.time_translate_data(
                data["waveform"][ifo.name], -post_trigger_duration
            ),
            sampling_frequency=domain.sampling_rate,
            duration=domain.duration,
        )
        ifo.strain_data.roll_off = 0.4
        ifo.strain_data.window_factor = injection.data_domain.window_factor
        ifo.start_time = injection.t_ref + post_trigger_duration - domain.duration
        ifo_list.append(ifo)
    ifo_list = InterferometerList(ifo_list)

    bilby_likelihood = GravitationalWaveTransient(
        interferometers=ifo_list,
        waveform_generator=bilby_wfg,
    )

    return bilby_likelihood


@pytest.fixture
def dingo_likelihood(injection, data):
    if isinstance(injection.waveform_generator, NewInterfaceWaveformGenerator):
        new_interface = True
    else:
        new_interface = False

    wfg_kwargs = dict(
        approximant=injection.waveform_generator.approximant_str,
        f_ref=injection.waveform_generator.f_ref,
        f_start=injection.waveform_generator.f_start,
        spin_conversion_phase=injection.waveform_generator.spin_conversion_phase,
        new_interface=new_interface,
    )

    dingo_likelihood = StationaryGaussianGWLikelihood(
        wfg_kwargs=wfg_kwargs,
        wfg_domain=injection.waveform_generator.domain,
        data_domain=injection.data_domain,
        event_data=data,
        t_ref=injection.t_ref,
    )

    return dingo_likelihood


def test_gw_likelihood(prior, bilby_likelihood, dingo_likelihood):
    # Evaluate the likelihood at a new parameter value.
    p = prior.sample()
    p = {
        k: float(v) for k, v in p.items()
    }  # FIXME: Initially a mix of float and np.float64

    # Evaluate Dingo and Bilby log likelihoods and check that they match.
    log_likelihood_dingo = dingo_likelihood.log_likelihood(p)

    # Bilby treats the time of coalescence differently.
    p_bilby = p.copy()
    p_bilby["geocent_time"] += dingo_likelihood.t_ref
    bilby_likelihood.parameters = p_bilby
    log_likelihood_bilby = bilby_likelihood.log_likelihood()

    assert np.abs(log_likelihood_bilby - log_likelihood_dingo) < 1e-1


@pytest.mark.parametrize("approximant", ["IMRPhenomXPHM"])
def test_gw_likelihood_multi(prior, dingo_likelihood):
    p = pd.DataFrame(prior.sample(5))

    # Check that multiprocessing log likelihoods gives the same result as single processing.
    log_likelihoods_multi, supplement_multi = dingo_likelihood.log_likelihood_multi(
        p,
        include_supplemental=True,
        num_processes=2,
    )
    supplement_multi = pd.DataFrame(supplement_multi)

    for i, row in p.iterrows():
        log_likelihood_single, supplement_single = dingo_likelihood.log_likelihood(
            row.to_dict(), include_supplemental=True
        )
        assert log_likelihood_single == log_likelihoods_multi[i]
        assert supplement_single == supplement_multi.iloc[i].to_dict()
