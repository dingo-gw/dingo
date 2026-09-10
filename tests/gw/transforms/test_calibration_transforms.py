import os

import numpy as np
import pytest
from bilby.gw.detector import InterferometerList

from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.transforms import (
    ApplyCalibrationToWaveform,
    SampleCalibrationParameters,
)

ENVELOPE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "calibration_envelope_test.txt"
)


@pytest.fixture
def calibration_setup():
    domain = UniformFrequencyDomain(20.0, 1024.0, delta_f=0.125)
    ifo_list = InterferometerList(["H1", "L1"])
    num_nodes = 10
    num_curves = 7
    sample_calibration = SampleCalibrationParameters(
        ifo_list,
        domain,
        calibration_envelope={ifo.name: ENVELOPE for ifo in ifo_list},
        num_calibration_curves=num_curves,
        num_calibration_nodes=num_nodes,
    )
    rng = np.random.default_rng(0)
    n = len(domain)
    waveform = {
        ifo.name: (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        * domain.frequency_mask
        for ifo in ifo_list
    }
    sample = {
        "parameters": {},
        "extrinsic_parameters": {},
        "waveform": waveform,
    }
    sample = sample_calibration(sample)
    return domain, ifo_list, sample, num_curves


def bilby_reference_curves(transform, ifo, domain, calib_params, num_curves):
    """Per-curve evaluation with bilby's CubicSpline, as done before vectorization."""
    prefix = f"recalib_{ifo.name}_"
    transform._ensure_calibration_model(ifo, 10)
    freqs = domain.sample_frequencies[domain.frequency_mask]
    curves = np.zeros((num_curves, len(domain)), dtype=complex)
    for i in range(num_curves):
        params_i = {k: v[i] for k, v in calib_params.items()}
        curves[i, domain.frequency_mask] = ifo.calibration_model.get_calibration_factor(
            freqs, prefix=prefix, **params_i
        )
    return curves


def test_vectorized_curves_match_bilby(calibration_setup):
    domain, ifo_list, sample, num_curves = calibration_setup
    transform = ApplyCalibrationToWaveform(ifo_list, domain)
    for ifo in ifo_list:
        prefix = f"recalib_{ifo.name}_"
        calib_params = {
            k: v for k, v in sample["extrinsic_parameters"].items() if k.startswith(prefix)
        }
        ref = bilby_reference_curves(transform, ifo, domain, calib_params, num_curves)
        curves = transform.calibration_curves(ifo, calib_params)
        assert curves.shape == (num_curves, len(domain))
        assert np.allclose(curves, ref, rtol=1e-12, atol=1e-14)


def test_expand_vs_attach(calibration_setup):
    domain, ifo_list, sample, num_curves = calibration_setup
    expanded = ApplyCalibrationToWaveform(ifo_list, domain)(sample)
    attached = ApplyCalibrationToWaveform(ifo_list, domain, expand_waveform=False)(sample)
    for ifo in ifo_list:
        assert expanded["waveform"][ifo.name].shape == (num_curves, len(domain))
        # attaching leaves the waveform untouched and provides the curves separately
        assert attached["waveform"][ifo.name] is sample["waveform"][ifo.name]
        curves = attached["calibration_curves"][ifo.name]
        assert curves.shape == (num_curves, len(domain))
        assert np.allclose(
            expanded["waveform"][ifo.name], sample["waveform"][ifo.name] * curves
        )
    # input sample must not be modified
    assert sample["waveform"]["H1"].ndim == 1


def test_scalar_parameters(calibration_setup):
    domain, ifo_list, sample, num_curves = calibration_setup
    scalar_sample = dict(sample)
    scalar_sample["extrinsic_parameters"] = {
        k: v[0] for k, v in sample["extrinsic_parameters"].items()
    }
    transform = ApplyCalibrationToWaveform(ifo_list, domain)
    out = transform(scalar_sample)
    for ifo in ifo_list:
        prefix = f"recalib_{ifo.name}_"
        calib_params = {
            k: v for k, v in sample["extrinsic_parameters"].items() if k.startswith(prefix)
        }
        ref = bilby_reference_curves(transform, ifo, domain, calib_params, 1)[0]
        assert out["waveform"][ifo.name].shape == (len(domain),)
        assert np.allclose(out["waveform"][ifo.name], sample["waveform"][ifo.name] * ref)


def test_no_calibration_parameters_is_noop(calibration_setup):
    domain, ifo_list, sample, _ = calibration_setup
    plain = dict(sample)
    plain["extrinsic_parameters"] = {"ra": 1.0}
    out = ApplyCalibrationToWaveform(ifo_list, domain)(plain)
    assert out["waveform"] is plain["waveform"]
