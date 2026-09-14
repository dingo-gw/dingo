"""
The calibration spline nodes span the event's per-detector analysis range -- the
range the likelihood masks its ASDs to -- rather than the data-domain bounds, in both
the marginalization transform and the spline applied to the waveform. Bilby places
its nodes over each interferometer's own range in the same way, so the two agree
whenever the ranges do.
"""

import numpy as np
import pytest
from bilby.gw.detector import InterferometerList, calibration
from bilby.gw.prior import CalibrationPriorDict

from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.likelihood import StationaryGaussianGWLikelihood
from dingo.gw.transforms import ApplyCalibrationToWaveform, SampleCalibrationParameters

DETECTORS = ["H1", "L1"]
N_NODES = 5
BOUNDS = {"H1": (30.0, 200.0), "L1": (25.0, 128.0)}


def _write_envelope(path):
    """A minimal LVC-format envelope (frequency, median amp/phase, -1 and +1 sigma
    amp/phase) spanning the test domain."""
    freqs = np.geomspace(15.0, 300.0, 8)
    data = np.column_stack(
        [
            freqs,
            np.ones_like(freqs),
            np.zeros_like(freqs),
            np.full_like(freqs, 0.99),
            np.full_like(freqs, -0.01),
            np.full_like(freqs, 1.01),
            np.full_like(freqs, 0.01),
        ]
    )
    np.savetxt(path, data)


@pytest.fixture
def envelopes(tmp_path):
    paths = {}
    for ifo in DETECTORS:
        path = tmp_path / f"{ifo}.txt"
        _write_envelope(path)
        paths[ifo] = str(path)
    return paths


@pytest.fixture
def domain():
    return UniformFrequencyDomain(20.0, 256.0, 0.5)


def _node_frequencies(prior, ifo):
    return np.array(
        [prior[f"recalib_{ifo}_frequency_{i}"].peak for i in range(N_NODES)]
    )


def _assert_prior_matches_bilby(prior, envelope, f_min, f_max, ifo):
    expected = CalibrationPriorDict.from_envelope_file(
        envelope, f_min, f_max, N_NODES, ifo, correction_type="data"
    )
    assert np.allclose(
        _node_frequencies(prior, ifo),
        np.exp(np.linspace(np.log(f_min), np.log(f_max), N_NODES)),
    )
    for i in range(N_NODES):
        for quantity in ("amplitude", "phase"):
            name = f"recalib_{ifo}_{quantity}_{i}"
            assert prior[name].mu == expected[name].mu
            assert prior[name].sigma == expected[name].sigma


@pytest.mark.parametrize("bounds", [None, BOUNDS])
def test_sampled_calibration_prior_nodes_span_bounds(domain, envelopes, bounds):
    transform = SampleCalibrationParameters(
        InterferometerList(DETECTORS),
        domain,
        envelopes,
        num_calibration_curves=3,
        num_calibration_nodes=N_NODES,
        frequency_bounds=bounds,
    )
    for ifo in DETECTORS:
        f_min, f_max = (domain.f_min, domain.f_max) if bounds is None else bounds[ifo]
        _assert_prior_matches_bilby(
            transform.calibration_prior[ifo], envelopes[ifo], f_min, f_max, ifo
        )


def _sample_with_zero_calibration(domain):
    n_bins = len(domain)
    extrinsic = {}
    for ifo in DETECTORS:
        for i in range(N_NODES):
            extrinsic[f"recalib_{ifo}_amplitude_{i}"] = 0.0
            extrinsic[f"recalib_{ifo}_phase_{i}"] = 0.0
    return {
        "waveform": {ifo: np.ones(n_bins, dtype=complex) for ifo in DETECTORS},
        "extrinsic_parameters": extrinsic,
    }


def test_apply_calibration_spline_follows_bounds_and_rebuilds_on_change(domain):
    ifos = InterferometerList(DETECTORS)
    ApplyCalibrationToWaveform(ifos, domain, frequency_bounds=BOUNDS)(
        _sample_with_zero_calibration(domain)
    )
    for ifo in ifos:
        f_min, f_max = BOUNDS[ifo.name]
        expected = calibration.CubicSpline(
            f"recalib_{ifo.name}_", f_min, f_max, N_NODES
        )
        assert np.array_equal(
            ifo.calibration_model._log_spline_points, expected._log_spline_points
        )
    # A transform with other bounds on the same interferometers rebuilds the
    # spline (the model is cached on the interferometer object).
    ApplyCalibrationToWaveform(ifos, domain)(_sample_with_zero_calibration(domain))
    for ifo in ifos:
        assert ifo.calibration_model.minimum_frequency == domain.f_min
        assert ifo.calibration_model.maximum_frequency == domain.f_max


def test_likelihood_masks_asds_and_places_nodes_over_the_same_range(domain, envelopes):
    frequency_update = {
        "minimum_frequency": {"H1": 30.0, "L1": 25.0},
        "maximum_frequency": 200.0,
    }
    mask = domain.frequency_mask
    event_data = {
        "waveform": {d: np.where(mask, (1.0 + 1j) * 1e-21, 0.0) for d in DETECTORS},
        "asds": {d: np.where(mask, 1e-21, 1.0) for d in DETECTORS},
    }
    likelihood = StationaryGaussianGWLikelihood(
        wfg_kwargs={"approximant": "IMRPhenomD", "f_ref": 20.0},
        wfg_domain=domain,
        data_domain=domain,
        event_data=event_data,
        t_ref=1126259462.4,
        calibration_marginalization_kwargs={
            "calibration_envelope": envelopes,
            "num_calibration_nodes": N_NODES,
            "num_calibration_curves": 3,
        },
        frequency_update=frequency_update,
    )
    transforms = likelihood.projection_transforms.transforms
    sampler = next(t for t in transforms if isinstance(t, SampleCalibrationParameters))
    applier = next(t for t in transforms if isinstance(t, ApplyCalibrationToWaveform))
    f = domain.sample_frequencies
    for ifo in DETECTORS:
        f_min = frequency_update["minimum_frequency"][ifo]
        f_max = frequency_update["maximum_frequency"]
        assert sampler.frequency_bounds[ifo] == (f_min, f_max)
        assert applier.frequency_bounds[ifo] == (f_min, f_max)
        _assert_prior_matches_bilby(
            sampler.calibration_prior[ifo], envelopes[ifo], f_min, f_max, ifo
        )
        # The ASD mask and the node range are the same per-detector range.
        inside = (f >= f_min) & (f <= f_max)
        assert np.all(likelihood.asd[ifo][inside] == 1e-21)
        assert np.all(likelihood.asd[ifo][~inside] == 1.0)


def test_likelihood_without_range_update_uses_domain_bounds(domain, envelopes):
    mask = domain.frequency_mask
    event_data = {
        "waveform": {d: np.where(mask, (1.0 + 1j) * 1e-21, 0.0) for d in DETECTORS},
        "asds": {d: np.where(mask, 1e-21, 1.0) for d in DETECTORS},
    }
    likelihood = StationaryGaussianGWLikelihood(
        wfg_kwargs={"approximant": "IMRPhenomD", "f_ref": 20.0},
        wfg_domain=domain,
        data_domain=domain,
        event_data=event_data,
        t_ref=1126259462.4,
        calibration_marginalization_kwargs={
            "calibration_envelope": envelopes,
            "num_calibration_nodes": N_NODES,
            "num_calibration_curves": 3,
        },
    )
    transforms = likelihood.projection_transforms.transforms
    sampler = next(t for t in transforms if isinstance(t, SampleCalibrationParameters))
    for ifo in DETECTORS:
        assert sampler.frequency_bounds[ifo] == (domain.f_min, domain.f_max)


def test_widened_analysis_range_reaches_likelihood_and_nodes(envelopes):
    # A network trained on 20-128 Hz of a 20-256 Hz dataset, analyzed at an
    # importance-sampling range up to 512 Hz: the event data (generated on the
    # wider grid) give the likelihood a 20-512 Hz domain, the waveform generator
    # extends to contain it, the ASDs are masked per detector inside it, and the
    # calibration nodes span each detector's range.
    from dingo.gw.inference.context import GWSamplerContext

    metadata = {
        "dataset_settings": {
            "domain": {
                "type": "UniformFrequencyDomain",
                "f_min": 20.0,
                "f_max": 256.0,
                "delta_f": 0.5,
            },
            "waveform_generator": {"approximant": "IMRPhenomD", "f_ref": 20.0},
        },
        "train_settings": {
            "data": {
                "detectors": DETECTORS,
                "ref_time": 1126259462.4,
                "domain_update": {"f_min": 20.0, "f_max": 128.0},
            }
        },
    }
    event_metadata = {
        "domain": {
            "type": "UniformFrequencyDomain",
            "f_min": 20.0,
            "f_max": 512.0,
            "delta_f": 0.5,
        },
        "minimum_frequency": 20.0,
        "maximum_frequency": {"H1": 512.0, "L1": 300.0},
    }
    wide = UniformFrequencyDomain(20.0, 512.0, 0.5)
    mask = wide.frequency_mask
    event_data = {
        "waveform": {d: np.where(mask, (1.0 + 1j) * 1e-21, 0.0) for d in DETECTORS},
        "asds": {d: np.where(mask, 1e-21, 1.0) for d in DETECTORS},
    }
    context = GWSamplerContext.from_model_metadata(metadata, event_data, event_metadata)
    assert context.domain.f_max == 128.0  # the network's
    likelihood = context.likelihood(
        calibration_marginalization_kwargs={
            "calibration_envelope": envelopes,
            "num_calibration_nodes": N_NODES,
            "num_calibration_curves": 3,
        }
    )
    assert likelihood.data_domain == wide
    assert likelihood.waveform_generator.domain.f_max >= 512.0
    f = wide.sample_frequencies
    sampler = next(
        t
        for t in likelihood.projection_transforms.transforms
        if isinstance(t, SampleCalibrationParameters)
    )
    for ifo in DETECTORS:
        f_max = event_metadata["maximum_frequency"][ifo]
        inside = (f >= 20.0) & (f <= f_max)
        assert np.all(likelihood.asd[ifo][inside] == 1e-21)
        assert np.all(likelihood.asd[ifo][~inside] == 1.0)
        assert sampler.frequency_bounds[ifo] == (20.0, f_max)
    # The network-input view cannot use the wider data.
    with pytest.raises(ValueError, match="network's grid"):
        context.prepared_data()
