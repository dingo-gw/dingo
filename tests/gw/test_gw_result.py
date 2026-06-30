import numpy as np
import pandas as pd
import pytest

from dingo.gw.domains import UniformFrequencyDomain
from dingo.gw.gwutils import get_extrinsic_prior_dict
from dingo.gw.likelihood import StationaryGaussianGWLikelihood
from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.result import Result


DETECTORS = ["H1", "L1"]
REF_TIME = 1126259462.391

DOMAIN_SETTINGS = {
    "type": "UniformFrequencyDomain",
    "f_min": 20.0,
    "f_max": 256.0,  # small (T=2s) so the likelihood is cheap
    "delta_f": 0.5,
}
WAVEFORM_GENERATOR = {"approximant": "IMRPhenomD", "f_ref": 20.0}

INTRINSIC_PRIOR = {
    "mass_1": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0, name='mass_1')",
    "mass_2": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0, name='mass_2')",
    "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass("
    "minimum=15.0, maximum=100.0, name='chirp_mass')",
    "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio("
    "minimum=0.125, maximum=1.0, name='mass_ratio')",
    "phase": "default",
    "chi_1": "bilby.gw.prior.AlignedSpin(name='chi_1', a_prior=Uniform(minimum=0, maximum=0.9))",
    "chi_2": "bilby.gw.prior.AlignedSpin(name='chi_2', a_prior=Uniform(minimum=0, maximum=0.9))",
    "theta_jn": "default",
    "luminosity_distance": 100.0,
    "geocent_time": 0.0,
}
EXTRINSIC_PRIOR = {
    "dec": "default",
    "ra": "default",
    "geocent_time": "bilby.core.prior.Uniform("
    "minimum=-0.10, maximum=0.10, name='geocent_time')",
    "psi": "default",
    "luminosity_distance": "bilby.core.prior.Uniform("
    "minimum=100.0, maximum=1000.0, name='luminosity_distance')",
}


def _metadata():
    return {
        "dataset_settings": {
            "domain": DOMAIN_SETTINGS,
            "waveform_generator": WAVEFORM_GENERATOR,
            "intrinsic_prior": INTRINSIC_PRIOR,
        },
        "train_settings": {
            "data": {
                "detectors": DETECTORS,
                "ref_time": REF_TIME,
                "extrinsic_prior": EXTRINSIC_PRIOR,
            }
        },
    }


def _context():
    """Synthetic strain + ASD context (constant above f_min, masked below)."""
    domain = UniformFrequencyDomain(
        DOMAIN_SETTINGS["f_min"], DOMAIN_SETTINGS["f_max"], DOMAIN_SETTINGS["delta_f"]
    )
    mask = domain.frequency_mask
    waveform = {d: np.where(mask, (1.0 + 1j) * 1e-21, 0.0) for d in DETECTORS}
    asds = {d: np.where(mask, 1e-21, 1.0) for d in DETECTORS}
    return {"waveform": waveform, "asds": asds}


def make_gw_result(n=5, drop_phase=False, event_metadata=None):
    """Build a gw Result with `n` rows drawn from its own prior (so columns/ranges are
    consistent with the waveform generator), plus a synthetic context."""
    full_prior = build_prior_with_defaults(
        {**INTRINSIC_PRIOR, **get_extrinsic_prior_dict(EXTRINSIC_PRIOR)}
    )
    samples = pd.DataFrame(full_prior.sample(n))
    samples["log_prob"] = 0.0
    if drop_phase:
        samples = samples.drop(columns="phase")
    return Result(
        dictionary={
            "samples": samples,
            "context": _context(),
            "event_metadata": {} if event_metadata is None else event_metadata,
            "settings": _metadata(),
        }
    )


# ---------------------------------------------------------------------------
# Simple property accessors
# ---------------------------------------------------------------------------


@pytest.fixture()
def gw_result():
    return make_gw_result()


@pytest.mark.parametrize(
    "attr, key",
    [
        ("synthetic_phase_kwargs", "synthetic_phase"),
        ("time_marginalization_kwargs", "time_marginalization"),
        ("phase_marginalization_kwargs", "phase_marginalization"),
        ("calibration_marginalization_kwargs", "calibration_marginalization"),
        ("calibration_sampling_kwargs", "calibration_sampling"),
    ],
)
def test_kwargs_round_trip_via_importance_sampling_metadata(gw_result, attr, key):
    assert getattr(gw_result, attr) is None  # not set yet
    value = {"some": "setting"}
    setattr(gw_result, attr, value)
    assert getattr(gw_result, attr) == value
    assert gw_result.importance_sampling_metadata[key] == value


def test_use_base_domain_default_and_noop_on_uniform_domain(gw_result):
    # UniformFrequencyDomain has no base_domain, so the setter is a no-op.
    assert gw_result.use_base_domain is False
    gw_result.use_base_domain = True
    assert gw_result.use_base_domain is False


def test_f_ref_and_approximant(gw_result):
    assert gw_result.f_ref == WAVEFORM_GENERATOR["f_ref"]
    assert gw_result.approximant == WAVEFORM_GENERATOR["approximant"]


def test_t_ref_defaults_to_ref_time(gw_result):
    assert gw_result.t_ref == REF_TIME


def test_t_ref_uses_event_time_when_present():
    result = make_gw_result(event_metadata={"time_event": REF_TIME + 100.0})
    assert result.t_ref == REF_TIME + 100.0


def test_minimum_maximum_frequency_default_to_domain(gw_result):
    assert gw_result.minimum_frequency == gw_result.domain.f_min
    assert gw_result.maximum_frequency == gw_result.domain.f_max


def test_minimum_maximum_frequency_override_and_setter(gw_result):
    result = make_gw_result(
        event_metadata={"minimum_frequency": 30.0, "maximum_frequency": 200.0}
    )
    assert result.minimum_frequency == 30.0
    assert result.maximum_frequency == 200.0

    gw_result.minimum_frequency = 25.0
    assert gw_result.event_metadata["minimum_frequency"] == 25.0
    assert gw_result.minimum_frequency == 25.0


def test_interferometers(gw_result):
    assert gw_result.interferometers == DETECTORS


def test_build_domain_creates_uniform_frequency_domain(gw_result):
    assert isinstance(gw_result.domain, UniformFrequencyDomain)
    assert gw_result.domain.f_min == DOMAIN_SETTINGS["f_min"]
    assert gw_result.domain.f_max == DOMAIN_SETTINGS["f_max"]


# ---------------------------------------------------------------------------
# Likelihood tests
# ---------------------------------------------------------------------------


def test_build_likelihood(gw_result):
    gw_result._build_likelihood()
    assert isinstance(gw_result.likelihood, StationaryGaussianGWLikelihood)
    assert np.isfinite(gw_result.likelihood.log_Zn)


def test_importance_sample_populates_columns_and_evidence(gw_result):
    gw_result.importance_sample(num_processes=1)
    for col in ("log_prior", "log_likelihood", "weights"):
        assert col in gw_result.samples.columns
    assert np.isfinite(gw_result.log_evidence)
    assert np.isfinite(gw_result.log_noise_evidence)
    # Normalized weights have mean 1.
    assert gw_result.samples["weights"].mean() == pytest.approx(1.0)


def test_importance_sample_requires_log_prob():
    result = make_gw_result()
    result.samples = result.samples.drop(columns="log_prob")
    with pytest.raises(KeyError, match="log probability"):
        result.importance_sample()


def test_sample_synthetic_phase_adds_phase_column():
    result = make_gw_result(drop_phase=True)
    assert "phase" not in result.samples.columns
    log_prob_before = result.samples["log_prob"].to_numpy().copy()

    result.sample_synthetic_phase({"n_grid": 16, "approximation_22_mode": True})

    assert "phase" in result.samples.columns
    phase = result.samples["phase"].to_numpy()
    assert np.all((phase >= 0) & (phase < 2 * np.pi))
    # log_prob is updated with the synthetic-phase conditional.
    assert not np.array_equal(result.samples["log_prob"].to_numpy(), log_prob_before)


def test_sample_synthetic_phase_requires_uniform_phase_prior():
    # When `phase` is in the samples, the phase prior is not split off (it is None),
    # so synthetic phase sampling is not applicable and must raise.
    result = make_gw_result(drop_phase=False)
    with pytest.raises(ValueError, match="[Pp]hase prior"):
        result.sample_synthetic_phase({"n_grid": 16})


# ---------------------------------------------------------------------------
# sample_calibration_parameters
# ---------------------------------------------------------------------------


def _write_envelope(path):
    """Write a minimal LVC-format calibration envelope file.

    Columns: frequency, median-amp, median-phase, -1sigma-amp, -1sigma-phase,
    +1sigma-amp, +1sigma-phase. At least 4 rows are needed (cubic spline), spanning
    the domain's [f_min, f_max].
    """
    freqs = np.geomspace(15.0, 300.0, 8)
    data = np.column_stack(
        [
            freqs,
            np.ones_like(freqs),  # median amplitude ~ 1
            np.zeros_like(freqs),  # median phase ~ 0
            np.full_like(freqs, 0.99),  # -1 sigma amplitude
            np.full_like(freqs, -0.01),  # -1 sigma phase
            np.full_like(freqs, 1.01),  # +1 sigma amplitude
            np.full_like(freqs, 0.01),  # +1 sigma phase
        ]
    )
    np.savetxt(path, data)


def _calibration_kwargs(tmp_path, correction_type="data", num_nodes=5):
    envelopes = {}
    for ifo in DETECTORS:
        path = tmp_path / f"{ifo}.txt"
        _write_envelope(path)
        envelopes[ifo] = str(path)
    return {
        "calibration_envelope": envelopes,
        "num_calibration_nodes": num_nodes,
        "correction_type": correction_type,
    }


def test_sample_calibration_parameters_adds_recalib_columns(tmp_path):
    result = make_gw_result()
    log_prob_before = result.samples["log_prob"].to_numpy().copy()
    n_nodes = 5

    result.sample_calibration_parameters(
        _calibration_kwargs(tmp_path, num_nodes=n_nodes)
    )

    # Amplitude + phase nodes per detector (the frequency nodes are delta functions
    # and are dropped before sampling).
    recalib_cols = [c for c in result.samples.columns if c.startswith("recalib_")]
    assert len(recalib_cols) == 2 * n_nodes * len(DETECTORS)
    # The calibration prior log_prob is folded into the proposal log_prob.
    assert not np.array_equal(
        result.samples["log_prob"].to_numpy(), log_prob_before
    )
    # The calibration priors are recorded for persistence and added to the prior.
    assert len(result.importance_sampling_metadata["prior_update"]) == len(recalib_cols)
    assert any("recalib" in key for key in result.prior.keys())


def test_sample_calibration_parameters_invalid_correction_type():
    result = make_gw_result()
    # Parsed before any envelope file is read, so no files are needed.
    with pytest.raises(ValueError, match="not understood"):
        result.sample_calibration_parameters({"correction_type": "bogus"})


@pytest.mark.parametrize(
    "correction_type",
    ["data", "template", {"H1": "data", "L1": "template"}, None],
)
def test_sample_calibration_parameters_correction_type_variants(
    tmp_path, correction_type
):
    result = make_gw_result()
    result.sample_calibration_parameters(
        _calibration_kwargs(tmp_path, correction_type=correction_type)
    )
    assert any(c.startswith("recalib_") for c in result.samples.columns)
