"""Resolution of dingo_pipe's importance-sampling settings: the recovery defaults
chosen from the sample columns."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from dingo.gw.domains import UniformFrequencyDomain
from dingo.pipe.default_settings import IMPORTANCE_SAMPLING_SETTINGS
from dingo.pipe.importance_sampling import ImportanceSamplingInput


def _resolve(columns, settings):
    """Run the settings setter on a minimal stand-in for ImportanceSamplingInput."""
    stub = SimpleNamespace(
        result=SimpleNamespace(
            samples=pd.DataFrame(columns=columns),
            domain=UniformFrequencyDomain(20.0, 512.0, 0.25),
        ),
        calibration_mode=None,
    )
    ImportanceSamplingInput.importance_sampling_settings.fset(stub, settings)
    return stub._importance_sampling_settings


@pytest.mark.parametrize(
    "columns, default",
    [
        (["chirp_mass", "psi"], "PhaseRecoveryDefault"),
        (["chirp_mass"], "PhasePsiRecoveryDefault"),
    ],
)
def test_recovery_default_follows_sample_columns(columns, default):
    resolved = _resolve(columns, "Default")
    assert resolved == IMPORTANCE_SAMPLING_SETTINGS[default]
    # The module-level defaults are copied, not updated in place.
    resolved["use_base_domain"] = True
    assert "use_base_domain" not in IMPORTANCE_SAMPLING_SETTINGS[default]


def test_no_recovery_default_when_phase_is_sampled():
    assert _resolve(["chirp_mass", "phase", "psi"], "Default") == {}


def test_synthetic_phase_key_is_rejected():
    # The former name raises instead of leaving the default settings silently in use.
    with pytest.raises(ValueError, match="renamed to synthetic_parameters"):
        _resolve(["chirp_mass"], "{synthetic_phase: {n_grid_phase: 64}}")


@pytest.mark.parametrize("psi_prior, cached", [(object(), True), (None, False)])
def test_cached_log_likelihood_without_approximation_22_mode_key(psi_prior, cached):
    # SyntheticPhasePsiFactor is always exact, so an omitted approximation_22_mode
    # only disables caching for the phase-only SyntheticPhaseFactor.
    result = MagicMock(psi_prior=psi_prior)
    stub = SimpleNamespace(
        result=result,
        importance_sampling_settings={"synthetic_parameters": {"n_grid_phase": 64}},
        prior_dict_updates=None,
        calibration_marginalization_kwargs=None,
        request_cpus=1,
        result_directory="outdir",
        label="label",
        _synthetic_parameters_modes_exact=lambda kwargs: True,
    )
    ImportanceSamplingInput.run_sampler(stub)
    assert (
        result.importance_sample.call_args.kwargs["use_cached_log_likelihood"] is cached
    )
