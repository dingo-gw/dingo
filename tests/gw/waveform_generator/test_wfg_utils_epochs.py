"""Tests for the batched (vectorized over phase) epoch computation used by the
TEOB deferred time shift. The batched function must reproduce the scalar
compute_epoch_from_resized_td_modes exactly, for every phase."""

import numpy as np
import pytest

from dingo.gw.waveform_generator import wfg_utils


@pytest.fixture
def synthetic_resized_td_modes():
    """Non-precessing-like set of TD modes with a peak near the middle of the array,
    plus low-level broadband content everywhere so that the peak search has to
    discriminate against many candidate samples."""
    rng = np.random.default_rng(1234)
    n = 4096
    delta_t = 1.0 / 1024
    t = np.arange(n) * delta_t
    t_peak = t[n // 2]
    envelope = np.exp(-(((t - t_peak) / 0.015) ** 2))
    background = 0.02 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))

    modes = {}
    for (l, m), amp in [((2, 2), 1.0), ((2, 1), 0.2), ((3, 3), 0.3), ((4, 4), 0.15)]:
        frequency = 120.0 * m / 2  # crude harmonic scaling of the (2,2) frequency
        h = amp * envelope * np.exp(-1j * (2 * np.pi * frequency * t + rng.uniform(0, 2 * np.pi)))
        h = h + amp * background
        modes[(l, m)] = h
        modes[(l, -m)] = (-1) ** l * np.conj(h)
    return modes, delta_t


def test_batched_epochs_match_scalar_for_every_phase(synthetic_resized_td_modes):
    modes, delta_t = synthetic_resized_td_modes
    iota = 0.7
    phases = np.linspace(0, 2 * np.pi, 733)

    expected = np.array(
        [
            wfg_utils.compute_epoch_from_resized_td_modes(modes, iota, p, delta_t)
            for p in phases
        ]
    )
    # The test is only meaningful if the epoch actually varies with phase.
    assert len(np.unique(expected)) > 1

    got = wfg_utils.compute_epochs_from_resized_td_modes(modes, iota, phases, delta_t)

    assert got.shape == phases.shape
    np.testing.assert_array_equal(got, expected)


def test_batched_epochs_accept_scalar_phase(synthetic_resized_td_modes):
    modes, delta_t = synthetic_resized_td_modes
    iota = 1.9
    phase = 2.3

    expected = wfg_utils.compute_epoch_from_resized_td_modes(modes, iota, phase, delta_t)
    got = wfg_utils.compute_epochs_from_resized_td_modes(modes, iota, phase, delta_t)

    assert got.shape == (1,)
    assert got[0] == expected
