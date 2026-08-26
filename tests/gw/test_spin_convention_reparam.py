"""Unit tests for ``SpinConventionReparam``: the step contract (in-place bijection,
guarded log_det / inverse), the metadata-driven convention including the
``None``-convention identity, and a real LAL roundtrip between the network and
physical conventions."""

import numpy as np
import pandas as pd
import pytest
import torch

from dingo.gw.inference.steps import SpinConventionReparam

_MODEL_METADATA_SC0 = {
    "dataset_settings": {
        "waveform_generator": {
            "approximant": "IMRPhenomXPHM",
            "f_ref": 20.0,
            "spin_conversion_phase": 0.0,
        }
    }
}

_MODEL_METADATA_NONE = {
    "dataset_settings": {
        "waveform_generator": {"approximant": "IMRPhenomXPHM", "f_ref": 20.0}
    }
}


def _samples(n=5, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "theta_jn": rng.uniform(0.1, np.pi - 0.1, n),
            "phi_jl": rng.uniform(0, 2 * np.pi, n),
            "phase": rng.uniform(0, 2 * np.pi, n),
            "chirp_mass": rng.uniform(20.0, 40.0, n),
            "mass_ratio": rng.uniform(0.5, 1.0, n),
            "a_1": rng.uniform(0.1, 0.9, n),
            "a_2": rng.uniform(0.1, 0.9, n),
            "tilt_1": rng.uniform(0.1, np.pi - 0.1, n),
            "tilt_2": rng.uniform(0.1, np.pi - 0.1, n),
            "phi_12": rng.uniform(0, 2 * np.pi, n),
        }
    )


def test_contract():
    reparam = SpinConventionReparam()
    assert reparam.parameters == ["theta_jn", "phi_jl"]
    # The bijection overwrites its parameters in place; the read-only
    # conditioning (phase, masses, tilts, ...) must not be consumed.
    assert reparam.consumes == []
    assert set(reparam.parameters) < set(reparam.conditioning)


def test_inverse_requires_given_and_closes_roundtrip():
    # The physical -> network direction needs the invariant conditioning, which
    # the reverse fold supplies as `given`; without it the call fails loudly.
    class _Context:
        model_metadata = _MODEL_METADATA_SC0

    reparam = SpinConventionReparam()
    with pytest.raises(ValueError, match="conditioning"):
        reparam.inverse({}, None)

    samples = _samples()
    given = {k: torch.as_tensor(samples[k].to_numpy()) for k in reparam.conditioning}
    physical, _ = reparam.sample_and_log_prob(1, _Context(), given)
    inverse_given = {**given, **physical}
    back = reparam.inverse(physical, _Context(), inverse_given)
    for k in reparam.parameters:
        assert np.allclose(back[k].numpy(), samples[k].to_numpy(), atol=1e-10)


def test_none_convention_is_identity():
    # A model trained without a fixed spin-conversion phase already outputs the
    # physical convention; the relabel is the identity.
    samples = _samples()
    out = SpinConventionReparam().to_physical(samples, _MODEL_METADATA_NONE)
    pd.testing.assert_frame_equal(out, samples.astype(float))


def test_roundtrip_network_physical_network():
    reparam = SpinConventionReparam()
    samples = _samples()
    physical = reparam.to_physical(samples, _MODEL_METADATA_SC0)
    back = reparam.to_network(physical, _MODEL_METADATA_SC0)
    # Only the two convention-dependent angles change (the others are invariants
    # of the common rotation, recomputed through LAL trig, so equal to roundoff);
    # the roundtrip closes; phase is explicitly restored, so it is exact.
    assert not np.allclose(physical["theta_jn"], samples["theta_jn"])
    for col in samples.columns:
        assert np.allclose(back[col], samples[col], atol=1e-10), col
    for col in ["tilt_1", "tilt_2", "phi_12", "a_1", "a_2"]:
        assert np.allclose(physical[col], samples[col], atol=1e-12), col
    assert np.array_equal(physical["phase"], samples["phase"].astype(float))


def test_jacobian_matches_sin_ratio():
    # The convention change rotates the line of sight rigidly about L, so it
    # preserves the spherical measure sin(theta_jn) dtheta dphi; in the flat
    # (theta_jn, phi_jl) coordinates the Jacobian determinant is then
    # sin(theta_jn) / sin(theta_jn'). Verify numerically by central finite
    # differences through the actual LAL conversion, and check that log_det /
    # sample_and_log_prob implement exactly this.
    reparam = SpinConventionReparam()
    base = _samples(n=20, seed=3)
    # Keep clear of the coordinate poles, where finite differences degrade.
    base["theta_jn"] = np.clip(base["theta_jn"], 0.3, np.pi - 0.3)
    h = 1e-6

    # One batched conversion: for each base row, rows [base, th+h, th-h, ph+h, ph-h].
    blocks = []
    for _, row in base.iterrows():
        for col, delta in [
            (None, 0.0),
            ("theta_jn", h),
            ("theta_jn", -h),
            ("phi_jl", h),
            ("phi_jl", -h),
        ]:
            r = row.copy()
            if col is not None:
                r[col] += delta
            blocks.append(r)
    stacked = pd.DataFrame(blocks).reset_index(drop=True)
    converted = reparam.to_physical(stacked, _MODEL_METADATA_SC0)

    def wrap(d):
        return (d + np.pi) % (2 * np.pi) - np.pi

    theta_out = converted["theta_jn"].to_numpy().reshape(-1, 5)
    phi_out = converted["phi_jl"].to_numpy().reshape(-1, 5)
    dtheta_dtheta = (theta_out[:, 1] - theta_out[:, 2]) / (2 * h)
    dphi_dtheta = wrap(phi_out[:, 1] - phi_out[:, 2]) / (2 * h)
    dtheta_dphi = (theta_out[:, 3] - theta_out[:, 4]) / (2 * h)
    dphi_dphi = wrap(phi_out[:, 3] - phi_out[:, 4]) / (2 * h)
    det = dtheta_dtheta * dphi_dphi - dtheta_dphi * dphi_dtheta

    expected = np.sin(base["theta_jn"].to_numpy()) / np.sin(theta_out[:, 0])
    assert np.max(np.abs(det / expected - 1)) < 1e-5
    # The map genuinely distorts the flat coordinate measure (order-one effect):
    # a zero log_det here would corrupt chain densities per-row.
    assert np.max(np.abs(expected - 1)) > 0.01

    # The class implements exactly this Jacobian, and the sampling path
    # contributes its negative alongside the forward transform.
    class _Context:
        model_metadata = _MODEL_METADATA_SC0

    given = {k: torch.as_tensor(base[k].to_numpy()) for k in reparam.conditioning}
    log_det = reparam.log_det(given, _Context())
    assert np.allclose(log_det.numpy(), np.log(expected), atol=1e-12)
    out, log_prob_contribution = reparam.sample_and_log_prob(1, _Context(), given)
    assert np.allclose(out["theta_jn"].numpy(), theta_out[:, 0])
    assert np.allclose(log_prob_contribution.numpy(), -np.log(expected), atol=1e-12)


def test_chain_dtype_preserved():
    # A reparametrization must not promote the chain's dtype: float32 columns in,
    # float32 transform and log_det contribution out (computation is double
    # internally).
    class _Context:
        model_metadata = _MODEL_METADATA_SC0

    reparam = SpinConventionReparam()
    samples = _samples()
    given = {
        k: torch.as_tensor(samples[k].to_numpy(), dtype=torch.float32)
        for k in reparam.conditioning
    }
    out, log_prob_contribution = reparam.sample_and_log_prob(1, _Context(), given)
    assert out["theta_jn"].dtype == torch.float32
    assert out["phi_jl"].dtype == torch.float32
    assert log_prob_contribution.dtype == torch.float32


def test_forward_matches_to_physical():
    class _Context:
        model_metadata = _MODEL_METADATA_SC0

    reparam = SpinConventionReparam()
    samples = _samples()
    given = {k: torch.as_tensor(samples[k].to_numpy()) for k in reparam.conditioning}
    out = reparam.forward(given, _Context())
    expected = reparam.to_physical(samples, _MODEL_METADATA_SC0)
    assert set(out) == {"theta_jn", "phi_jl"}
    for k in out:
        assert np.allclose(out[k].numpy(), expected[k].to_numpy())
