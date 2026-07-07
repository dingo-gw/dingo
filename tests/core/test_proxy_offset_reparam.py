"""Unit tests for ``ProxyOffsetReparam``: the offset reconstruction
`X = delta_X + X_proxy` used by proxy-conditioned networks (e.g. DINGO-BNS chirp
mass), including the reverse fold through a chain."""

import pytest
import torch

from dingo.core.factors import ChainComposer, ProxyOffsetReparam, SampleTableFactor


def test_contract():
    reparam = ProxyOffsetReparam("chirp_mass")
    assert reparam.parameters == ["chirp_mass"]
    assert reparam.conditioning == ["delta_chirp_mass", "chirp_mass_proxy"]
    # The offset is consumed; the proxy stays in the chain (recorded with the
    # samples, like the GNPE time proxies).
    assert reparam.consumes == ["delta_chirp_mass"]


def test_forward_inverse_and_log_det():
    reparam = ProxyOffsetReparam("chirp_mass")
    given = {
        "delta_chirp_mass": torch.tensor([-0.004, 0.0, 0.003], dtype=torch.float64),
        "chirp_mass_proxy": torch.tensor([1.1975, 1.1975, 1.1975], dtype=torch.float64),
    }
    out, log_prob_contribution = reparam.sample_and_log_prob(1, None, given)
    assert torch.allclose(
        out["chirp_mass"], torch.tensor([1.1935, 1.1975, 1.2005], dtype=torch.float64)
    )
    # A pure shift at fixed proxy: measure-preserving.
    assert torch.equal(log_prob_contribution, torch.zeros(3))

    back = reparam.inverse(
        out, None, given={"chirp_mass_proxy": given["chirp_mass_proxy"]}
    )
    assert torch.allclose(back["delta_chirp_mass"], given["delta_chirp_mass"])
    with pytest.raises(ValueError, match="proxy"):
        reparam.inverse(out, None)


def test_chain_replug_rebuilds_offset():
    # In a chain the reverse fold supplies the proxy to the inverse, so log_prob
    # at the emitted samples reproduces the stored chain density.
    table = SampleTableFactor(
        {
            "delta_chirp_mass": torch.tensor([-0.002, 0.001]),
            "chirp_mass_proxy": torch.tensor([1.4868, 1.4868]),
        },
        log_prob=torch.tensor([0.7, -0.3]),
    )
    chain = ChainComposer([table, ProxyOffsetReparam("chirp_mass")])
    samples, log_prob = chain.sample_and_log_prob(2, context=None)
    assert "chirp_mass" in samples and "delta_chirp_mass" not in samples
    assert torch.allclose(log_prob, torch.tensor([0.7, -0.3]))

    # Full-chain re-plug traverses the reparametrization inverse first (the fold
    # supplies the proxy as given) and only then fails at the table root, which
    # is not a density -- the specific error proves the inverse succeeded.
    with pytest.raises(NotImplementedError, match="not a density"):
        chain.log_prob(samples, context=None)
