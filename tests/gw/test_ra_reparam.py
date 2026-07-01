"""RAReparam: the right-ascension frame reparametrization (network training frame
``ra@t_ref`` <-> event frame ``ra``). Covers the forward/inverse round trip -- the inverse
direction the model-based parity harnesses do not exercise -- and the no-op case (event
time equal to, or absent from, the reference time)."""

import types

import numpy as np
import torch

from dingo.gw.inference.factors import RAReparam


def _context(t_ref, t_event):
    """A minimal stand-in for the SamplerContext fields RAReparam reads."""
    event_metadata = None if t_event is None else {"time_event": t_event}
    return types.SimpleNamespace(t_ref=t_ref, event_metadata=event_metadata)


def test_ra_reparam_round_trip():
    rp = RAReparam()
    ctx = _context(t_ref=1126259462.4, t_event=1264316116.4)  # differ -> real rotation
    ra_tref = torch.rand(1000, dtype=torch.float32) * (2 * np.pi)

    ra = rp.forward({"ra@t_ref": ra_tref}, ctx)["ra"]
    back = rp.inverse({"ra": ra}, ctx)["ra@t_ref"]

    assert ra.dtype == torch.float32  # a bounded angle; float32 is sufficient
    assert not torch.allclose(ra, ra_tref)  # the sidereal rotation actually moved it
    assert torch.allclose(back, ra_tref, atol=1e-6)  # forward then inverse recovers it


def test_ra_reparam_is_noop_without_a_distinct_event_time():
    rp = RAReparam()
    ra_tref = torch.rand(100, dtype=torch.float32) * (2 * np.pi)
    for ctx in (_context(1126259462.4, 1126259462.4), _context(1126259462.4, None)):
        out = rp.forward({"ra@t_ref": ra_tref}, ctx)["ra"]
        assert torch.equal(out, ra_tref)  # untouched: no cast, no modulo


def test_ra_reparam_declared_names():
    rp = RAReparam()
    assert rp.conditioning == ["ra@t_ref"]
    assert rp.parameters == ["ra"]
