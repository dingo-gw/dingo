"""RAToEventFrame: the right-ascension frame reparametrization (network training frame
``ra@t_ref`` <-> event frame ``ra``). Covers the forward/inverse round trip -- the inverse
direction the model-based parity harnesses do not exercise -- and the no-op case (event
time equal to, or absent from, the reference time)."""

import types

import numpy as np
import torch
from astropy.time import Time
from astropy.utils import iers

from dingo.gw.inference.steps import RAToEventFrame

# Avoid network access (and the associated timeout) when astropy computes sidereal
# time; the bundled IERS data is sufficient for tests.
iers.conf.auto_download = False


def _context(t_ref, t_event):
    """A minimal stand-in for the SamplerContext fields RAToEventFrame reads."""
    event_metadata = None if t_event is None else {"time_event": t_event}
    return types.SimpleNamespace(t_ref=t_ref, event_metadata=event_metadata)


def test_ra_reparam_round_trip():
    rp = RAToEventFrame()
    ctx = _context(t_ref=1126259462.4, t_event=1264316116.4)  # differ -> real rotation
    ra_tref = torch.rand(1000, dtype=torch.float32) * (2 * np.pi)

    ra = rp.forward({"ra@t_ref": ra_tref}, ctx)["ra"]
    back = rp.inverse({"ra": ra}, ctx)["ra@t_ref"]

    assert ra.dtype == torch.float32  # a bounded angle; float32 is sufficient
    assert not torch.allclose(ra, ra_tref)  # the sidereal rotation actually moved it
    assert torch.allclose(back, ra_tref, atol=1e-6)  # forward then inverse recovers it


def test_ra_reparam_matches_sidereal_shift():
    """The forward rotation must shift `ra` by the apparent-sidereal-time difference.

    The round-trip test above holds for any bijection; this pins the magnitude and
    direction of the rotation to the physical sidereal shift (event minus reference).
    """
    t_ref, t_event = 1126259462.4, 1126259462.4 + 3600.0  # 1 h -> real rotation
    rp = RAToEventFrame()
    ctx = _context(t_ref=t_ref, t_event=t_event)
    ra_tref = torch.rand(1000, dtype=torch.float32) * (2 * np.pi)

    ra_correction = (
        Time(t_event, format="gps", scale="utc").sidereal_time(
            "apparent", "greenwich"
        )
        - Time(t_ref, format="gps", scale="utc").sidereal_time("apparent", "greenwich")
    ).rad

    ra = rp.forward({"ra@t_ref": ra_tref}, ctx)["ra"]
    expected = ((ra_tref.double() + ra_correction) % (2 * np.pi)).float()
    assert torch.equal(ra, expected)
    assert torch.all((ra >= 0) & (ra < 2 * np.pi))


def test_ra_reparam_is_noop_without_a_distinct_event_time():
    rp = RAToEventFrame()
    ra_tref = torch.rand(100, dtype=torch.float32) * (2 * np.pi)
    for ctx in (_context(1126259462.4, 1126259462.4), _context(1126259462.4, None)):
        out = rp.forward({"ra@t_ref": ra_tref}, ctx)["ra"]
        assert torch.equal(out, ra_tref)  # untouched: no cast, no modulo


def test_ra_reparam_declared_names():
    rp = RAToEventFrame()
    assert rp.conditioning == ["ra@t_ref"]
    assert rp.parameters == ["ra"]
