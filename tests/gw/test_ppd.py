"""Tests for the strain-PPD helpers in :mod:`dingo.gw.utils.plotting`."""

from types import SimpleNamespace

import numpy as np

from dingo.gw.utils.plotting import one_sided_fd_to_td


def test_one_sided_fd_to_td_single_tone() -> None:
    """A one-sided spectrum with a single populated bin k inverse-FFTs to a real cosine
    of frequency k * delta_f, with the documented 2 * n - 1 output length."""
    n = 65
    delta_f = 1.0
    f_max = (n - 1) * delta_f  # one_sided_fd_to_td only reads domain.f_max
    domain = SimpleNamespace(f_max=f_max)

    k = 8
    fd = np.zeros(n, dtype=np.complex128)
    fd[k] = 1.0

    times, td = one_sided_fd_to_td(fd, domain)

    # Hermitian spectrum -> real time series, and the documented length.
    assert len(td) == 2 * n - 1
    assert np.max(np.abs(td.imag)) < 1e-9

    # Dominant frequency of the recovered tone is k * delta_f.
    dt = times[1] - times[0]
    rfreqs = np.fft.rfftfreq(len(td), d=dt)
    recovered = rfreqs[np.argmax(np.abs(np.fft.rfft(td.real)))]
    assert abs(recovered - k * delta_f) < 1.0  # within one frequency bin
