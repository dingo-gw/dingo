"""SVDBasis method='gram' (eigenvectors of A^H A) against a direct SVD, on complex
matrices with a known spectrum. Bases are compared as subspaces (projectors), since
vectors inside a near-degenerate cluster are not unique."""

import numpy as np
import pytest
import scipy.linalg

from dingo.gw.SVD import SVDBasis


def _matrix(rng, rows, cols, s):
    """Complex rows x cols matrix with singular values s and random singular vectors."""
    k = len(s)
    U, _ = np.linalg.qr(
        rng.standard_normal((rows, k)) + 1j * rng.standard_normal((rows, k))
    )
    V, _ = np.linalg.qr(
        rng.standard_normal((cols, k)) + 1j * rng.standard_normal((cols, k))
    )
    return (U * s) @ V.conj().T


def test_gram_matches_direct_svd_with_weak_and_clustered_modes():
    rng = np.random.default_rng(0)
    s = np.logspace(0, -4, 40)  # condition number 1e4, 1e8 in the Gram matrix
    s[5:8] = 0.3 * (1 + np.array([0, 1e-10, 2e-10]))  # near-degenerate retained cluster
    A = _matrix(rng, 200, 60, s)
    n = 20

    gram = SVDBasis()
    gram.generate_basis(A, n, method="gram")
    _, s_ref, Vh_ref = scipy.linalg.svd(A, full_matrices=False)
    V_ref = Vh_ref[:n].conj().T

    assert gram.V.shape == (60, n) and gram.n == n
    np.testing.assert_allclose(gram.s, s_ref[:n], rtol=1e-8)
    np.testing.assert_allclose(gram.V.conj().T @ gram.V, np.eye(n), atol=1e-12)
    projector_diff = gram.V @ gram.V.conj().T - V_ref @ V_ref.conj().T
    assert np.linalg.norm(projector_diff, 2) < 1e-8
    # Same reconstruction as the default method.
    default = SVDBasis()
    default.generate_basis(A, n)
    np.testing.assert_allclose(
        A @ gram.V @ gram.Vh, A @ default.V @ default.Vh, atol=1e-10
    )


def test_gram_refuses_modes_below_rounding_error():
    rng = np.random.default_rng(1)
    A = _matrix(rng, 200, 60, np.logspace(0, -9, 30))  # s**2 reaches 1e-18 < eps
    with pytest.raises(ValueError, match="not resolved"):
        SVDBasis().generate_basis(A, 30, method="gram")
    # The well-resolved leading modes of the same matrix are fine.
    basis = SVDBasis()
    basis.generate_basis(A, 10, method="gram")
    np.testing.assert_allclose(basis.s, scipy.linalg.svdvals(A)[:10], rtol=1e-8)


def test_gram_promotes_single_precision_input():
    rng = np.random.default_rng(2)
    A = _matrix(rng, 100, 30, np.logspace(0, -2, 20)).astype(np.complex64)
    basis = SVDBasis()
    basis.generate_basis(A, 5, method="gram")
    assert basis.V.dtype == np.complex128


def test_gram_basis_initialises_projection_layer():
    """The basis goes into torch.from_numpy in LinearProjectionRB.init_layers, which
    rejects arrays with negative strides."""
    import torch

    rng = np.random.default_rng(3)
    basis = SVDBasis()
    A = _matrix(rng, 100, 30, np.logspace(0, -2, 20))
    basis.generate_basis(A, 5, method="gram")
    assert basis.V.flags.c_contiguous
    torch.from_numpy(basis.V.real)


def test_test_mismatches_match_row_by_row_definition():
    """Blocked compute_test_mismatches against the per-row definition, across blocks."""
    import pandas as pd

    rng = np.random.default_rng(4)
    A = _matrix(rng, 1200, 40, np.logspace(0, -3, 30))
    basis = SVDBasis()
    basis.generate_basis(A[:600], 12)
    val = A[600:] + 1e-3 * rng.standard_normal(A[600:].shape)
    basis.compute_test_mismatches(
        val, parameters=pd.DataFrame(index=range(600)), increment=5
    )
    assert list(basis.mismatches.columns) == [f"mismatch n={n}" for n in (5, 10, 12)]
    for n in (5, 10, 12):
        expected = []
        for d in val:
            rec = d @ basis.V[:, :n] @ basis.Vh[:n]
            inner = np.sum(d.conj() * rec).real
            expected.append(1 - inner / (np.linalg.norm(d) * np.linalg.norm(rec)))
        np.testing.assert_allclose(
            basis.mismatches[f"mismatch n={n}"], expected, rtol=1e-10, atol=1e-15
        )
