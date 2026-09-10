"""SVDBasis file round trip: the constructor's file path passes through the
DingoDataset loader, so the from_file override must accept its keywords."""

import numpy as np

from dingo.gw.SVD import SVDBasis


def test_svd_basis_round_trips_through_file(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((20, 8)) + 1j * rng.standard_normal((20, 8))
    basis = SVDBasis()
    basis.generate_basis(data, n=4)
    basis.to_file(str(tmp_path / "svd.hdf5"))

    loaded = SVDBasis(file_name=str(tmp_path / "svd.hdf5"))
    np.testing.assert_allclose(loaded.V, basis.V)
    np.testing.assert_allclose(loaded.s, basis.s)
    assert loaded.n == basis.n
