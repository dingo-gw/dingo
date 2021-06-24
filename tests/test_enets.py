import pytest
from testutils import *
from dingo.core.nn.enets import LinearProjectionRB


def test_enet_reduced_basis_projection():
    # define dimensions
    batch_size, num_bins, n_rb, num_channels = 1000, 200, 10, 3
    # generate datasets
    (y1, y2), (V1, V2) = generate_1d_datasets_and_reduced_basis(batch_size, num_bins, plot=False)
    # define
    projection_layer = LinearProjectionRB(input_dims=(2,num_channels,num_bins), n_rb=10, V_rb=[V1, V2])
    # prepare data for projection_layer
    y_batch_a = get_y_batch([y1, y2], num_channels=num_channels)
    y_batch_b = get_y_batch([np.ones_like(y1), np.zeros_like(y1)], num_channels=num_channels)
    # project onto reduced basis with projection layer
    out_a = np.array(projection_layer(y_batch_a).detach())
    out_b = np.array(projection_layer(y_batch_b).detach())
    # compute reduced basis projections with matrix multiplications as comparison
    ref_a = project_onto_reduced_basis_and_concat([y1, y2], [V1, V2], n_rb)
    ref_b = project_onto_reduced_basis_and_concat([np.ones_like(y1), np.zeros_like(y1)], [V1, V2], n_rb)
    # check if results agree
    thr = 1e-5
    assert np.max(np.abs(out_a - ref_a)) < thr
    assert np.max(np.abs(out_b - ref_b)) < thr
    assert np.max(np.abs(out_a - ref_b)) > thr

if __name__ == '__main__':
    pass
