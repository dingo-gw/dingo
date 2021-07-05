import pytest
from testutils_enets import *
from dingo.core.nn.enets import *


def test_projection_of_LinearProjectionRB():
    """
    This function tests the LinearProjectionRB embedding network for the
    dimensions and the correct application of the reduced basis projection.
    """
    # define dimensions
    batch_size, num_bins, n_rb, num_channels = 1000, 200, 10, 3
    # generate datasets
    (y1, y2), (V1, V2) = \
        generate_1d_datasets_and_reduced_basis(batch_size, num_bins)
    # define projection layer
    projection_layer = LinearProjectionRB(
        input_dims=(2, num_channels, num_bins), n_rb=10, V_rb_list=(V1, V2))
    # prepare data for projection_layer
    y_batch_a = get_y_batch([y1, y2], num_channels=num_channels)
    y_batch_b = get_y_batch([np.ones_like(y1), np.zeros_like(y1)],
                            num_channels=num_channels)
    # project onto reduced basis with projection layer
    out_a = np.array(projection_layer(y_batch_a).detach())
    out_b = np.array(projection_layer(y_batch_b).detach())
    # compute reduced basis projections with matrix multiplications as
    # comparison
    ref_a = project_onto_reduced_basis_and_concat(
        [y1, y2], [V1, V2], n_rb)
    ref_b = project_onto_reduced_basis_and_concat(
        [np.ones_like(y1), np.zeros_like(y1)], [V1, V2], n_rb)
    # check if results agree
    thr = 1e-5
    assert np.max(np.abs(out_a - ref_a)) < thr
    assert np.max(np.abs(out_b - ref_b)) < thr
    # check that results for different inputs disagree
    assert np.max(np.abs(out_a - ref_b)) > thr
    # check that channels with index >= 2 don't affect the projection
    y_batch_a[:, :, 2:, :] += torch.rand_like(y_batch_a[:, :, 2:, :])
    assert np.all(out_a == np.array(projection_layer(y_batch_a).detach())), \
        'Channels with index >= 2 should not affect rb projection.'

    # check that channels with index >= 2 do affect the projection when layer
    # is not initialized with the reduced basis
    projection_layer = LinearProjectionRB(
        input_dims=(2, num_channels, num_bins), n_rb=10, V_rb_list=None)
    out_a_1 = np.array(projection_layer(y_batch_a).detach())
    y_batch_a[:, :, 2:, :] -= torch.rand_like(y_batch_a[:, :, 2:, :])
    out_a_2 = np.array(projection_layer(y_batch_a).detach())
    assert np.all(out_a_1 != out_a_2), \
        'Channels with index >= 2 should affect projection layer when not ' \
        'initialized with reduced basis.'

    # check that Error is raised if layer is initialized with inconsistent input
    with pytest.raises(ValueError):
        LinearProjectionRB(input_dims=(2, num_channels, num_bins), n_rb=10,
                           V_rb_list=(V1, np.zeros_like(y1)))
    with pytest.raises(ValueError):
        LinearProjectionRB(input_dims=(2, num_channels, num_bins), n_rb=10,
                           V_rb_list=(V1, V2, V2))
    with pytest.raises(ValueError):
        LinearProjectionRB(input_dims=(2, num_channels, num_bins + 1), n_rb=10,
                           V_rb_list=(V1, V2))
    with pytest.raises(ValueError):
        LinearProjectionRB(input_dims=(2, 1, num_bins), n_rb=10,
                           V_rb_list=(V1, V2))
    with pytest.raises(ValueError):
        LinearProjectionRB(input_dims=(2, num_channels, num_bins, 1), n_rb=10,
                           V_rb_list=(V1, V2))


def test_forward_pass_of_LinearProjectionRB():
    """
    Test forward pass of the LinearProjectionRB embedding network.
    """
    batch_size, n_rb, num_blocks, num_channels, num_bins = 1000, 10, 2, 3, 200
    _, (V1, V2) = generate_1d_datasets_and_reduced_basis(batch_size, num_bins)
    enet = LinearProjectionRB(input_dims=(num_blocks, num_channels, num_bins),
                              n_rb=10, V_rb_list=(V1, V2))
    check_model_forward_pass(enet, (num_blocks, num_channels, num_bins),
                             [enet.output_dim], batch_size)


def test_backward_pass_of_LinearProjectionRB():
    """
    Test backward pass of the LinearProjectionRB embedding network.
    """
    batch_size, n_rb, num_blocks, num_channels, num_bins = 1000, 10, 2, 3, 200
    _, (V1, V2) = generate_1d_datasets_and_reduced_basis(batch_size, num_bins)
    enet = LinearProjectionRB(input_dims=(num_blocks, num_channels, num_bins),
                              n_rb=10, V_rb_list=(V1, V2))
    check_model_backward_pass(enet, (num_blocks, num_channels, num_bins),
                              batch_size)


def test_forward_pass_of_DenseResidualNet():
    """
    Test forward pass of the DenseResidualNet embedding network.
    """
    batch_size = 100
    input_dim, output_dim, hidden_dims = 120, 8, (128, 64, 32, 64, 16, 16)
    enet = DenseResidualNet(input_dim, output_dim, hidden_dims)
    check_model_forward_pass(enet, [input_dim], [output_dim], batch_size)


def test_backward_pass_of_DenseResidualNet():
    """
    Test backward pass of the DenseResidualNet embedding network.
    """
    batch_size = 100
    input_dim, output_dim, hidden_dims = 120, 8, (128, 64, 32, 64, 16, 16)
    enet = DenseResidualNet(input_dim, output_dim, hidden_dims)
    check_model_backward_pass(enet, [input_dim], batch_size)


def test_forward_pass_of_EnetProjectionWithResnet():
    """
    Test forward pass of the EnetProjectionWithResnet embedding network.
    """
    batch_size, n_rb, num_blocks, num_channels, num_bins = 1000, 10, 2, 3, 200
    _, (V1, V2) = generate_1d_datasets_and_reduced_basis(batch_size, num_bins)

    enet_kwargs = {
        'input_dims': (num_blocks, num_channels, num_bins),
        'n_rb': n_rb,
        'V_rb_list': (V1, V2),
        'output_dim': 8,
        'hidden_dims': [32, 16, 16, 8],
        'activation': torch.nn.functional.elu,
        'dropout': 0.0,
        'batch_norm': True,
    }
    # define projection layer
    enet = EnetProjectionWithResnet(**enet_kwargs)
    check_model_forward_pass(enet, enet_kwargs['input_dims'],
                             [enet_kwargs['output_dim']], batch_size)


def test_backward_pass_of_EnetProjectionWithResnet():
    """
    Test forward pass of the EnetProjectionWithResnet embedding network.
    """
    batch_size, n_rb, num_blocks, num_channels, num_bins = 1000, 10, 2, 3, 200
    _, (V1, V2) = generate_1d_datasets_and_reduced_basis(batch_size, num_bins)

    enet_kwargs = {
        'input_dims': (num_blocks, num_channels, num_bins),
        'n_rb': n_rb,
        'V_rb_list': (V1, V2),
        'output_dim': 8,
        'hidden_dims': [32, 16, 16, 8],
        'activation': torch.nn.functional.elu,
        'dropout': 0.0,
        'batch_norm': True,
    }
    # define projection layer
    enet = EnetProjectionWithResnet(**enet_kwargs)
    check_model_backward_pass(enet, enet_kwargs['input_dims'], batch_size)


def test_ModuleMerger():
    """
    TODO
    """
    batch_size, n_rb, num_blocks, num_channels, num_bins = 1000, 10, 2, 3, 200
    _, (V1, V2) = generate_1d_datasets_and_reduced_basis(batch_size, num_bins)
    enet_kwargs = {
        'input_dims': (num_blocks, num_channels, num_bins),
        'n_rb': n_rb,
        'V_rb_list': (V1, V2),
        'output_dim': 8,
        'hidden_dims': [32, 16, 16, 8],
        'activation': torch.nn.functional.elu,
        'dropout': 0.0,
        'batch_norm': True,
    }
    enet = ModuleMerger((nn.Identity, EnetProjectionWithResnet), ({}, enet_kwargs))
    x = (torch.ones(batch_size, 3),
         torch.rand(batch_size, *enet_kwargs['input_dims']))
    optimizer = optim.Adam(enet.parameters(), lr=0.001)
    loss_fn = nn.L1Loss()
    out = enet(x)
    loss = loss_fn(out, torch.zeros_like(out))
    loss.backward()
    optimizer.step()
    out_updated = enet(x)
    loss_updated = loss_fn(out_updated, torch.zeros_like(out))
    # check that identity mapping is applied correctly
    assert torch.all(out[:,:3] == 1), \
        'Individual embedding nets not applied correctly.'
    # check that loss improved
    assert loss_updated < loss, 'Backward pass or optimizer step did not work.'
    assert torch.all(out_updated[:,:3] == 1), \
        'Individual embedding nets not applied correctly.'


if __name__ == '__main__':
    pass
