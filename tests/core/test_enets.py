import pytest
from testutils_enets import *
from dingo.core.nn.enets import LinearProjectionRB, DenseResidualNet, \
    ModuleMerger, create_enet_with_projection_layer_and_dense_resnet


@pytest.fixture()
def data_setup_rb():
    batch_size = 1000
    num_blocks, num_channels, num_bins = 2, 3, 200
    n_rb = 10
    (y1, y2), (V1, V2) = \
        generate_1d_datasets_and_reduced_basis(batch_size, num_bins)
    enet_kwargs = {
        'input_dims': (num_blocks, num_channels, num_bins),
        #'n_rb': n_rb,
        'V_rb_list': (V1, V2),
        'output_dim': 8,
        'hidden_dims': [32, 16, 16, 8],
        'activation': 'elu',
        'dropout': 0.0,
        'batch_norm': True,
        'svd': {'size': n_rb}
    }
    return {
        'batch_size': batch_size,
        'num_blocks': num_blocks,
        'num_channels': num_channels,
        'num_bins': num_bins,
        'y1': y1,
        'y2': y2,
        'V1': V1,
        'V2': V2,
        'n_rb': n_rb,
        'enet_kwargs': enet_kwargs,
    }


def test_projection_of_LinearProjectionRB(data_setup_rb):
    """
    This function tests the LinearProjectionRB embedding network for the
    dimensions and the correct application of the reduced basis projection.
    """
    d = data_setup_rb
    # define projection layer
    projection_layer = LinearProjectionRB(
        input_dims=[d['num_blocks'], d['num_channels'], d['num_bins']],
        n_rb=d['n_rb'], V_rb_list=(d['V1'], d['V2']))
    # prepare data for projection_layer
    y_batch_a = get_y_batch([d['y1'], d['y2']], num_channels=d['num_channels'])
    y_batch_b = get_y_batch([np.ones_like(d['y1']), np.zeros_like(d['y1'])],
                            num_channels=d['num_channels'])
    # project onto reduced basis with projection layer
    out_a = np.array(projection_layer(y_batch_a).detach())
    out_b = np.array(projection_layer(y_batch_b).detach())
    # compute reduced basis projections with matrix multiplications as
    # comparison
    ref_a = project_onto_reduced_basis_and_concat(
        [d['y1'], d['y2']], [d['V1'], d['V2']], d['n_rb'])
    ref_b = project_onto_reduced_basis_and_concat(
        [np.ones_like(d['y1']), np.zeros_like(d['y1'])],
        [d['V1'], d['V1']], d['n_rb'])
    # check if results agree
    thr = 2e-5
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
        input_dims=[2, d['num_channels'], d['num_bins']],
        n_rb=d['n_rb'], V_rb_list=None)
    out_a_1 = np.array(projection_layer(y_batch_a).detach())
    y_batch_a[:, :, 2:, :] -= torch.rand_like(y_batch_a[:, :, 2:, :])
    out_a_2 = np.array(projection_layer(y_batch_a).detach())
    assert np.all(out_a_1 != out_a_2), \
        'Channels with index >= 2 should affect projection layer when not ' \
        'initialized with reduced basis.'


def test_valueerrors_of_LinearProjectionRB(data_setup_rb):
    """
    Check that Error is raised if layer is initialized with inconsistent input.
    """
    d = data_setup_rb
    with pytest.raises(ValueError):
        LinearProjectionRB(
            input_dims=[2, d['num_channels'], d['num_bins']],
            n_rb=d['n_rb'], V_rb_list=(d['V1'], np.zeros_like(d['y1'])))
    with pytest.raises(ValueError):
        LinearProjectionRB(
            input_dims=[2, d['num_channels'], d['num_bins']],
            n_rb=d['n_rb'], V_rb_list=(d['V1'], d['V2'], d['V2']))
    with pytest.raises(ValueError):
        LinearProjectionRB(
            input_dims=[2, d['num_channels'], d['num_bins'] + 1],
            n_rb=d['n_rb'], V_rb_list=(d['V1'], d['V2']))
    with pytest.raises(ValueError):
        LinearProjectionRB(
            input_dims=[2, 1, d['num_bins']],
            n_rb=d['n_rb'], V_rb_list=(d['V1'], d['V2']))
    with pytest.raises(ValueError):
        LinearProjectionRB(
            input_dims=[2, d['num_channels'], d['num_bins'], 1],
            n_rb=d['n_rb'], V_rb_list=(d['V1'], d['V2']))


def test_forward_pass_of_LinearProjectionRB(data_setup_rb):
    """
    Test forward pass of the LinearProjectionRB embedding network.
    """
    d = data_setup_rb
    enet = LinearProjectionRB(
        input_dims=[d['num_blocks'], d['num_channels'], d['num_bins']],
        n_rb=d['n_rb'], V_rb_list=(d['V1'], d['V2']))
    check_model_forward_pass(
        enet, [enet.output_dim],
        (d['num_blocks'], d['num_channels'], d['num_bins']), d['batch_size'])


def test_backward_pass_of_LinearProjectionRB(data_setup_rb):
    """
    Test backward pass of the LinearProjectionRB embedding network.
    """
    d = data_setup_rb
    enet = LinearProjectionRB(
        input_dims=[d['num_blocks'], d['num_channels'], d['num_bins']],
        n_rb=d['n_rb'], V_rb_list=(d['V1'], d['V2']))
    check_model_backward_pass(
        enet, (d['num_blocks'], d['num_channels'], d['num_bins']),
        d['batch_size'])


def test_forward_pass_of_DenseResidualNet():
    """
    Test forward pass of the DenseResidualNet embedding network.
    """
    batch_size = 100
    input_dim, output_dim, hidden_dims = 120, 8, (128, 64, 32, 64, 16, 16)
    enet = DenseResidualNet(input_dim, output_dim, hidden_dims)
    check_model_forward_pass(enet, [output_dim], [input_dim], batch_size)


def test_backward_pass_of_DenseResidualNet():
    """
    Test backward pass of the DenseResidualNet embedding network.
    """
    batch_size = 100
    input_dim, output_dim, hidden_dims = 120, 8, (128, 64, 32, 64, 16, 16)
    enet = DenseResidualNet(input_dim, output_dim, hidden_dims)
    check_model_backward_pass(enet, [input_dim], batch_size)


def test_forward_pass_of_2stage_enet(data_setup_rb):
    """
    Test forward pass of the embedding network built by
    create_enet_with_projection_layer_and_dense_resnet.
    """
    d = data_setup_rb
    enet = create_enet_with_projection_layer_and_dense_resnet(
        **d['enet_kwargs'])
    check_model_forward_pass(enet, [d['enet_kwargs']['output_dim']],
                             d['enet_kwargs']['input_dims'], d['batch_size'])


def test_backward_pass_of_2stage_enet(data_setup_rb):
    """
    Test backward pass of the embedding network built by
    create_enet_with_projection_layer_and_dense_resnet.
    """
    d = data_setup_rb
    enet = create_enet_with_projection_layer_and_dense_resnet(
        **d['enet_kwargs'])
    check_model_backward_pass(enet, d['enet_kwargs']['input_dims'],
                              d['batch_size'])


def test_ModuleMerger(data_setup_rb):
    """
    Test the ModuleMerger class for correct outputs and backward passes.
    """
    d = data_setup_rb
    enet_kwargs = d['enet_kwargs']
    enet = ModuleMerger(
        (nn.Identity(),
         create_enet_with_projection_layer_and_dense_resnet(**enet_kwargs)))
    x = (torch.ones(d['batch_size'], 3),
         torch.rand(d['batch_size'], *enet_kwargs['input_dims']))

    # check backward pass and optimizer step for the model
    out_0, out_1 = check_model_backward_pass(enet, x=x)

    # check that additional context is left unchanged
    assert torch.all(out_0[:, :3] == 1), \
        'Individual embedding nets not applied correctly.'
    assert torch.all(out_1[:, :3] == 1), \
        'Individual embedding nets not applied correctly.'


def test_forward_pass_of_2stage_enet_with_context(data_setup_rb):
    """
    Test forward pass of the embedding network built by
    create_enet_with_projection_layer_and_dense_resnet with additional context.
    Check that ValueError is raised when enet is provided wrong input.
    """
    d = data_setup_rb
    enet_kwargs = d['enet_kwargs']
    enet = create_enet_with_projection_layer_and_dense_resnet(
        **enet_kwargs, added_context=True)

    # define primary and additional context
    x = torch.rand((d['batch_size'], *enet_kwargs['input_dims']))
    z = torch.ones((d['batch_size'], 2))

    check_model_forward_pass(enet, [enet_kwargs['output_dim'] + 2], x=(x, z))

    _ = enet(x, z)
    with pytest.raises(ValueError):
        enet(x)
    with pytest.raises(ValueError):
        enet((x,))


def test_backward_pass_of_2stage_enet_with_context(data_setup_rb):
    """
    Test backward pass of the embedding network built by
    create_enet_with_projection_layer_and_dense_resnet with additional context.
    """
    d = data_setup_rb
    enet_kwargs = d['enet_kwargs']
    enet = create_enet_with_projection_layer_and_dense_resnet(
        **enet_kwargs, added_context=True)

    # define primary and additional context
    x = torch.rand((d['batch_size'], *enet_kwargs['input_dims']))
    z = torch.ones((d['batch_size'], 2))

    y1, y2 = check_model_backward_pass(enet, x=(x, z))
    assert torch.all(y1[:, -2:] == 1) and torch.all(y2[:, -2:] == 1), \
        'Indentity mapping for additional context is broken.'


if __name__ == '__main__':
    pass
