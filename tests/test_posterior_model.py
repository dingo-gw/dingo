import pytest
import types
import os
from os.path import join
import torch
from dingo.core.nn.nsf import create_nsf_with_rb_projection_embedding_net
from dingo.core.models import PosteriorModel
from dingo.core.utils import torchutils


@pytest.fixture()
def data_setup_pm_1():
    d = types.SimpleNamespace()

    tmp_dir = './tmp_files'
    os.makedirs(tmp_dir, exist_ok=True)
    d.model_filename = join(tmp_dir, 'model.pt')

    d.nsf_kwargs = {
        "input_dim": 4,
        "context_dim": 10,
        "num_flow_steps": 5,
        "base_transform_kwargs": {
            "hidden_dim": 64,
            "num_transform_blocks": 2,
            "activation": "elu",
            "dropout_probability": 0.0,
            "batch_norm": True,
            "num_bins": 8,
            "base_transform_type": "rq-coupling",
        },
    }
    d.embedding_net_kwargs = {
        'input_dims': (2, 3, 20),
        'n_rb': 10,
        'V_rb_list': None,
        'output_dim': 8,
        'hidden_dims': [32, 16, 8],
        'activation': 'elu',
        'dropout': 0.0,
        'batch_norm': True,
        'added_context': True,
    }
    return d


def test_pm_saving_and_loading(data_setup_pm_1):
    """
    Test the most basic functionality of initializing, saving and loading a
    model. Two models built with identical hyperparameters should have the
    same architecture (names and shapes of parameters), but they should be
    initialized differently. A model loaded from a saved file should have the
    exact same parameter data as the original model.
    """
    d = data_setup_pm_1

    # initialize model and save it
    pm_0 = PosteriorModel(
        model_builder=create_nsf_with_rb_projection_embedding_net,
        model_kwargs={'nsf_kwargs': d.nsf_kwargs,
                      'embedding_net_kwargs': d.embedding_net_kwargs},
    )
    pm_0.save_model(d.model_filename)

    # load saved model
    pm_1 = PosteriorModel(
        model_builder=create_nsf_with_rb_projection_embedding_net,
        model_filename=d.model_filename,
    )

    # build a model with identical kwargs
    pm_2 = PosteriorModel(
        model_builder=create_nsf_with_rb_projection_embedding_net,
        model_kwargs={'nsf_kwargs': d.nsf_kwargs,
                      'embedding_net_kwargs': d.embedding_net_kwargs},
    )

    # check that module names are identical in saved and loaded model
    module_names_0 = [name for name, param in pm_0.model.named_parameters()]
    module_names_1 = [name for name, param in pm_1.model.named_parameters()]
    module_names_2 = [name for name, param in pm_2.model.named_parameters()]
    assert module_names_0 == module_names_1 == module_names_2, \
        'Models have different module names.'

    # check that number of parameters are identical in saved and loaded model
    num_params_0 = torchutils.get_number_of_model_parameters(pm_0.model)
    num_params_1 = torchutils.get_number_of_model_parameters(pm_1.model)
    num_params_2 = torchutils.get_number_of_model_parameters(pm_2.model)
    assert num_params_0 == num_params_1 == num_params_2, \
        'Models have different number of parameters.'

    # check that the weights of pm_0 and pm_1 are identical, but different
    # compared to pm_2
    for name, param_0, param_1, param_2 in zip(module_names_0,
                                               pm_0.model.parameters(),
                                               pm_1.model.parameters(),
                                               pm_2.model.parameters()):
        assert param_0.data.shape == param_1.data.shape == param_2.data.shape, \
            'Models have different parameter shapes.'
        assert torch.all(param_0.data == param_1.data), \
            'Saved and loaded model have different parameter data.'
        if not torch.std(param_0) == 0:
            assert not torch.all(param_0.data == param_2.data), \
                'Model initialization does not seem to be random.'
