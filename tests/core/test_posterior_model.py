import pytest
import types
import os
from os.path import join
import numpy as np
import torch
from dingo.core.models.posterior_model import PosteriorModel
from dingo.core.utils import torchutils


@pytest.fixture()
def data_setup_pm_1():
    d = types.SimpleNamespace()

    tmp_dir = "./tmp_files"
    os.makedirs(tmp_dir, exist_ok=True)
    d.model_filename = join(tmp_dir, "model.pt")

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
        "input_dims": (2, 3, 20),
        # 'n_rb': 10,
        "svd": {"size": 10},
        "V_rb_list": None,
        "output_dim": 8,
        "hidden_dims": [32, 16, 8],
        "activation": "elu",
        "dropout": 0.0,
        "batch_norm": True,
        "added_context": True,
    }

    d.model_kwargs = {
        "type": "nsf+embedding",
        "nsf_kwargs": d.nsf_kwargs,
        "embedding_net_kwargs": d.embedding_net_kwargs,
    }

    d.metadata = {
        "train_settings": {"model": d.model_kwargs}
    }

    return d


@pytest.fixture()
def data_setup_optimizer_scheduler():
    d = types.SimpleNamespace()

    d.adam_kwargs = {
        "type": "adam",
        "lr": 0.0001,
    }
    d.sgd_kwargs = {
        "type": "sgd",
        "lr": 0.0001,
        "momentum": 0.9,
    }

    d.step_kwargs = {"type": "step", "step_size": 3, "gamma": 0.5}
    d.cosine_kwargs = {
        "type": "cosine",
        "T_max": 10,
    }
    d.rop_kwargs = {
        "type": "reduce_on_plateau",
        "factor": 0.5,
        "patience": 3,
    }

    d.epochs = range(1, 10 + 1)
    d.losses = [-1, -2, -3, -2, -2, -2, -2, -7, -8, -9]
    d.rop_factors = [1, 1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5]
    d.cosine_factors = (1 + np.cos(np.linspace(0, np.pi, 11))[:-1]) / 2
    d.step_factors = [
        0.5 ** 0,
        0.5 ** 0,
        0.5 ** 0,
        0.5 ** 1,
        0.5 ** 1,
        0.5 ** 1,
        0.5 ** 2,
        0.5 ** 2,
        0.5 ** 2,
        0.5 ** 3,
    ]

    return d


def test_pm_saving_and_loading_basic(data_setup_pm_1):
    """
    Test the most basic functionality of initializing, saving and loading a
    model. Two models built with identical hyperparameters should have the
    same architecture (names and shapes of parameters), but they should be
    initialized differently. A model loaded from a saved file should have the
    exact same parameter data as the original model.
    """
    d = data_setup_pm_1

    # initialize model and save it
    pm_0 = PosteriorModel(metadata=d.metadata, device='cpu')
    pm_0.save_model(d.model_filename)

    # load saved model
    pm_1 = PosteriorModel(model_filename=d.model_filename, device='cpu')

    # build a model with identical kwargs
    pm_2 = PosteriorModel(metadata=d.metadata, device='cpu')

    # check that module names are identical in saved and loaded model
    module_names_0 = [name for name, param in pm_0.model.named_parameters()]
    module_names_1 = [name for name, param in pm_1.model.named_parameters()]
    module_names_2 = [name for name, param in pm_2.model.named_parameters()]
    assert (
        module_names_0 == module_names_1 == module_names_2
    ), "Models have different module names."

    # check that number of parameters are identical in saved and loaded model
    num_params_0 = torchutils.get_number_of_model_parameters(pm_0.model)
    num_params_1 = torchutils.get_number_of_model_parameters(pm_1.model)
    num_params_2 = torchutils.get_number_of_model_parameters(pm_2.model)
    assert (
        num_params_0 == num_params_1 == num_params_2
    ), "Models have different number of parameters."

    # check that the weights of pm_0 and pm_1 are identical, but different
    # compared to pm_2
    for name, param_0, param_1, param_2 in zip(
        module_names_0,
        pm_0.model.parameters(),
        pm_1.model.parameters(),
        pm_2.model.parameters(),
    ):
        assert (
            param_0.data.shape == param_1.data.shape == param_2.data.shape
        ), "Models have different parameter shapes."
        assert torch.all(
            param_0.data == param_1.data
        ), "Saved and loaded model have different parameter data."
        if not torch.std(param_0) == 0:
            assert not torch.all(
                param_0.data == param_2.data
            ), "Model initialization does not seem to be random."


def test_pm_scheduler(data_setup_pm_1, data_setup_optimizer_scheduler):
    """
    Test that the scheduler of PosteriorModel works as intended, also when
    saving and loading the model.
    """
    d = data_setup_pm_1
    e = data_setup_optimizer_scheduler

    # Test step scheduler
    optimizer_kwargs = e.adam_kwargs
    scheduler_kwargs = e.step_kwargs
    pm = PosteriorModel(metadata=d.metadata, device='cpu')
    pm.optimizer_kwargs = optimizer_kwargs
    pm.scheduler_kwargs = scheduler_kwargs
    pm.initialize_optimizer_and_scheduler()
    pm.optimizer.step()  # this is to suppress a pytorch warning
    factors = []
    for epoch, loss in zip(e.epochs, e.losses):
        if epoch == 5:
            pm.save_model(d.model_filename, save_training_info=True)
            pm = PosteriorModel(model_filename=d.model_filename, device='cpu')
        lr = pm.optimizer.state_dict()["param_groups"][0]["lr"]
        factors.append(lr / pm.optimizer.defaults["lr"])
        torchutils.perform_scheduler_step(pm.scheduler, loss)
    assert factors == e.step_factors, "Scheduler does not load correctly."

    # Test reduce_on_plateau scheduler
    optimizer_kwargs = e.sgd_kwargs
    scheduler_kwargs = e.rop_kwargs
    pm = PosteriorModel(metadata=d.metadata, device='cpu')
    pm.optimizer_kwargs = optimizer_kwargs
    pm.scheduler_kwargs = scheduler_kwargs
    pm.initialize_optimizer_and_scheduler()
    pm.optimizer.step()  # this is to suppress a pytorch warning
    factors = []
    for epoch, loss in zip(e.epochs, e.losses):
        if epoch == 5:
            pm.save_model(d.model_filename, save_training_info=True)
            pm = PosteriorModel(model_filename=d.model_filename, device='cpu')
        lr = pm.optimizer.state_dict()["param_groups"][0]["lr"]
        factors.append(lr / pm.optimizer.defaults["lr"])
        torchutils.perform_scheduler_step(pm.scheduler, loss)
    assert factors == e.rop_factors, "Scheduler does not load correctly."

    # Test cosine scheduler
    optimizer_kwargs = e.sgd_kwargs
    scheduler_kwargs = e.cosine_kwargs
    pm = PosteriorModel(metadata=d.metadata, device='cpu')
    pm.optimizer_kwargs = optimizer_kwargs
    pm.scheduler_kwargs = scheduler_kwargs
    pm.initialize_optimizer_and_scheduler()
    pm.optimizer.step()  # this is to suppress a pytorch warning
    factors = []
    for epoch, loss in zip(e.epochs, e.losses):
        if epoch == 5:
            pm.save_model(d.model_filename, save_training_info=True)
            pm = PosteriorModel(model_filename=d.model_filename, device='cpu')
        lr = pm.optimizer.state_dict()["param_groups"][0]["lr"]
        factors.append(lr / pm.optimizer.defaults["lr"])
        torchutils.perform_scheduler_step(pm.scheduler, loss)
    assert np.allclose(factors, e.cosine_factors), "Scheduler does not load correctly."
