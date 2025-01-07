import pytest
import types
import os
from os.path import join
import numpy as np
import torch
from dingo.core.posterior_models.normalizing_flow import NormalizingFlowPosteriorModel
from dingo.core.utils import torchutils
from dingo.core.utils.scheduler import perform_scheduler_step


@pytest.fixture()
def data_setup_pm_1():
    d = types.SimpleNamespace()

    tmp_dir = "./tmp_files"
    os.makedirs(tmp_dir, exist_ok=True)
    d.model_filename = join(tmp_dir, "model.pt")

    d.posterior_kwargs = {
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
    d.embedding_kwargs = {
        "input_dims": (2, 3, 20),
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
        "posterior_model_type": "normalizing_flow",
        "posterior_kwargs": d.posterior_kwargs,
        "embedding_type": "DenseResidualNet",
        "embedding_kwargs": d.embedding_kwargs,
    }

    d.metadata = {"train_settings": {"model": d.model_kwargs}}

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

    d.epochs = range(1, 10 + 1)
    d.losses = [-1, -2, -3, -2, -2, -2, -2, -7, -8, -9]

    d.step_kwargs = {"type": "step", "step_size": 3, "gamma": 0.5}
    d.step_factors = [
        0.5**0,
        0.5**0,
        0.5**0,
        0.5**1,
        0.5**1,
        0.5**1,
        0.5**2,
        0.5**2,
        0.5**2,
        0.5**3,
    ]

    d.rop_kwargs = {
        "type": "reduce_on_plateau",
        "factor": 0.5,
        "patience": 3,
    }
    d.rop_factors = [1, 1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5]

    d.cosine_kwargs = {
        "type": "cosine",
        "T_max": 10,
    }
    d.cosine_factors = (1 + np.cos(np.linspace(0, np.pi, 11))[:-1]) / 2
    d.optimizer_steps_per_epoch = 5
    d.cosine_kwargs_opt_step = {
        "type": "cosine",
        "T_max": 10,
        "update_every_optimizer_step": True,
    }

    d.sequential_kwargs = {
        "type": "sequential",
        "milestones": [3, 6],
        "scheduler_0": {
            "type": "linear",
            "start_factor": 0.2,
            "total_iters": 3,
            "update_every_optimizer_step": True,
        },
        "scheduler_1": {
            "type": "reduce_on_plateau",
            "mode": "min",
            "factor": 0.1,
            "patience": 3,
            "update_every_optimizer_step": False,
            # True does not work for reduce_on_plateau because it depends on validation loss
        },
        "scheduler_2": {
            "type": "cosine",
            "T_max": 3,
            "update_every_optimizer_step": False,
        },
    }
    # Contributions:
    # - linear warmup from factor 0.2 to 1 for 4 epochs (0,1,2,3)
    # - constant lr of reduceLROnPlateau for 3 epochs (4,5,6)
    # - cosine annealing starting with factor=1.0 for 4 epochs (7,8,9)
    d.sequential_factors = np.concatenate(
        [
            np.array([0.2, 0.2 + (1 - 0.2) / 3, 0.2 + 2 * (1 - 0.2) / 3, 1]),
            np.array([1, 1, 1]),
            (1 + np.cos(np.linspace(0, np.pi, 3 + 1))[:-1]) / 2,
        ]
    ).tolist()

    d.sequential_v2_kwargs = {
        "type": "sequential",
        "milestones": [3, 6],
        "scheduler_0": {
            "type": "linear",
            "start_factor": 0.2,
            "total_iters": 3,
            "update_every_optimizer_step": True,
        },
        "scheduler_1": {
            "type": "reduce_on_plateau",
            "mode": "min",
            "factor": 0.1,
            "patience": 3,
            "update_every_optimizer_step": False,
            # True does not work for reduce_on_plateau because it depends on validation loss
        },
        "scheduler_2": {
            "type": "cosine",
            "T_max": 3,
            "update_every_optimizer_step": True,
        },
    }
    # Same contributions as above, but:
    # If we update the Cosine Annealing scheduler every optimizer step instead of every epoch, the learning rate values
    # change if the learning rates are saved after the lr update.
    # The spacing for updates every optimizer step is epochs * optimizer_steps_per_epoch and the learning rate is
    # logged after optimizer_steps_per_epoch-1 + i * optimizer_steps_per_epoch.
    cos_annealing = (
        1 + np.cos(np.linspace(0, np.pi, 3 * d.optimizer_steps_per_epoch + 1))
    )[:-1] / 2
    ca_every_epoch_downsampled = cos_annealing[
        d.optimizer_steps_per_epoch - 1 :: d.optimizer_steps_per_epoch
    ]
    d.sequential_factors_v2 = np.concatenate(
        [
            np.array([0.2, 0.2 + (1 - 0.2) / 3, 0.2 + 2 * (1 - 0.2) / 3, 1]),
            np.array([1, 1, 1]),
            ca_every_epoch_downsampled,
        ]
    ).tolist()

    d.sequential_v3_kwargs = {
        "type": "sequential",
        "milestones": [2, 5],
        "scheduler_0": {
            "type": "linear",
            "start_factor": 0.2,
            "total_iters": 2,
            "update_every_optimizer_step": True,
        },
        "scheduler_1": {
            "type": "reduce_on_plateau",
            "mode": "min",
            "factor": 0.1,
            "patience": 3,
            "update_every_optimizer_step": False,
            # True does not work for reduce_on_plateau because it depends on validation loss
        },
        "scheduler_2": {
            "type": "cosine",
            "T_max": 4,
            "update_every_optimizer_step": True,
        },
    }
    # Same contributions as above, but:
    # If we update the Cosine Annealing scheduler every optimizer step instead of every epoch, the learning rate values
    # change if the learning rates are saved after the lr update.
    # The spacing for updates every optimizer step is epochs * optimizer_steps_per_epoch and the learning rate is
    # logged after optimizer_steps_per_epoch-1 + i * optimizer_steps_per_epoch.
    cos_annealing = (
        1 + np.cos(np.linspace(0, np.pi, 4 * d.optimizer_steps_per_epoch + 1))
    )[:-1] / 2
    ca_every_epoch_downsampled = cos_annealing[
        d.optimizer_steps_per_epoch - 1 :: d.optimizer_steps_per_epoch
    ]
    d.sequential_factors_v3 = np.concatenate(
        [
            np.array([0.2, 0.2 + (1 - 0.2) / 2, 1]),
            np.array([1, 1, 1]),
            ca_every_epoch_downsampled,
        ]
    ).tolist()

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
    pm_0 = NormalizingFlowPosteriorModel(metadata=d.metadata, device="cpu")
    pm_0.save_model(d.model_filename)

    # load saved model
    pm_1 = NormalizingFlowPosteriorModel(model_filename=d.model_filename, device="cpu")

    # build a model with identical kwargs
    pm_2 = NormalizingFlowPosteriorModel(metadata=d.metadata, device="cpu")

    # check that module names are identical in saved and loaded model
    module_names_0 = [name for name, param in pm_0.network.named_parameters()]
    module_names_1 = [name for name, param in pm_1.network.named_parameters()]
    module_names_2 = [name for name, param in pm_2.network.named_parameters()]
    assert (
        module_names_0 == module_names_1 == module_names_2
    ), "Models have different module names."

    # check that number of parameters are identical in saved and loaded model
    num_params_0 = torchutils.get_number_of_model_parameters(pm_0.network)
    num_params_1 = torchutils.get_number_of_model_parameters(pm_1.network)
    num_params_2 = torchutils.get_number_of_model_parameters(pm_2.network)
    assert (
        num_params_0 == num_params_1 == num_params_2
    ), "Models have different number of parameters."

    # check that the weights of pm_0 and pm_1 are identical, but different
    # compared to pm_2
    for name, param_0, param_1, param_2 in zip(
        module_names_0,
        pm_0.network.parameters(),
        pm_1.network.parameters(),
        pm_2.network.parameters(),
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
    pm = NormalizingFlowPosteriorModel(metadata=d.metadata, device="cpu")
    pm.optimizer_kwargs = optimizer_kwargs
    pm.scheduler_kwargs = scheduler_kwargs
    pm.initialize_optimizer_and_scheduler()
    pm.optimizer.step()  # this is to suppress a pytorch warning
    factors = []
    for epoch, loss in zip(e.epochs, e.losses):
        if epoch == 5:
            pm.save_model(d.model_filename, save_training_info=True)
            pm = NormalizingFlowPosteriorModel(
                model_filename=d.model_filename, device="cpu"
            )
        lr = pm.optimizer.state_dict()["param_groups"][0]["lr"]
        factors.append(lr / pm.optimizer.defaults["lr"])
        perform_scheduler_step(pm.scheduler)
    assert (
        factors == e.step_factors
    ), "Learning rate factors of StepLR scheduler are not close."

    # Test reduce_on_plateau scheduler
    optimizer_kwargs = e.sgd_kwargs
    scheduler_kwargs = e.rop_kwargs
    pm = NormalizingFlowPosteriorModel(metadata=d.metadata, device="cpu")
    pm.optimizer_kwargs = optimizer_kwargs
    pm.scheduler_kwargs = scheduler_kwargs
    pm.initialize_optimizer_and_scheduler()
    pm.optimizer.step()  # this is to suppress a pytorch warning
    factors = []
    for epoch, loss in zip(e.epochs, e.losses):
        if epoch == 5:
            pm.save_model(d.model_filename, save_training_info=True)
            pm = NormalizingFlowPosteriorModel(
                model_filename=d.model_filename, device="cpu"
            )
        lr = pm.optimizer.state_dict()["param_groups"][0]["lr"]
        factors.append(lr / pm.optimizer.defaults["lr"])
        perform_scheduler_step(pm.scheduler, loss=loss)
    assert (
        factors == e.rop_factors
    ), "Learning rate factors of ReduceLROnPlateau scheduler are not close."

    # Test cosine scheduler
    optimizer_kwargs = e.sgd_kwargs
    scheduler_kwargs = e.cosine_kwargs
    pm = NormalizingFlowPosteriorModel(metadata=d.metadata, device="cpu")
    pm.optimizer_kwargs = optimizer_kwargs
    pm.scheduler_kwargs = scheduler_kwargs
    pm.initialize_optimizer_and_scheduler()
    pm.optimizer.step()  # this is to suppress a pytorch warning
    factors = []
    for epoch, loss in zip(e.epochs, e.losses):
        if epoch == 5:
            pm.save_model(d.model_filename, save_training_info=True)
            pm = NormalizingFlowPosteriorModel(
                model_filename=d.model_filename,
                device="cpu"
            )
        lr = pm.optimizer.state_dict()["param_groups"][0]["lr"]
        factors.append(lr / pm.optimizer.defaults["lr"])
        perform_scheduler_step(pm.scheduler)
    assert np.allclose(
        factors, e.cosine_factors
    ), "Learning rate factors of CosineAnnealingLR scheduler are not close."

    # Test cosine scheduler with update_every_optimizer_step
    optimizer_kwargs = e.sgd_kwargs
    scheduler_kwargs = e.cosine_kwargs_opt_step
    pm = NormalizingFlowPosteriorModel(metadata=d.metadata, device="cpu")
    pm.optimizer_kwargs = optimizer_kwargs
    pm.scheduler_kwargs = scheduler_kwargs
    pm.initialize_optimizer_and_scheduler(
        num_optimizer_steps=e.optimizer_steps_per_epoch
    )
    pm.optimizer.step()  # this is to suppress a pytorch warning
    factors = []
    for epoch, loss in zip(e.epochs, e.losses):
        if epoch == 5:
            pm.save_model(d.model_filename, save_training_info=True)
            pm = NormalizingFlowPosteriorModel(
                model_filename=d.model_filename,
                device="cpu"
            )
        lr = pm.optimizer.state_dict()["param_groups"][0]["lr"]
        factors.append(lr / pm.optimizer.defaults["lr"])
        for _ in range(e.optimizer_steps_per_epoch):
            perform_scheduler_step(
                pm.scheduler,
                scheduler_kwargs=scheduler_kwargs,
                update_level="optimizer_step",
            )
        perform_scheduler_step(
            pm.scheduler, scheduler_kwargs=scheduler_kwargs, update_level="epoch"
        )
    assert np.allclose(
        factors, e.cosine_factors
    ), "Learning rate factors of CosineAnnealingLR scheduler are not close."


def test_pm_sequential_scheduler(data_setup_pm_1, data_setup_optimizer_scheduler):
    """
    Test that the sequential scheduler of PosteriorModel works as intended, also when
    saving and loading the model.
    """
    d = data_setup_pm_1
    e = data_setup_optimizer_scheduler

    # Test sequential scheduler
    optimizer_kwargs = e.adam_kwargs
    scheduler_kwargs = e.sequential_kwargs
    pm = NormalizingFlowPosteriorModel(metadata=d.metadata, device="cpu")
    pm.optimizer_kwargs = optimizer_kwargs
    pm.scheduler_kwargs = scheduler_kwargs
    pm.initialize_optimizer_and_scheduler(
        num_optimizer_steps=e.optimizer_steps_per_epoch
    )
    pm.optimizer.step()  # this is to suppress a pytorch warning
    factors = []
    for epoch, loss in zip(e.epochs, e.losses):
        if epoch == 5:
            pm.save_model(d.model_filename, save_training_info=True)
            pm = NormalizingFlowPosteriorModel(
                model_filename=d.model_filename,
                device="cpu"
            )
        lr = pm.optimizer.state_dict()["param_groups"][0]["lr"]
        factors.append(lr / pm.optimizer.defaults["lr"])
        for _ in range(e.optimizer_steps_per_epoch):
            perform_scheduler_step(
                pm.scheduler,
                scheduler_kwargs=scheduler_kwargs,
                update_level="optimizer_step",
            )
        perform_scheduler_step(
            pm.scheduler,
            loss=loss,
            scheduler_kwargs=scheduler_kwargs,
            update_level="epoch",
        )
    assert np.allclose(
        factors, e.sequential_factors
    ), "Learning rate factors of CustomSequentialLR scheduler are not close."

    # Test sequential scheduler but with changed updates_every_optimizer_step
    optimizer_kwargs = e.adam_kwargs
    scheduler_kwargs = e.sequential_v2_kwargs
    pm = NormalizingFlowPosteriorModel(metadata=d.metadata, device="cpu")
    pm.optimizer_kwargs = optimizer_kwargs
    pm.scheduler_kwargs = scheduler_kwargs
    pm.initialize_optimizer_and_scheduler(
        num_optimizer_steps=e.optimizer_steps_per_epoch
    )
    pm.optimizer.step()  # this is to suppress a pytorch warning
    factors = []
    for epoch, loss in zip(e.epochs, e.losses):
        if epoch == 5:
            pm.save_model(d.model_filename, save_training_info=True)
            pm = NormalizingFlowPosteriorModel(
                model_filename=d.model_filename,
                device="cpu"
            )
        lr = pm.optimizer.state_dict()["param_groups"][0]["lr"]
        factors.append(lr / pm.optimizer.defaults["lr"])
        for _ in range(e.optimizer_steps_per_epoch):
            perform_scheduler_step(
                pm.scheduler,
                scheduler_kwargs=scheduler_kwargs,
                update_level="optimizer_step",
            )
        perform_scheduler_step(
            pm.scheduler,
            loss=loss,
            scheduler_kwargs=scheduler_kwargs,
            update_level="epoch",
        )
    assert np.allclose(
        factors, e.sequential_factors_v2
    ), "Learning rate factors of CustomSequentialLR scheduler are not close."

    # Test sequential scheduler but with changed milestone values
    optimizer_kwargs = e.adam_kwargs
    scheduler_kwargs = e.sequential_v3_kwargs
    pm = NormalizingFlowPosteriorModel(metadata=d.metadata, device="cpu")
    pm.optimizer_kwargs = optimizer_kwargs
    pm.scheduler_kwargs = scheduler_kwargs
    pm.initialize_optimizer_and_scheduler(
        num_optimizer_steps=e.optimizer_steps_per_epoch
    )
    pm.optimizer.step()  # this is to suppress a pytorch warning
    factors = []
    for epoch, loss in zip(e.epochs, e.losses):
        if epoch == 5:
            pm.save_model(d.model_filename, save_training_info=True)
            pm = NormalizingFlowPosteriorModel(model_filename=d.model_filename, device="cpu")
        lr = pm.optimizer.state_dict()["param_groups"][0]["lr"]
        factors.append(lr / pm.optimizer.defaults["lr"])
        for _ in range(e.optimizer_steps_per_epoch):
            perform_scheduler_step(
                pm.scheduler,
                scheduler_kwargs=scheduler_kwargs,
                update_level="optimizer_step",
            )
        perform_scheduler_step(
            pm.scheduler,
            loss=loss,
            scheduler_kwargs=scheduler_kwargs,
            update_level="epoch",
        )
    assert np.allclose(
        factors, e.sequential_factors_v3
    ), "Learning rate factors of CustomSequentialLR scheduler are not close."
