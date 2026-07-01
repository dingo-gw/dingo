import pytest
import torch
import torch.nn as nn

from dingo.core.utils import torchutils


def test_get_activation_function_from_string():
    activation = torchutils.get_activation_function_from_string("elu")
    # Returns a callable activation (torch.nn.functional.elu).
    assert callable(activation)
    out = activation(torch.tensor([-1.0, 1.0]))
    assert out.shape == (2,)


def test_get_activation_function_unknown_raises():
    with pytest.raises(ValueError):
        torchutils.get_activation_function_from_string("not_an_activation")


def test_get_optimizer_from_kwargs():
    model = nn.Linear(2, 2)
    optimizer = torchutils.get_optimizer_from_kwargs(
        model.parameters(), type="adam", lr=0.01
    )
    assert isinstance(optimizer, torch.optim.Adam)


def test_get_scheduler_from_kwargs():
    model = nn.Linear(2, 2)
    optimizer = torchutils.get_optimizer_from_kwargs(
        model.parameters(), type="adam", lr=0.01
    )
    scheduler = torchutils.get_scheduler_from_kwargs(optimizer, type="cosine", T_max=10)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)


def test_split_dataset_into_train_and_test():
    dataset = torch.utils.data.TensorDataset(torch.arange(100.0))
    train, test = torchutils.split_dataset_into_train_and_test(dataset, 0.8)
    assert len(train) == 80
    assert len(test) == 20
    # The split is a partition (no shared indices, full coverage).
    assert len(train) + len(test) == len(dataset)


def test_get_number_of_model_parameters():
    model = nn.Linear(2, 3)  # weight 2*3 + bias 3 = 9
    assert torchutils.get_number_of_model_parameters(model) == 9


def test_get_lr_returns_optimizer_learning_rates():
    optimizer = torchutils.get_optimizer_from_kwargs(
        nn.Linear(2, 2).parameters(), type="adam", lr=0.007
    )
    assert torchutils.get_lr(optimizer) == [0.007]


def test_torch_detach_to_cpu():
    tensor = torch.ones(3, requires_grad=True)
    detached = torchutils.torch_detach_to_cpu(tensor)
    assert not detached.requires_grad
    # Non-tensor inputs pass through unchanged.
    assert torchutils.torch_detach_to_cpu(5) == 5


def test_set_requires_grad_flag_by_name():
    model = nn.Linear(2, 3)
    torchutils.set_requires_grad_flag(
        model, name_contains="weight", requires_grad=False
    )
    assert model.weight.requires_grad is False
    assert model.bias.requires_grad is True
