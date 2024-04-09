import copy
from typing import Optional, Union, Tuple, Iterable

import bilby
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


def fix_random_seeds(_):
    """Utility function to set random seeds when using multiple workers for DataLoader."""
    np.random.seed(int(torch.initial_seed()) % (2**32 - 1))
    try:
        bilby.core.utils.random.seed(int(torch.initial_seed()) % (2**32 - 1))
    except AttributeError:  # In case using an old version of Bilby.
        pass


def get_activation_function_from_string(activation_name: str):
    """
    Returns an activation function, based on the name provided.

    :param activation_name: str
        name of the activation function, one of {'elu', 'relu', 'leaky_rely'}
    :return: function
        corresponding activation function
    """
    if activation_name.lower() == "elu":
        return F.elu
    elif activation_name.lower() == "relu":
        return F.relu
    elif activation_name.lower() == "leaky_relu":
        return F.leaky_relu
    elif activation_name.lower() == "gelu":
        return F.gelu
    else:
        raise ValueError("Invalid activation function.")


def forward_pass_with_unpacked_tuple(
    model: nn.Module,
    x: Union[Tuple, torch.Tensor],
):
    """
    Performs forward pass of model with input x. If x is a tuple, it return
    y = model(*x), else it returns y = model(x).
    :param model: nn.Module
        model for forward pass
    :param x: Union[Tuple, torch.Tensor]
        input for forward pass
    :return: torch.Tensor
        output of the forward pass, either model(*x) or model(x)
    """
    if isinstance(x, Tuple):
        return model(*x)
    else:
        return model(x)


def get_number_of_model_parameters(
    model: nn.Module,
    requires_grad_flags: tuple = (True, False),
):
    """
    Counts parameters of the module. The list requires_grad_flag can be used
    to specify whether all parameters should be counted, or only those with
    requires_grad = True or False.
    :param model: nn.Module
        model
    :param requires_grad_flags: tuple
        tuple of bools, for requested requires_grad flags
    :return:
        number of parameters of the model with requested required_grad flags
    """
    num_params = 0
    for p in list(model.parameters()):
        if p.requires_grad in requires_grad_flags:
            n = 1
            for s in list(p.size()):
                n = n * s
            num_params += n
    return num_params


def get_optimizer_from_kwargs(
    model_parameters: Iterable,
    **optimizer_kwargs,
):
    """
    Builds and returns an optimizer for model_parameters. The type of the
    optimizer is determined by kwarg type, the remaining kwargs are passed to
    the optimizer.

    Parameters
    ----------
    model_parameters: Iterable
        iterable of parameters to optimize or dicts defining parameter groups
    optimizer_kwargs:
        kwargs for optimizer; type needs to be one of [adagrad, adam, adamw,
        lbfgs, RMSprop, sgd], the remaining kwargs are used for specific
        optimizer kwargs, such as learning rate and momentum

    Returns
    -------
    optimizer
    """
    optimizers_dict = {
        "adagrad": torch.optim.Adagrad,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "lbfgs": torch.optim.LBFGS,
        "RMSprop": torch.optim.RMSprop,
        "sgd": torch.optim.SGD,
    }
    if not "type" in optimizer_kwargs:
        raise KeyError("Optimizer type needs to be specified.")
    if not optimizer_kwargs["type"].lower() in optimizers_dict:
        raise ValueError("No valid optimizer specified.")
    optimizer = optimizers_dict[optimizer_kwargs.pop("type")]
    return optimizer(model_parameters, **optimizer_kwargs)


def get_scheduler_from_kwargs(
    optimizer: torch.optim.Optimizer,
    num_batches: Optional[int] = None,
    **scheduler_kwargs,
):
    """
    Builds and returns a scheduler for optimizer. The type of the
    scheduler is determined by kwarg type, the remaining kwargs are passed to
    the scheduler.

    Parameters
    ----------
    optimizer: torch.optim.optimizer.Optimizer
        optimizer for which the scheduler is used
    num_batches: Optional[int]
        number of batches to update the scheduler parameters if scheduler.update_scheduler_every_batch=True
    scheduler_kwargs:
        kwargs for scheduler; type needs to be one of [step, cosine,
        reduce_on_plateau, sequential, linear], the remaining kwargs are used for
        specific scheduler kwargs, such as learning rate and momentum

    Returns
    -------
    scheduler
    """

    def adapt_scheduler_kwargs_to_update_every_batch(kwargs, n_batches):
        if kwargs["type"] == "step":
            kwargs["step_size"] = kwargs["step_size"] * n_batches
        elif kwargs["type"] == "cosine":
            kwargs["T_max"] = kwargs["T_max"] * n_batches
        elif kwargs["type"] == "reduce_on_plateau":
            raise ValueError(
                "The scheduler ReduceOnPlateau cannot be used with update_scheduler_every_batch=True,"
                "because it depends on the validation loss."
            )
        elif kwargs["type"] == "linear":
            kwargs["total_iters"] = kwargs["total_iters"] * n_batches
        if "last_epoch" in kwargs:
            kwargs["last_epoch"] = kwargs["last_epoch"] * n_batches

    scheduler_kwargs = copy.deepcopy(scheduler_kwargs)

    schedulers_dict = {
        "step": torch.optim.lr_scheduler.StepLR,
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
        "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "sequential": torch.optim.lr_scheduler.SequentialLR,
        "linear": torch.optim.lr_scheduler.LinearLR,
    }
    if not "type" in scheduler_kwargs:
        raise KeyError("Scheduler type needs to be specified.")
    if not scheduler_kwargs["type"].lower() in schedulers_dict:
        raise ValueError("No valid scheduler specified.")

    if scheduler_kwargs["type"] == "sequential":
        # Load individual schedulers
        scheduler_keys = []
        num_scheduler = 0
        while True:
            scheduler_key = f"scheduler_{num_scheduler}"
            if scheduler_key in scheduler_kwargs:
                scheduler_keys.append(scheduler_key)
                num_scheduler += 1
            else:
                break
        if len(scheduler_keys) < 2:
            raise KeyError(
                "At least two schedulers need to be specified via "
                "'scheduler_0': {...}, 'scheduler_1: {...}' when using sequential."
            )
        update_scheduler_every_batch = scheduler_kwargs.pop(
            "update_scheduler_every_batch"
        )
        if update_scheduler_every_batch:
            scheduler_kwargs["milestones"] = [
                milestone * num_batches for milestone in scheduler_kwargs["milestones"]
            ]
        schedulers = []
        for scheduler_key in scheduler_keys:
            individual_scheduler_kwargs = scheduler_kwargs.pop(scheduler_key)
            if "type" not in individual_scheduler_kwargs:
                raise KeyError(
                    f"Scheduler type of {scheduler_key} needs to be specified."
                )
            if update_scheduler_every_batch and num_batches is not None:
                adapt_scheduler_kwargs_to_update_every_batch(
                    individual_scheduler_kwargs, num_batches
                )
            individual_scheduler_type = individual_scheduler_kwargs.pop("type").lower()
            if individual_scheduler_type not in schedulers_dict:
                raise ValueError(f"No valid scheduler specified for {scheduler_key}.")
            individual_scheduler = schedulers_dict[individual_scheduler_type](
                optimizer, **individual_scheduler_kwargs
            )
            schedulers.append(individual_scheduler)

        # Create SequentialScheduler
        scheduler_kwargs.pop("type")
        return schedulers_dict["sequential"](optimizer, schedulers, **scheduler_kwargs)
    else:
        if (
            scheduler_kwargs.pop("update_scheduler_every_batch")
            and num_batches is not None
        ):
            adapt_scheduler_kwargs_to_update_every_batch(scheduler_kwargs, num_batches)
        scheduler_type = scheduler_kwargs.pop("type")
        scheduler = schedulers_dict[scheduler_type]
        return scheduler(optimizer, **scheduler_kwargs)


def perform_scheduler_step(
    scheduler,
    loss=None,
):
    """
    Wrapper for scheduler.step(). If scheduler is ReduceLROnPlateau,
    then scheduler.step(loss) is called, if not, scheduler.step().

    Parameters
    ----------

    scheduler:
        scheduler for learning rate
    loss:
        validation loss
    """
    if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
        scheduler.step(loss)
    else:
        scheduler.step()


def get_lr(optimizer):
    """Returns a list with the learning rates of the optimizer."""
    return [param_group["lr"] for param_group in optimizer.state_dict()["param_groups"]]


def split_dataset_into_train_and_test(dataset, train_fraction):
    """
    Splits dataset into a trainset of size int(train_fraction * len(dataset)),
    and a testset with the remainder. Uses fixed random seed for
    reproducibility.

    Parameters
    ----------
    dataset: torch.utils.data.Datset
        dataset to be split
    train_fraction: float
        fraction of the dataset to be used for trainset

    Returns
    -------
    trainset, testset
    """
    train_size = int(train_fraction * len(dataset))
    test_size = len(dataset) - train_size
    return torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )


def build_train_and_test_loaders(
    dataset: torch.utils.data.Dataset,
    train_fraction: float,
    batch_size: int,
    num_workers: int,
):
    """
    Split the dataset into train and test sets, and build corresponding DataLoaders.
    The random split uses a fixed seed for reproducibility.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
    train_fraction : float
        Fraction of dataset to use for training. The remainder is used for testing.
        Should lie between 0 and 1.
    batch_size : int
    num_workers : int

    Returns
    -------
    (train_loader, test_loader)
    """

    # Split the dataset. This function uses a fixed seed for reproducibility.
    train_dataset, test_dataset = split_dataset_into_train_and_test(
        dataset, train_fraction
    )

    # Build DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=fix_random_seeds,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=fix_random_seeds,
    )

    return train_loader, test_loader


def set_requires_grad_flag(
    model, name_startswith=None, name_contains=None, requires_grad=True
):
    """
    Set param.requires_grad of all model parameters with a name starting with
    name_startswith, or name containing name_contains, to requires_grad.
    """
    for name, param in model.named_parameters():
        if (
            name_startswith is not None
            and name.startswith(name_startswith)
            or name_contains is not None
            and name_contains in name
        ):
            param.requires_grad = requires_grad


def torch_detach_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return x
