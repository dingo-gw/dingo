import copy
import os
from typing import List, Literal, Optional, Union, Tuple, Iterable

import bilby
from bisect import bisect_right
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import (
    SequentialLR,
    ReduceLROnPlateau,
    _check_verbose_deprecated_warning,
)
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist


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


def adapt_scheduler_kwargs_to_update_every_optimizer_step(
    kwargs: dict, n_steps_per_epoch: int
) -> dict:
    """
    Adjusts scheduler kwargs to updates per optimizer step (instead of updates per epoch).

    Parameters
    ----------
    kwargs: dict
        scheduler kwargs
    n_steps_per_epoch: int
        number of optimizer steps per epoch to which the scheduler kwargs should be adapted

    Returns
    -------
    kwargs: dict
        updated scheduler kwargs
    """
    if kwargs["type"] == "step":
        kwargs["step_size"] = kwargs["step_size"] * n_steps_per_epoch
    elif kwargs["type"] == "cosine":
        kwargs["T_max"] = kwargs["T_max"] * n_steps_per_epoch
    elif kwargs["type"] == "reduce_on_plateau":
        raise ValueError(
            "The scheduler ReduceOnPlateau cannot be used with update_every_optimizer_step=True,"
            "because it depends on the validation loss."
        )
    elif kwargs["type"] == "linear":
        kwargs["total_iters"] = kwargs["total_iters"] * n_steps_per_epoch
    if "last_epoch" in kwargs:
        kwargs["last_epoch"] = kwargs["last_epoch"] * n_steps_per_epoch
    return kwargs


class CustomSequentialLR(SequentialLR):
    """
    Custom Sequential learning rate scheduler.
    - Overwrite __init__ to remove error message for ReduceLROnPlateau scheduler.
    - Overwrite step() to allow for a metric value that has to be passed to ReduceLROnPlateau.step(metric).
    These modifications can be removed once this PR is merged: https://github.com/pytorch/pytorch/issues/125531
    which will fix this bug https://github.com/pytorch/pytorch/issues/68978.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        schedulers: List[torch.optim.lr_scheduler.LRScheduler],
        milestones: List[int],
        last_epoch=-1,
        verbose="deprecated",
    ):
        if len(schedulers) < 1:
            raise ValueError(
                f"{self.__class__.__name__} expects at least one scheduler, but got no scheduler."
            )

        for scheduler_idx, scheduler in enumerate(schedulers):
            if not hasattr(scheduler, "optimizer"):
                raise TypeError(
                    f"{self.__class__.__name__} at index {scheduler_idx} should have `optimizer` as its attribute."
                )
            # Disable error message
            # if isinstance(scheduler, ReduceLROnPlateau):
            #     raise ValueError(
            #         f"{self.__class__.__name__} does not support `ReduceLROnPlateau` scheduler as it "
            #         "requires additional kwargs to be specified when calling `step`, "
            #         f"but got one at index {scheduler_idx} in the given schedulers sequence."
            #     )
            if optimizer != scheduler.optimizer:
                raise ValueError(
                    f"{self.__class__.__name__} expects all schedulers to belong to the same optimizer, but "
                    f"got scheduler {scheduler.__class__.__name__} at index {scheduler_idx} has {scheduler.optimizer}, "
                    f"which is different from {optimizer.__class__.__name__}."
                )

        if len(milestones) != len(schedulers) - 1:
            raise ValueError(
                "Sequential Schedulers expects number of schedulers provided to be one more "
                f"than the number of milestone points, but got number of schedulers {len(schedulers)} and the "
                f"number of milestones to be equal to {len(milestones)}"
            )
        _check_verbose_deprecated_warning(verbose)
        self._schedulers = schedulers
        self._milestones = milestones
        self.last_epoch = last_epoch + 1
        self.optimizer = optimizer

        # Reset learning rates back to initial values
        for group in self.optimizer.param_groups:
            group["lr"] = group["initial_lr"]

        # "Undo" the step performed by other schedulers
        for scheduler in self._schedulers:
            scheduler.last_epoch -= 1

        # Perform the initial step for only the first scheduler
        self._schedulers[0]._initial_step()

        self._last_lr = schedulers[0].get_last_lr()

    def step(self, metrics: Optional[float] = None) -> None:
        self.last_epoch += 1
        # Get index of active scheduler with -1 to ensure previous scheduler stopped when it should
        idx = bisect_right(self._milestones, self.last_epoch - 1)
        scheduler = self._schedulers[idx]
        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(metrics)
            else:
                scheduler.step(0)
        else:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(metrics)
            else:
                scheduler.step()

        self._last_lr = scheduler.get_last_lr()

    def get_active_scheduler_index(self):
        return bisect_right(self._milestones, self.last_epoch)


def get_scheduler_from_kwargs(
    optimizer: torch.optim.Optimizer,
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
    scheduler_kwargs:
        kwargs for scheduler; type needs to be one of [step, cosine,
        reduce_on_plateau, sequential, linear], the remaining kwargs are used for
        specific scheduler kwargs, such as learning rate and momentum

    Returns
    -------
    scheduler
    """
    scheduler_kwargs = copy.deepcopy(scheduler_kwargs)

    schedulers_dict = {
        "step": torch.optim.lr_scheduler.StepLR,
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
        "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "sequential": CustomSequentialLR,  # torch.optim.lr_scheduler.SequentialLR,
        "linear": torch.optim.lr_scheduler.LinearLR,
    }
    if "type" not in scheduler_kwargs:
        raise KeyError("Scheduler type needs to be specified.")
    if scheduler_kwargs["type"].lower() not in schedulers_dict:
        raise ValueError("No valid scheduler specified.")

    if scheduler_kwargs["type"] == "sequential":
        # List of schedulers
        # Collect scheduler list
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
                "'scheduler_0': {...}, 'scheduler_1: {...}' when using type sequential."
            )
        if scheduler_kwargs["milestones"] != sorted(scheduler_kwargs["milestones"]):
            raise ValueError(
                f"Milestones list is not monotonically increasing: {scheduler_kwargs['milestones']}"
            )

        num_optimizer_steps = scheduler_kwargs.pop("num_optimizer_steps_per_epoch", 1)
        epochs_per_scheduler = np.concatenate(
            (
                [scheduler_kwargs["milestones"][0]],
                np.diff(np.array(scheduler_kwargs["milestones"])),
            )
        ).tolist()
        if len(scheduler_kwargs["milestones"]) != num_scheduler - 1:
            raise ValueError(
                f"Length of milestones list: {scheduler_kwargs['milestones']} is not one less than the "
                f"number of schedulers: {num_scheduler}."
            )

        schedulers = []
        for i, scheduler_key in enumerate(scheduler_keys):
            # Get scheduler kwargs
            individual_scheduler_kwargs = scheduler_kwargs.pop(scheduler_key)
            if "type" not in individual_scheduler_kwargs:
                raise KeyError(
                    f"Scheduler type of {scheduler_key} needs to be specified."
                )
            # Check whether to update scheduler every optimizer step
            update_ever_optimizer_step = individual_scheduler_kwargs.pop(
                "update_every_optimizer_step", False
            )
            if update_ever_optimizer_step and num_optimizer_steps > 1:
                # Adapt scheduler kwargs (in place) to update every optimizer step
                individual_scheduler_kwargs = (
                    adapt_scheduler_kwargs_to_update_every_optimizer_step(
                        individual_scheduler_kwargs, num_optimizer_steps
                    )
                )
                # Adapt milestones (except for last scheduler because its milestone is defined by the number of epochs)
                if i < len(epochs_per_scheduler):
                    if i == 0:
                        scheduler_kwargs["milestones"][i] *= num_optimizer_steps
                    else:
                        scheduler_updates = (
                            epochs_per_scheduler[i] * num_optimizer_steps
                        )
                        scheduler_kwargs["milestones"][i] = (
                            scheduler_kwargs["milestones"][i - 1] + scheduler_updates
                        )
                    # Shift subsequent milestones
                    for j in range(i + 1, len(scheduler_kwargs["milestones"])):
                        scheduler_kwargs["milestones"][j] = (
                            scheduler_kwargs["milestones"][i] + epochs_per_scheduler[j]
                        )

            # Get type of scheduler
            individual_scheduler_type = individual_scheduler_kwargs.pop("type").lower()
            if individual_scheduler_type not in schedulers_dict:
                raise ValueError(f"No valid scheduler specified for {scheduler_key}.")
            # Initialize scheduler
            individual_scheduler = schedulers_dict[individual_scheduler_type](
                optimizer, **individual_scheduler_kwargs
            )
            schedulers.append(individual_scheduler)

        if scheduler_kwargs["milestones"] != sorted(scheduler_kwargs["milestones"]):
            raise ValueError(
                f"Modified milestones list is not monotonically increasing: {scheduler_kwargs['milestones']}"
            )

        # Create SequentialScheduler
        scheduler_kwargs.pop("type")
        return schedulers_dict["sequential"](optimizer, schedulers, **scheduler_kwargs)
    else:
        # Single scheduler

        # Check whether to update scheduler every optimizer step
        update_ever_optimizer_step = scheduler_kwargs.pop(
            "update_every_optimizer_step", False
        )
        if (
            update_ever_optimizer_step
            and "num_optimizer_steps_per_epoch" in scheduler_kwargs
        ):
            num_optimizer_steps = scheduler_kwargs.pop("num_optimizer_steps_per_epoch")
            # Adapt scheduler kwargs (in place) to update every optimizer step
            scheduler_kwargs = adapt_scheduler_kwargs_to_update_every_optimizer_step(
                scheduler_kwargs, num_optimizer_steps
            )
        # Create scheduler
        scheduler_type = scheduler_kwargs.pop("type")
        scheduler = schedulers_dict[scheduler_type]
        return scheduler(optimizer, **scheduler_kwargs)


def perform_scheduler_step(
    scheduler,
    scheduler_kwargs: dict,
    loss: float = None,
    update_level: str = "epoch",
):
    """
    Wrapper for scheduler.step(). If scheduler is ReduceLROnPlateau,
    then scheduler.step(loss) is called, if not, scheduler.step().

    Parameters
    ----------

    scheduler:
        scheduler for learning rate
    scheduler_kwargs: dict
        scheduler arguments for one or multiple schedulers. Each scheduler arguments can contain
        'update_scheduler_every_optimizer_step' (default=False) which determines whether to do a scheduler step every
        optimizer step or every epoch.
    loss: float, optional
        validation loss
    update_level: str, optional
        Describes from where this function is called, either on the epoch level or on the level of an 'optimizer_step'.
    """

    def perform_step(sched, metric: float = None):
        if metric is not None:
            sched.step(metric)
        else:
            sched.step()

    # Standard scheduler
    if not isinstance(scheduler, CustomSequentialLR):
        update_per_step = scheduler_kwargs.get("update_every_optimizer_step", False)
        if (update_level == "epoch" and not update_per_step) or (
            update_level == "optimizer_step" and update_per_step
        ):
            perform_step(scheduler, loss)
    # Sequential scheduler
    elif isinstance(scheduler, CustomSequentialLR):
        # Get currently active scheduler
        active_scheduler_index = scheduler.get_active_scheduler_index()
        active_scheduler = f"scheduler_{active_scheduler_index}"
        update_per_step = scheduler_kwargs[active_scheduler].get(
            "update_every_optimizer_step", False
        )
        if (update_level == "epoch" and not update_per_step) or (
            update_level == "optimizer_step" and update_per_step
        ):
            print(
                f"{update_level} update --- current:",
                scheduler.last_epoch,
                "of",
                scheduler._milestones,
                "-> activate scheduler index:",
                active_scheduler_index,
            )
            perform_step(scheduler, loss)


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
    world_size: int = None,
    rank: int = None,
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
    world_size: int = None
        total number of devices required for distributed data parallel training
    rank: int = None
        device rank required for distributed data parallel training

    Returns
    -------
    (train_loader, test_loader, Optional(train_sampler, None))
    """

    # Split the dataset. This function uses a fixed seed for reproducibility.
    train_dataset, test_dataset = split_dataset_into_train_and_test(
        dataset, train_fraction
    )

    if num_workers > 0:
        persistent_workers = True
    else:
        persistent_workers = False

    # Create dataloaders for multi-GPU training separately, because arguments `shuffle` and `sampler` in DataLoader
    # are mutually exclusive
    if rank is not None and world_size is not None:
        # Create DistributedSampler
        train_sampler = DistributedSampler(
            train_dataset, shuffle=True, num_replicas=world_size, rank=rank
        )
        test_sampler = DistributedSampler(
            test_dataset, shuffle=False, num_replicas=world_size, rank=rank
        )

        # Build DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            pin_memory=False,
            num_workers=num_workers,
            worker_init_fn=fix_random_seeds,
            sampler=train_sampler,
            persistent_workers=persistent_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            pin_memory=False,
            num_workers=num_workers,
            worker_init_fn=fix_random_seeds,
            sampler=test_sampler,
            persistent_workers=persistent_workers,
        )
    else:
        train_sampler = None
        # Build DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            worker_init_fn=fix_random_seeds,
            persistent_workers=persistent_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            worker_init_fn=fix_random_seeds,
            persistent_workers=persistent_workers,
        )

    return train_loader, test_loader, train_sampler


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


def set_seed_based_on_rank(rank: int):
    """
    Sets Numpy and Torch seeds for each GPU process based on the torch seed
    to ensure that they are different.
    """
    initial_torch_seed = torch.initial_seed()
    torch.manual_seed(initial_torch_seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(initial_torch_seed + rank)
        # Only use deterministic convolution algorithms
        torch.backends.cudnn.deterministic = True

    # Numpy expect a different seed range
    reduced_seed = int(initial_torch_seed) % (2**32 - 1)
    np.random.seed(reduced_seed + rank)


def setup_ddp(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    if dist.is_nccl_available():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    else:
        raise ValueError(
            "Backends nccl not available for multi-GPU training with distributed data parallel."
            "Go back to single-GPU training."
        )
    # Assign correct device to process
    torch.cuda.set_device(rank)

    print(
        f"Process group initialized with backend {dist.get_backend()}, rank {dist.get_rank()}, "
        f"world size {dist.get_world_size()}."
    )


def replace_BatchNorm_with_SyncBatchNorm(network: nn.Module):
    return nn.SyncBatchNorm.convert_sync_batchnorm(network)


def cleanup_ddp():
    dist.destroy_process_group()
    print(f"Destroyed process group.")
