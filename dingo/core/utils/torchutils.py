import os
from typing import Any, Iterable, Tuple, Union
from pathlib import Path

import bilby
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist


def get_cuda_info() -> dict[str, Any]:
    """Get information about the CUDA devices available in the system."""

    # No CUDA devices available
    if not torch.cuda.is_available():
        return {}

    # CUDA devices are available
    return {
        "cuDNN version": torch.backends.cudnn.version(),  # type: ignore
        "CUDA version": torch.version.cuda,
        "device count": torch.cuda.device_count(),
        "device name": torch.cuda.get_device_name(0),
        "memory (GB)": round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
        ),
    }


def document_gpus(target_dir: Path) -> None:
    """
    Document the current GPU resources to a `requirements.txt` file
    inside the given `target_dir`.
    """
    cuda_info = get_cuda_info()
    # Write the environment to a requirements file
    with open(target_dir / "info_gpus.txt", "w") as file:
        file.write(f"# CUDA information:\n")
        for c_info in cuda_info:
            file.write(f"{c_info}\n")


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
