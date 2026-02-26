"""
This module contains the abstract base class for representing posterior models,
as well as functions for training and testing across an epoch.
"""

import ctypes
import json
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from multiprocessing import Value
from collections.abc import Sized
from os.path import join
from typing import Optional, Tuple, Union

import h5py
import numpy as np
import torch
import torch.distributed as dist
from threadpoolctl import threadpool_limits
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset

import dingo.core.utils as utils
import dingo.core.utils.trainutils
from dingo.core.utils.backward_compatibility import update_model_config
from dingo.core.utils.misc import get_version
from dingo.core.utils.trainutils import EarlyStopping, RuntimeLimits


class BasePosteriorModel(ABC):
    """
    Abstract base class for PosteriorModels. This is intended to construct and hold a
    neural network for estimating the posterior density, as well as saving / loading,
    and training.

    Subclasses must implement methods for constructing the specific network, sampling,
    density evaluation, and computing the loss during training.
    """

    def __init__(
        self,
        model_filename: str = None,
        metadata: dict = None,
        initial_weights: dict = None,
        device: str = "cuda",
        load_training_info: bool = True,
    ):
        """
        Initialize a model for the posterior distribution.

        Parameters
        ----------
        model_filename: str
            If given, loads data from the given file.
        metadata: dict
            If given, initializes the model from these settings
        initial_weights: dict
            Initial weights for the model
        device: str
        load_training_info: bool
        """

        self.version = f"dingo={get_version()}"  # dingo version

        self.device = None
        self.rank = None
        self.optimizer_kwargs = None
        self.network_kwargs = None
        self.scheduler_kwargs = None
        self.initial_weights = initial_weights

        self.metadata = metadata
        if self.metadata is not None:
            self.model_kwargs = self.metadata["train_settings"]["model"]
            # Expect self.optimizer_settings and self.scheduler_settings to be set
            # separately, and before calling initialize_optimizer_and_scheduler().

        self.epoch = 0
        self.iteration = 0
        self.network = None
        self.optimizer = None
        self.scheduler = None
        self.context = None
        self.event_metadata = None

        # build model
        if model_filename is not None:
            self.load_model(
                model_filename, load_training_info=load_training_info, device=device
            )
        else:
            self.initialize_network()
            self.network_to_device(device)

    @abstractmethod
    def initialize_network(self):
        """
        Initialize the network backbone for the posterior model.
        """
        pass

    @abstractmethod
    def sample(self, *context: torch.Tensor, num_samples: int = 1):
        """
        Sample parameters theta from the posterior model,

        theta ~ p(theta | context)

        Parameters
        ----------
        context: torch.Tensor
            Context information (typically observed data). Should have a batch
            dimension (even if size B = 1).
        num_samples: int = 1
            Number of samples to generate.

        Returns
        -------
        samples: torch.Tensor
            Shape (B, num_samples, dim(theta))
        """
        pass

    @abstractmethod
    def sample_and_log_prob(self, *context: torch.Tensor, num_samples: int = 1):
        """
        Sample parameters theta from the posterior model,

        theta ~ p(theta | context)

        and also return the log_prob. For models such as normalizing flows, it is more
        economical to calculate the log_prob at the same time as sampling, rather than
        as a separate step.

        Parameters
        ----------
        context: torch.Tensor
            Context information (typically observed data). Should have a batch
            dimension (even if size B = 1).
        num_samples: int = 1
            Number of samples to generate.

        Returns
        -------
        samples, log_prob: torch.Tensor, torch.Tensor
            Shapes (B, num_samples, dim(theta)), (B, num_samples)
        """
        pass

    @abstractmethod
    def log_prob(self, theta: torch.Tensor, *context: torch.Tensor):
        """
        Evaluate the log posterior density,

        log p(theta | context)

        Parameters
        ----------
        theta: torch.Tensor
            Parameter values at which to evaluate the density. Should have a batch
            dimension (even if size B = 1).
        context: torch.Tensor
            Context information (typically observed data). Must have context.shape[0] = B.

        Returns
        -------
        log_prob: torch.Tensor
            Shape (B,)
        """
        pass

    @abstractmethod
    def loss(self, theta: torch.Tensor, *context: torch.Tensor):
        """
        Compute the loss for a batch of data.

        Parameters
        ----------
        theta: torch.Tensor
            Parameter values at which to evaluate the density. Should have a batch
            dimension (even if size B = 1).
        context: torch.Tensor
            Context information (typically observed data). Must have the same leading
            (batch) dimension as theta.

        Returns
        -------
        loss: torch.Tensor
            Mean loss across the batch (a scalar).
        """
        pass

    def network_to_device(self, device: str) -> None:
        """
        Put model to device and set ``self.device`` accordingly.

        Accepts plain device strings (``"cpu"``, ``"cuda"``) as well as
        rank-qualified CUDA strings (``"cuda:0"``, ``"cuda:1"``, …).  In the
        latter case ``self.rank`` is set to the integer rank index.
        """
        if "cpu" not in device and "cuda" not in device:
            raise ValueError(f"Device should contain 'cpu' or 'cuda', got {device}.")
        if ":" in device:
            self.rank = int(device.split(":")[1])
        self.device = torch.device(device)
        print(f"Putting posterior model to device {self.device}.")
        self.network.to(self.device)

    def initialize_optimizer_and_scheduler(self):
        """
        Initializes the optimizer and scheduler with self.optimizer_kwargs
        and self.scheduler_kwargs, respectively.
        """
        if self.optimizer_kwargs is not None:
            self.optimizer = utils.get_optimizer_from_kwargs(
                self.network.parameters(), **self.optimizer_kwargs
            )
        if self.scheduler_kwargs is not None:
            self.scheduler = utils.get_scheduler_from_kwargs(
                self.optimizer, **self.scheduler_kwargs
            )

    def save_model(
        self,
        model_filename: str,
        save_training_info: bool = True,
    ):
        """
        Save the posterior model to the disk.

        Parameters
        ----------
        model_filename: str
            filename for saving the model
        save_training_info: bool
            specifies whether information required to proceed with training is
            saved, e.g. optimizer state dict

        """
        # Strip the DDP wrapper so the checkpoint can be loaded on any number of GPUs.
        if isinstance(self.network, DDP):
            model_state_dict = self.network.module.state_dict()
        else:
            model_state_dict = self.network.state_dict()

        model_dict = {
            "model_kwargs": self.model_kwargs,
            "model_state_dict": model_state_dict,
            "epoch": self.epoch,
            "iteration": self.iteration,
            "version": self.version,
        }

        if self.metadata is not None:
            model_dict["metadata"] = self.metadata

        if self.context is not None:
            model_dict["context"] = self.context

        if self.event_metadata is not None:
            model_dict["event_metadata"] = self.event_metadata

        if save_training_info:
            model_dict["optimizer_kwargs"] = self.optimizer_kwargs
            model_dict["scheduler_kwargs"] = self.scheduler_kwargs
            if self.optimizer is not None:
                model_dict["optimizer_state_dict"] = self.optimizer.state_dict()
            if self.scheduler is not None:
                model_dict["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(model_dict, model_filename)

    def _load_model_from_hdf5(self, model_filename: str):
        """
        Helper function to load a trained model that has been
        saved in HDF5 format using `dingo_pt_to_hdf5`.
        Parameters
        ----------
        model_filename: str
            path to saved model; must have extension '.hdf5'
        Returns
        -------
        d: dict
            A stripped down version of the dict saved by torch.save()
            Specifically, it does not include 'optimizer_state_dict'
            to save space at inference time.
        """
        d = {}
        with h5py.File(model_filename, "r") as fp:
            model_basename = os.path.basename(model_filename)
            if fp.attrs["CANONICAL_FILE_BASENAME"] != model_basename:
                raise ValueError(
                    "HDF5 attribute CANONICAL_FILE_BASENAME differs from model name",
                    model_basename,
                )

            # Load small nested dicts from json
            for k, v in fp["serialized_dicts"].items():
                d[k] = json.loads(v[()])

            # Load model weights
            model_state_dict = OrderedDict()
            for k, v in fp["model_weights"].items():
                model_state_dict[k] = torch.from_numpy(np.array(v, dtype=np.float32))
            d["model_state_dict"] = model_state_dict

        return d

    def load_model(
        self,
        model_filename: str,
        load_training_info: bool = True,
        device: str = "cuda",
    ):
        """
        Load a posterior model from the disk.

        Parameters
        ----------
        model_filename: str
            path to saved model
        load_training_info: bool #TODO: load information for training
            specifies whether information required to proceed with training is
            loaded, e.g. optimizer state dict
        device: str
        """

        # Make sure that when the model is loaded, the torch tensors are put on the
        # device indicated in the saved metadata. External routines run on a cpu
        # machine may have moved the model from 'cuda' to 'cpu'.
        ext = os.path.splitext(model_filename)[-1]
        if ext == ".pt":
            d = torch.load(model_filename, map_location=device)
        elif ext == ".hdf5":
            d = self._load_model_from_hdf5(model_filename)
        else:
            raise ValueError("Models should be ether in .pt or .hdf5 format.")

        self.version = d.get("version")

        self.model_kwargs = d["model_kwargs"]
        update_model_config(self.model_kwargs)  # For backward compatibility

        self.epoch = d["epoch"]
        self.iteration = d.get("iteration", 0)

        self.metadata = d["metadata"]

        if "context" in d:
            self.context = d["context"]

        if "event_metadata" in d:
            self.event_metadata = d["event_metadata"]

        if device != "meta":
            self.initialize_network()
            self.network.load_state_dict(d["model_state_dict"])

            self.network_to_device(device)

            if load_training_info:
                if "optimizer_kwargs" in d:
                    self.optimizer_kwargs = d["optimizer_kwargs"]
                if "scheduler_kwargs" in d:
                    self.scheduler_kwargs = d["scheduler_kwargs"]
                # initialize optimizer and scheduler
                self.initialize_optimizer_and_scheduler()
                # load optimizer and scheduler state dict
                if "optimizer_state_dict" in d:
                    self.optimizer.load_state_dict(d["optimizer_state_dict"])
                if "scheduler_state_dict" in d:
                    self.scheduler.load_state_dict(d["scheduler_state_dict"])
            else:
                # put model in evaluation mode
                self.network.eval()

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        train_dir: str,
        train_sampler: Optional[torch.utils.data.DistributedSampler] = None,
        runtime_limits: Optional[RuntimeLimits] = None,
        checkpoint_epochs: Optional[int] = None,
        use_wandb: bool = False,
        test_only: bool = False,
        early_stopping: Optional[EarlyStopping] = None,
        gradient_updates_per_optimizer_step: int = 1,
        automatic_mixed_precision: bool = False,
        world_size: int = 1,
        global_epoch: ctypes.c_int = Value(ctypes.c_int, 1),
    ):
        """
        Train the network for one or more epochs.

        Parameters
        ----------
        train_loader : DataLoader
        test_loader : DataLoader
        train_dir : str
            Directory for saving models and history.
        train_sampler : DistributedSampler, optional
            Required for DDP training to re-shuffle data each epoch.
        runtime_limits : RuntimeLimits, optional
        checkpoint_epochs : int, optional
            Save a named checkpoint every this many epochs.
        use_wandb : bool
        test_only : bool
            If True, skip training and only evaluate on the test set.
        early_stopping : EarlyStopping, optional
        gradient_updates_per_optimizer_step : int
            Accumulate gradients over this many batches before calling
            ``optimizer.step()``.  Use >1 to simulate a larger effective batch
            size without increasing per-GPU memory usage.
        automatic_mixed_precision : bool
            Train with ``torch.amp`` mixed precision (FP16/BF16 forward pass,
            FP32 parameter updates).
        world_size : int
            Number of GPUs (used only for logging the effective batch size).
        global_epoch : multiprocessing.Value
            Shared counter updated so that the WaveformDataset can query the
            current epoch from any worker process.
        """
        is_primary = self.rank is None or self.rank == 0

        if test_only:
            test_loss = test_epoch(self, dataloader=test_loader)
            if is_primary:
                print(f"test loss: {test_loss:.3f}")
            return

        while not runtime_limits.limits_exceeded(self.epoch):
            self.epoch += 1
            global_epoch.value = self.epoch

            lr = utils.get_lr(self.optimizer)
            with threadpool_limits(limits=1, user_api="blas"):
                if train_sampler is not None:
                    # Ensure each epoch sees a different random shuffle.
                    train_sampler.set_epoch(self.epoch)

                if is_primary:
                    print(f"\nStart training epoch {self.epoch} with lr {lr}")
                time_start = torch.tensor(time.time(), device=self.device)

                train_loss, n_iter = train_epoch(
                    self,
                    dataloader=train_loader,
                    gradient_updates_per_optimizer_step=gradient_updates_per_optimizer_step,
                    automatic_mixed_precision=automatic_mixed_precision,
                    world_size=world_size,
                )
                self.iteration += n_iter

                if self.rank is not None:
                    dist.barrier()
                    dist.all_reduce(time_start, op=dist.ReduceOp.MIN)
                train_time = time.time() - time_start.item()

                if is_primary:
                    print(
                        "Done. This took {:2.0f}:{:2.0f} min.".format(
                            *divmod(train_time, 60)
                        )
                    )
                    print(f"Start testing epoch {self.epoch}")

                time_start = torch.tensor(time.time(), device=self.device)
                test_loss = test_epoch(
                    self,
                    dataloader=test_loader,
                    gradient_updates_per_optimizer_step=gradient_updates_per_optimizer_step,
                    world_size=world_size,
                )
                if self.rank is not None:
                    dist.barrier()
                    dist.all_reduce(time_start, op=dist.ReduceOp.MIN)
                test_time = time.time() - time_start.item()

                if is_primary:
                    print(
                        "Done. This took {:2.0f}:{:2.0f} min.".format(
                            *divmod(test_time, 60)
                        )
                    )

            utils.perform_scheduler_step(self.scheduler, test_loss)

            if is_primary:
                utils.write_history(train_dir, self.epoch, train_loss, test_loss, lr)
                utils.save_model(self, train_dir, checkpoint_epochs=checkpoint_epochs)
                if use_wandb:
                    try:
                        import wandb

                        wandb.define_metric("epoch")
                        wandb.define_metric("*", step_metric="epoch")
                        wandb.log(
                            {
                                "epoch": self.epoch,
                                "learning_rate": lr[0],
                                "train_loss": train_loss,
                                "test_loss": test_loss,
                                "train_time": train_time,
                                "test_time": test_time,
                            }
                        )
                    except ImportError:
                        print("wandb not installed. Skipping logging to wandb.")

            if early_stopping is not None:
                early_stopping_loss = (
                    test_loss if early_stopping.metric == "validation" else train_loss
                )
                is_best_model = early_stopping(early_stopping_loss)
                if is_best_model and is_primary:
                    self.save_model(
                        join(train_dir, "best_model.pt"), save_training_info=False
                    )
                if early_stopping.early_stop:
                    if is_primary:
                        print("Early stopping")
                    break

            if is_primary:
                print(f"Finished training epoch {self.epoch}.\n")


def _dataset_len(dataloader: torch.utils.data.DataLoader) -> int:
    """Return the number of samples in a DataLoader's dataset.

    ``DataLoader.dataset`` is typed as ``Dataset``, which does not inherit from
    ``Sized`` in PyTorch's stubs even though all concrete datasets implement
    ``__len__``.  This helper asserts the contract and returns the length.
    """
    dataset = dataloader.dataset
    if not isinstance(dataset, Sized):
        raise TypeError(
            f"DataLoader dataset of type {type(dataset).__name__} does not "
            "implement __len__. Cannot determine dataset length."
        )
    return len(dataset)


def train_epoch(
    pm: "BasePosteriorModel",
    dataloader: torch.utils.data.DataLoader,
    gradient_updates_per_optimizer_step: int = 1,
    automatic_mixed_precision: bool = False,
    world_size: int = 1,
) -> Tuple[float, int]:
    """
    Train the network for one epoch.

    Parameters
    ----------
    pm : BasePosteriorModel
    dataloader : DataLoader
    gradient_updates_per_optimizer_step : int
        Accumulate gradients over this many mini-batches before stepping.
        Values >1 simulate a larger effective batch without extra GPU memory.
    automatic_mixed_precision : bool
        Use ``torch.amp`` (FP16 forward pass, FP32 updates).
    world_size : int
        Number of GPUs, used only for logging the effective batch size.

    Returns
    -------
    Tuple[float, int]
        Average loss over the epoch and number of optimizer steps performed.
    """
    pm.network.train()

    if pm.rank is None:
        effective_bs = dataloader.batch_size
    else:
        effective_bs = dataloader.batch_size * world_size

    loss_info = dingo.core.utils.trainutils.LossInfo(
        epoch=pm.epoch,
        len_dataset=_dataset_len(dataloader),
        batch_size_per_grad_update=effective_bs,
        mode="Train",
        print_freq=1,
        device=pm.device,
    )

    scaler = GradScaler("cuda") if automatic_mixed_precision else None

    for batch_idx, data in enumerate(dataloader):
        loss_info.update_timer("Dataloader")

        if batch_idx % gradient_updates_per_optimizer_step == 0:
            pm.optimizer.zero_grad(set_to_none=True)

        data = [d.to(pm.device, non_blocking=True) for d in data]

        if automatic_mixed_precision:
            with autocast("cuda"):
                result = pm.loss(data[0], *data[1:])
            loss = result[0] if isinstance(result, tuple) else result
            scaler.scale(loss).backward()
        else:
            result = pm.loss(data[0], *data[1:])
            loss = result[0] if isinstance(result, tuple) else result
            loss.backward()

        loss_info.cache_loss(loss=loss, n=len(data[0]))

        if (batch_idx + 1) % gradient_updates_per_optimizer_step == 0:
            if automatic_mixed_precision:
                scaler.step(pm.optimizer)
                scaler.update()
            else:
                pm.optimizer.step()

            loss_info.update()
            if pm.rank is None or pm.rank == 0:
                loss_info.print_info(batch_idx)

    return loss_info.get_avg(), loss_info.get_iteration()


def test_epoch(
    pm: "BasePosteriorModel",
    dataloader: torch.utils.data.DataLoader,
    gradient_updates_per_optimizer_step: int = 1,
    world_size: int = 1,
) -> float:
    """
    Evaluate the network on the test set.

    Parameters
    ----------
    pm : BasePosteriorModel
    dataloader : DataLoader
    gradient_updates_per_optimizer_step : int
        Used to match the effective batch size of the training loop for
        comparable loss values.
    world_size : int
        Number of GPUs, used only for logging.

    Returns
    -------
    float
        Average loss over the test set.
    """
    with torch.no_grad():
        pm.network.eval()

        if pm.rank is None:
            effective_bs = dataloader.batch_size
        else:
            effective_bs = dataloader.batch_size * world_size

        if _dataset_len(dataloader) < effective_bs:
            if pm.rank is None or pm.rank == 0:
                print(
                    f"Warning: test dataset (len {_dataset_len(dataloader)}) is smaller "
                    f"than effective_batch_size={effective_bs}. "
                    "Test loss computed over full dataset; may not be comparable to train loss."
                )
            effective_bs = _dataset_len(dataloader)
            gradient_updates_per_optimizer_step = 1

        loss_info = dingo.core.utils.trainutils.LossInfo(
            epoch=pm.epoch,
            len_dataset=_dataset_len(dataloader),
            batch_size_per_grad_update=effective_bs,
            mode="Test",
            print_freq=1,
            device=pm.device,
        )

        for batch_idx, data in enumerate(dataloader):
            loss_info.update_timer()
            data = [d.to(pm.device, non_blocking=True) for d in data]
            result = pm.loss(data[0], *data[1:])
            loss = result[0] if isinstance(result, tuple) else result
            loss_info.cache_loss(loss, len(data[0]))
            if (batch_idx + 1) % gradient_updates_per_optimizer_step == 0:
                loss_info.update()
                if pm.rank is None or pm.rank == 0:
                    loss_info.print_info(batch_idx)

        return loss_info.get_avg()
