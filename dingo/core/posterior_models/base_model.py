"""
This module contains the abstract base class for representing posterior models,
as well as functions for training and testing across an epoch.
"""

import ctypes
import h5py
import json
import numpy as np
import os
import sys
import time

from abc import abstractmethod, ABC
from collections import OrderedDict
from multiprocessing import Value
from os.path import join
from threadpoolctl import threadpool_limits
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast

import dingo.core.utils.trainutils
import dingo.core.utils as utils

from dingo.core.utils.backward_compatibility import update_model_config
from dingo.core.utils.misc import get_version

from dingo.core.utils.scheduler import get_scheduler_from_kwargs, perform_scheduler_step
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
        embedding_net_builder: Callable = None,
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
        embedding_net_builder: Callable
            If given, builds embedding network using this function
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
        self.embedding_net_builder = embedding_net_builder
        self.initial_weights = initial_weights

        self.metadata = metadata
        if self.metadata is not None:
            self.model_kwargs = self.metadata["train_settings"]["model"]
            # Expect self.optimizer_settings and self.scheduler_settings to be set
            # separately, and before calling initialize_optimizer_and_scheduler().

        self.epoch = 0
        # iteration = number of optimizer steps
        self.iteration = 0
        self.logging_info = {}
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

    def network_to_device(self, device):
        """
        Put model to device, and set self.device accordingly.
        """
        if "cpu" not in device and "cuda" not in device:
            raise ValueError(f"Device should contain either cpu or cuda, got {device}.")
        if ":" in device:
            self.rank = int(device.split(":")[1])
        self.device = torch.device(device)
        print(f"Putting posterior model to device {self.device}.")
        self.network.to(self.device)

    def initialize_optimizer_and_scheduler(
        self, num_optimizer_steps: Optional[int] = None
    ):
        """
        Initializes the optimizer and scheduler with self.optimizer_kwargs
        and self.scheduler_kwargs, respectively.
        """
        if self.optimizer_kwargs is not None:
            self.optimizer = utils.get_optimizer_from_kwargs(
                self.network.parameters(), **self.optimizer_kwargs
            )
        if self.scheduler_kwargs is not None:
            # Number of optimizer steps per epoch required if scheduler updates are performed per optimizer step
            if num_optimizer_steps is not None:
                self.scheduler_kwargs["num_optimizer_steps_per_epoch"] = (
                    num_optimizer_steps
                )

            self.scheduler = get_scheduler_from_kwargs(
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
        model_dict = {
            "model_kwargs": self.model_kwargs,
            "epoch": self.epoch,
            "iteration": self.iteration,
            "version": self.version,
        }

        # Remove DDP wrapper
        if any("module." in key for key in self.network.state_dict()):
            # Remove "module." prefix from the state_dict keys
            model_state_dict = {
                k.replace("module.", ""): v
                for k, v in self.network.state_dict().items()
            }
            model_dict["model_state_dict"] = model_state_dict
        else:
            model_dict["model_state_dict"] = self.network.state_dict()

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
            raise ValueError("Models should be either in .pt or .hdf5 format.")

        self.version = d.get("version")

        self.model_kwargs = d["model_kwargs"]
        update_model_config(self.model_kwargs)  # For backward compatibility

        self.epoch = d["epoch"]
        self.iteration = d.get("iteration", 0)
        self.logging_info = d.get("logging_info", {})

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
        train_sampler: torch.utils.data.DistributedSampler,
        train_dir: str,
        runtime_limits: RuntimeLimits = None,
        checkpoint_epochs: int = None,
        use_wandb: bool = False,
        test_only: bool = False,
        early_stopping: Optional[EarlyStopping] = None,
        gradient_updates_per_optimizer_step: int = 1,
        automatic_mixed_precision: bool = False,
        world_size: int = 1,
        global_epoch: ctypes.c_int = Value(ctypes.c_int, 1),
    ):
        """

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            torch data loader with training data
        test_loader: torch.utils.data.DataLoader
            torch data loader with test data
        train_sampler: torch.distributed.DistributedSampler
            torch distributed sampler for training data
        train_dir: str
            directory for saving models and history
        runtime_limits: RuntimeLimits = None
            allows to check whether the runtime limit is exceeded
        checkpoint_epochs: int=None
            number of epochs between checkpoints
        use_wandb: bool=False
            whether to use wand
        test_only: bool = False
            if True, training is skipped
        early_stopping: EarlyStopping
            Optional EarlyStopping instance.
        gradient_updates_per_optimizer_step: int
            number of gradient updates to perform for every optimizer step. Useful to simulate multi-GPU training on
            a single GPU by choosing n gradient updates for n GPUs.
        automatic_mixed_precision: bool
            whether to train with automatic mixed precision. Warning: Implementing gradient clipping requires additional
            modifications (https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html).
        world_size: int=1
            the number of GPUs used for training, value is used to adjust the batch_size when printing updates.

        Returns
        -------

        """

        if test_only:
            test_loss = test_epoch(self, dataloader=test_loader)
            # Only print for one device
            if self.rank is None or self.rank == 0:
                print(f"test loss: {test_loss:.3f}")

        else:

            while not runtime_limits.limits_exceeded(self.epoch):
                self.epoch += 1

                # Training
                lr = utils.get_lr(self.optimizer)
                with threadpool_limits(limits=1, user_api="blas"):
                    if train_sampler is not None:
                        # Ensure that data is shuffled every epoch in multi-GPU training
                        train_sampler.set_epoch(self.epoch)
                    # Set epoch
                    global_epoch.value = self.epoch
                    # Only print for one device
                    if self.rank is None or self.rank == 0:
                        print(f"\nStart training epoch {self.epoch} with lr {lr}")
                    time_start = time.time()
                    train_loss, iteration, logging_info = train_epoch(
                        self,
                        dataloader=train_loader,
                        gradient_updates_per_optimizer_step=gradient_updates_per_optimizer_step,
                        automatic_mixed_precision=automatic_mixed_precision,
                        world_size=world_size,
                    )
                    train_time = torch.tensor(
                        time.time() - time_start, device=self.device
                    )
                    self.iteration += iteration
                    for k, v in logging_info.items():
                        self.logging_info[k] = v
                    if "num_tokens" in logging_info.keys():
                        # Add cumulatively tracked values
                        for k in ["num_tokens", "num_all_tokens"]:
                            if f"{k}_cumulative" not in self.logging_info.keys():
                                self.logging_info[f"{k}_cumulative"] = logging_info[k]
                            else:
                                self.logging_info[f"{k}_cumulative"] += logging_info[k]
                    if self.rank is not None:
                        # Sync all processes before aggregating value
                        dist.barrier()
                        # Aggregate maximal time
                        dist.reduce(train_time, dst=0, op=dist.ReduceOp.MAX)

                    # Only print for one device
                    if self.rank is None or self.rank == 0:
                        print(
                            "Done. This took {:2.0f}:{:2.0f} min.".format(
                                *divmod(train_time.detach().item(), 60)
                            )
                        )

                        # Testing
                        print(f"Start testing epoch {self.epoch}")
                    time_start = time.time()
                    test_loss = test_epoch(
                        self,
                        dataloader=test_loader,
                        gradient_updates_per_optimizer_step=gradient_updates_per_optimizer_step,
                        world_size=world_size,
                    )
                    test_time = torch.tensor(
                        time.time() - time_start, device=self.device
                    )

                    if self.rank is not None:
                        # Sync all processes before aggregating value
                        dist.barrier()
                        # Aggregate values
                        dist.reduce(test_time, dst=0, op=dist.ReduceOp.MAX)

                    # Only print for one device
                    if self.rank is None or self.rank == 0:
                        print(
                            "Done. This took {:2.0f}:{:2.0f} min.".format(
                                *divmod(test_time.detach().item(), 60)
                            )
                        )
                # Update scheduler if update_every_optimizer_step == False
                perform_scheduler_step(
                    self.scheduler,
                    loss=test_loss,
                    scheduler_kwargs=self.scheduler_kwargs,
                    update_level="epoch",
                )

                if self.rank is None or self.rank == 0:
                    # write history and save model
                    aux = None
                    if "num_tokens" in logging_info.keys():
                        aux = [
                            self.logging_info["num_tokens"],
                            self.logging_info["num_all_tokens"],
                        ]
                    utils.write_history(
                        log_dir=train_dir,
                        epoch=self.epoch,
                        train_loss=train_loss,
                        test_loss=test_loss,
                        learning_rates=lr,
                        aux=aux,
                    )
                    utils.save_model(
                        self, train_dir, checkpoint_epochs=checkpoint_epochs
                    )
                    if use_wandb:
                        try:
                            import wandb

                            wandb.define_metric("epoch")
                            wandb.define_metric("iteration")
                            if "num_tokens" in logging_info.keys():
                                wandb.define_metric("num_tokens")
                            if world_size is None or world_size == 1:
                                wandb.define_metric("*", step_metric="epoch")
                            else:
                                wandb.define_metric("*", step_metric="iteration")
                            wandb_log_info = {
                                "epoch": self.epoch,
                                "iteration": self.iteration,
                                "learning_rate": lr[0],
                                "train_loss": train_loss,
                                "test_loss": test_loss,
                                "train_time": train_time,
                                "test_time": test_time,
                            }
                            for k, v in self.logging_info.items():
                                wandb_log_info[k] = v
                            wandb.log(wandb_log_info)
                        except ImportError:
                            print("wandb not installed. Skipping logging to wandb.")

                if early_stopping is not None:
                    # Whether to use train or test loss
                    early_stopping_loss = (
                        test_loss
                        if early_stopping.metric == "validation"
                        else train_loss
                    )
                    is_best_model = early_stopping(early_stopping_loss)
                    if is_best_model and (self.rank is None or self.rank == 0):
                        self.save_model(
                            join(train_dir, "best_model.pt"), save_training_info=False
                        )
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break
                print(f"Finished training epoch {self.epoch}.\n")
                # Force flush of print statements to info.out
                sys.stdout.flush()


def train_epoch(
    pm,
    dataloader,
    gradient_updates_per_optimizer_step: int = 1,
    automatic_mixed_precision: bool = False,
    world_size: int = 1,
):
    pm.network.train()
    # Compute effective batch size
    if pm.rank is None:
        effective_batch_size_per_grad_update = dataloader.batch_size
    else:
        effective_batch_size_per_grad_update = dataloader.batch_size * world_size
    loss_info = dingo.core.utils.trainutils.LossInfo(
        epoch=pm.epoch,
        len_dataset=len(dataloader.dataset),
        batch_size_per_grad_update=effective_batch_size_per_grad_update,
        mode="Train",
        print_freq=1,
        device=pm.device,
    )
    scaler = None
    if automatic_mixed_precision:
        # Create scaler for automatic mixed precision (amp)
        # Warning: gradient clipping requires special treatment in amp
        scaler = GradScaler()

    for batch_idx, data in enumerate(dataloader):
        loss_info.update_timer("Dataloader")
        if batch_idx % gradient_updates_per_optimizer_step == 0:
            pm.optimizer.zero_grad(set_to_none=True)
        # Data to device
        data = [d.to(pm.device, non_blocking=True) for d in data]
        if automatic_mixed_precision:
            with autocast():
                # Compute loss
                loss, logging_info = pm.loss(data[0], *data[1:])
            # Backward pass, Note: Backward passes under autocast are not recommended
            # Scales loss before calling backward()
            scaler.scale(loss).backward()
        else:
            # Compute loss
            loss, logging_info = pm.loss(data[0], *data[1:])
            # Backward pass
            loss.backward()
        # Cache loss
        loss_info.cache_loss(loss=loss, n=len(data[0]), logging_info=logging_info)

        # Optimizer step
        if (batch_idx + 1) % gradient_updates_per_optimizer_step == 0:
            if automatic_mixed_precision:
                # Take a step with the optimizer
                # Warning: Optimizer.step() is skipped if the unscaled gradients of the optimizer parameters contain
                # inf or Nan
                scaler.step(pm.optimizer)
                scaler.update()
            else:
                pm.optimizer.step()

            # Update loss for history and logging
            loss_info.update()
            if pm.rank is None or pm.rank == 0:
                loss_info.print_info(batch_idx)

            # Update currently active scheduler if update_every_optimizer_step == True
            perform_scheduler_step(
                pm.scheduler,
                scheduler_kwargs=pm.scheduler_kwargs,
                update_level="optimizer_step",
            )

    return loss_info.get_avg(), loss_info.get_iteration(), loss_info.get_logging_info()


def test_epoch(
    pm, dataloader, gradient_updates_per_optimizer_step: int = 1, world_size: int = 1
):
    with torch.no_grad():
        pm.network.eval()
        # Compute effective batch size
        if pm.rank is None:
            effective_batch_size_per_grad_update = dataloader.batch_size
        else:
            effective_batch_size_per_grad_update = dataloader.batch_size * world_size
        if len(dataloader.dataset) < effective_batch_size_per_grad_update:
            if pm.rank is None or pm.rank == 0:
                print(
                    f"Warning: Test dataset (len {len(dataloader.dataset)}) smaller than "
                    f"effective_batch_size_per_grad_update={effective_batch_size_per_grad_update}. "
                    f"Test loss computed over full test dataset, might not be comparable to train loss. "
                )
                effective_batch_size_per_grad_update = len(dataloader.dataset)
                gradient_updates_per_optimizer_step = 1
        loss_info = dingo.core.utils.trainutils.LossInfo(
            epoch=pm.epoch,
            len_dataset=len(dataloader.dataset),
            batch_size_per_grad_update=effective_batch_size_per_grad_update,
            mode="Test",
            print_freq=1,
            device=pm.device,
        )

        for batch_idx, data in enumerate(dataloader):
            loss_info.update_timer()
            # Data to device
            data = [d.to(pm.device, non_blocking=True) for d in data]
            # Compute loss
            loss, _ = pm.loss(data[0], *data[1:])
            # Update loss for history and logging
            loss_info.cache_loss(loss, len(data[0]))
            # Average loss over multiple batches, equivalent to training
            if (batch_idx + 1) % gradient_updates_per_optimizer_step == 0:
                loss_info.update()
                if pm.rank is None or pm.rank == 0:
                    loss_info.print_info(batch_idx)

        return loss_info.get_avg()
