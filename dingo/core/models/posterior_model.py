"""
TODO: Docstring
"""

from typing import Callable
import torch
import dingo.core.utils as utils
from torch.utils.data import Dataset
import os
import time
import numpy as np
from threadpoolctl import threadpool_limits
import dingo.core.utils.trainutils
import math
import h5py
import json
from collections import OrderedDict

from dingo.core.nn.nsf import (
    create_nsf_with_rb_projection_embedding_net,
    create_nsf_wrapped,
)
from dingo.core.utils.misc import get_version


class PosteriorModel:
    """
    TODO: Docstring

    Methods
    -------

    initialize_model:
        initialize the NDE (including embedding net) as posterior model
    initialize_training:
        initialize for training, that includes storing the epoch, building
        an optimizer and a learning rate scheduler
    save_model:
        save the model, including all information required to rebuild it,
        except for the builder function
    load_model:
        load and build a model from a file
    train_model:
        train the model
    inference:
        perform inference
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

        Parameters
        ----------

        model_builder: Callable
            builder function for the model,
            self.model = model_builder(**model_kwargs)
        model_kwargs: dict = None
            kwargs for for the model,
            self.model = model_builder(**model_kwargs)
        model_filename: str = None
            path to filename of loaded model
        optimizer_kwargs: dict = None
            kwargs for optimizer
        scheduler_kwargs: dict = None
            kwargs for scheduler
        init_for_training: bool = False
            flag whether initialization for training (e.g., optimizer) required
        metadata: dict = None
            dict with metadata, used to save dataset_settings and train_settings
        """
        self.version = f"dingo={get_version()}"  # dingo version

        self.optimizer_kwargs = None
        self.model_kwargs = None
        self.scheduler_kwargs = None
        self.initial_weights = initial_weights

        self.metadata = metadata
        if self.metadata is not None:
            self.model_kwargs = self.metadata["train_settings"]["model"]
            # Expect self.optimizer_settings and self.scheduler_settings to be set
            # separately, and before calling initialize_optimizer_and_scheduler().

        self.epoch = 0
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
            self.initialize_model()
            self.model_to_device(device)

    def model_to_device(self, device):
        """
        Put model to device, and set self.device accordingly.
        """
        if device not in ("cpu", "cuda"):
            raise ValueError(f"Device should be either cpu or cuda, got {device}.")
        self.device = torch.device(device)
        # Commented below so that code runs on first cuda device in the case of multiple.
        # if device == 'cuda' and torch.cuda.device_count() > 1:
        #     print("Using", torch.cuda.device_count(), "GPUs.")
        #     raise NotImplementedError('This needs testing!')
        #     # dim = 0 [512, ...] -> [256, ...], [256, ...] on 2 GPUs
        #     self.model = torch.nn.DataParallel(self.model)
        print(f"Putting posterior model to device {self.device}.")
        self.model.to(self.device)

    def initialize_model(self):
        """
        Initialize a model for the posterior by calling the
        self.model_builder with self.model_kwargs.

        """
        model_builder = get_model_callable(self.model_kwargs["type"])
        model_kwargs = {k: v for k, v in self.model_kwargs.items() if k != "type"}
        if self.initial_weights is not None:
            model_kwargs["initial_weights"] = self.initial_weights
        self.model = model_builder(**model_kwargs)

    def initialize_optimizer_and_scheduler(self):
        """
        Initializes the optimizer and scheduler with self.optimizer_kwargs
        and self.scheduler_kwargs, respectively.
        """
        if self.optimizer_kwargs is not None:
            self.optimizer = utils.get_optimizer_from_kwargs(
                self.model.parameters(), **self.optimizer_kwargs
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
        model_dict = {
            "model_kwargs": self.model_kwargs,
            "model_state_dict": self.model.state_dict(),
            "epoch": self.epoch,
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
            # TODO

        torch.save(model_dict, model_filename)


    def _load_model_from_hdf5(
        self,
        model_filename: str
    ):
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
        with h5py.File(model_filename, 'r') as fp:
            model_basename = os.path.basename(model_filename)
            if fp.attrs['CANONICAL_FILE_BASENAME'] != model_basename:
                raise ValueError('HDF5 attribute CANONICAL_FILE_BASENAME differs from model name',
                        model_basename)

            # Load small nested dicts from json
            for k, v in fp['serialized_dicts'].items():
                d[k] = json.loads(v[()])

            # Load model weights
            model_state_dict = OrderedDict()
            for k, v in fp['model_weights'].items():
                model_state_dict[k] = torch.from_numpy(np.array(v, dtype=np.float32))
            d['model_state_dict'] = model_state_dict

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
        """
        # Make sure that when the model is loaded, the torch tensors are put on the
        # device indicated in the saved metadata. External routines run on a cpu
        # machine may have moved the model from 'cuda' to 'cpu'.
        ext = os.path.splitext(model_filename)[-1]
        if ext == '.pt':
            d = torch.load(model_filename, map_location=device)
        elif ext == '.hdf5':
            d = self._load_model_from_hdf5(model_filename)
        else:
            raise ValueError('Models should be ether in .pt or .hdf5 format.')

        self.version = d.get("version")

        self.model_kwargs = d["model_kwargs"]
        self.initialize_model()
        self.model.load_state_dict(d["model_state_dict"])

        self.epoch = d["epoch"]

        self.metadata = d["metadata"]

        if "context" in d:
            self.context = d["context"]

        if "event_metadata" in d:
            self.event_metadata = d["event_metadata"]

        self.model_to_device(device)

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
            self.model.eval()

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        train_dir: str,
        runtime_limits: object = None,
        checkpoint_epochs: int = None,
        use_wandb=False,
        test_only=False,
    ):
        """

        Parameters
        ----------
        train_loader
        test_loader
        train_dir
        runtime_limits
        checkpoint_epochs
        use_wandb
        test_only: bool = False
            if True, training is skipped

        Returns
        -------

        """
        if test_only:
            test_loss = test_epoch(self, test_loader)
            print(f"test loss: {test_loss:.3f}")

        else:
            while not runtime_limits.limits_exceeded(self.epoch):
                self.epoch += 1

                # Training
                lr = utils.get_lr(self.optimizer)
                with threadpool_limits(limits=1, user_api="blas"):
                    print(f"\nStart training epoch {self.epoch} with lr {lr}")
                    time_start = time.time()
                    train_loss = train_epoch(self, train_loader)
                    train_time = time.time() - time_start

                    print(
                        "Done. This took {:2.0f}:{:2.0f} min.".format(
                            *divmod(train_time, 60)
                        )
                    )

                    # Testing
                    print(f"Start testing epoch {self.epoch}")
                    time_start = time.time()
                    test_loss = test_epoch(self, test_loader)
                    test_time = time.time() - time_start

                    print(
                        "Done. This took {:2.0f}:{:2.0f} min.".format(
                            *divmod(time.time() - time_start, 60)
                        )
                    )

                # scheduler step for learning rate
                utils.perform_scheduler_step(self.scheduler, test_loss)

                # write history and save model
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

                print(f"Finished training epoch {self.epoch}.\n")

    def sample(
        self,
        *x,
        batch_size=None,
        get_log_prob=False,
    ):
        """
        Sample from posterior model, conditioned on context x. x is expected to have a
        batch dimension, i.e., to obtain N samples with additional context requires
        x = x_.expand(N, *x_.shape).

        This method takes care of the batching, makes sure that self.model is in
        evaluation mode and disables gradient computation.

        Parameters
        ----------
        *x:
            input context to the neural network; has potentially multiple elements for,
            e.g., gnpe proxies
        batch_size: int = None
            batch size for sampling
        get_log_prob: bool = False
            if True, also return log probability along with the samples

        Returns
        -------
        samples: torch.Tensor
            samples from posterior model
        """
        self.model.eval()
        with torch.no_grad():
            if batch_size is None:
                samples = self.model.sample(*x)
                if get_log_prob:
                    log_prob = self.model.log_prob(samples, *x)
            else:
                samples = []
                if get_log_prob:
                    log_prob = []
                num_batches = math.ceil(len(x[0]) / batch_size)
                for idx_batch in range(num_batches):
                    lower, upper = idx_batch * batch_size, (idx_batch + 1) * batch_size
                    x_batch = [xi[lower:upper] for xi in x]
                    samples.append(self.model.sample(*x_batch, num_samples=1))
                    if get_log_prob:
                        log_prob.append(self.model.log_prob(samples[-1], *x_batch))
                samples = torch.cat(samples, dim=0)
                if get_log_prob:
                    log_prob = torch.cat(log_prob, dim=0)
        if not get_log_prob:
            return samples
        else:
            return samples, log_prob


def get_model_callable(model_type: str):
    if model_type == "nsf+embedding":
        return create_nsf_with_rb_projection_embedding_net
    elif model_type == "nsf":
        return create_nsf_wrapped
    else:
        raise KeyError("Invalid model type.")


def train_epoch(pm, dataloader):
    pm.model.train()
    loss_info = dingo.core.utils.trainutils.LossInfo(
        pm.epoch,
        len(dataloader.dataset),
        dataloader.batch_size,
        mode="Train",
        print_freq=1,
    )

    for batch_idx, data in enumerate(dataloader):
        loss_info.update_timer()
        pm.optimizer.zero_grad()
        # data to device
        data = [d.to(pm.device, non_blocking=True) for d in data]
        # compute loss
        loss = -pm.model(data[0], *data[1:]).mean()
        # backward pass and optimizer step
        loss.backward()
        pm.optimizer.step()
        # update loss for history and logging
        loss_info.update(loss.detach().item(), len(data[0]))
        loss_info.print_info(batch_idx)

    return loss_info.get_avg()


def test_epoch(pm, dataloader):
    with torch.no_grad():
        pm.model.eval()
        loss_info = dingo.core.utils.trainutils.LossInfo(
            pm.epoch,
            len(dataloader.dataset),
            dataloader.batch_size,
            mode="Test",
            print_freq=1,
        )

        for batch_idx, data in enumerate(dataloader):
            loss_info.update_timer()
            # data to device
            data = [d.to(pm.device, non_blocking=True) for d in data]
            # compute loss
            loss = -pm.model(data[0], *data[1:]).mean()
            # update loss for history and logging
            loss_info.update(loss.item(), len(data[0]))
            loss_info.print_info(batch_idx)

        return loss_info.get_avg()
