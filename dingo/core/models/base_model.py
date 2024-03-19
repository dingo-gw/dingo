from abc import abstractmethod
from collections import OrderedDict
import json
import os
import time
from threadpoolctl import threadpool_limits

import h5py
import numpy as np
import torch

import dingo
import dingo.core.utils as utils
from dingo.core.models.posterior_model import get_model_callable
from dingo.core.utils.misc import get_version


class BaseModel:
    """
    Generic model class that builds a model based on a Callable and handles model independent tasks such as initializing
    the model, the optimizer, and the scheduler, saving and loading the model, as well as putting it on a specific
    device. This base class can be used to define derived classes with more specific methods such as 'train', 'evaluate'
    or 'sample'.

    This class is based on posterior_model.py without train and sample.
    If this code is at one point included in the main dingo code, other model classes such as PosteriorModel should
    inherit from this class to minimize duplicate code.

    Methods
    -------

    initialize_model:
        initialize the embedding net and the additional networks used for pretraining
    initialize_optimizer_and_scheduler:
        initialize optimizer and learning rate scheduler for training
    model_to_device:
        put model on specific device
    load_model:
        load and build a model from a file
    save_model:
        save the model, including all information required to rebuild it,
        except for the builder function
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

        model_filename: str = None
            path to filename of loaded model
        metadata: dict = None
            dict with metadata, e.g. model_kwargs, scheduler_kwargs, etc.
            used to save dataset_settings and train_settings
        initial_weights: dict = None
            dict with initial weights of model
        device: str = cuda
            device for model
        load_training_info: bool = True
            whether to load the training info from the latest saved model
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

    def initialize_model(self):
        """
        Initialize a model by calling the
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
            device to load
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

    def save_model(
            self,
            model_filename: str,
            save_training_info: bool = True,
    ):
        """
        Save the model to the disk.

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

    @abstractmethod
    def loss(self, data, context):
        """
        Compute the loss for a batch of data.
        """
        pass

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
        loss = pm.loss(data[0], *data[1:])
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
            loss = pm.loss(data[0], *data[1:])
            # update loss for history and logging
            loss_info.update(loss.item(), len(data[0]))
            loss_info.print_info(batch_idx)

        return loss_info.get_avg()
