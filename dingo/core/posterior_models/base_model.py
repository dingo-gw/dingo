"""
TODO: Docstring
"""
from abc import abstractmethod
import os
from os.path import join
from typing import OrderedDict
import h5py

import torch
import dingo.core.utils as utils
from torch.utils.data import Dataset
import time
from threadpoolctl import threadpool_limits
import dingo.core.utils.trainutils
import math
from dingo.core.utils.misc import get_version
from dingo.core.utils.trainutils import EarlyStopping

class Base:
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
        Initialize the network backbone for the posterior model

        """
        pass

    @abstractmethod
    def sample_batch(self, *context_data):
        """
        Sample a batch of data from the posterior model.

        Parameters
        ----------
        context: Tensor
        Returns
        -------
        batch: dict
            dictionary with batch data
        """
        pass

    @abstractmethod
    def sample_and_log_prob_batch(self, *context_data):
        """
        Sample a batch of data and log probs from the posterior model.

        Parameters
        ----------
        context: Tensor
        Returns
        -------
        batch: dict
            dictionary with batch data
        """
        pass

    @abstractmethod
    def log_prob_batch(self, data, *context_data):
        """
        Sample a batch of data from the posterior model.

        Parameters
        ----------
        data: Tensor
        context: Tensor

        Returns
        -------
        batch: dict
            dictionary with batch data
        """
        pass

    @abstractmethod
    def loss(self, data, context):
        """
        Compute the loss for a batch of data.

        Parameters
        ----------
        data: Tensor
        context: Tensor

        Returns
        -------
        loss: Tensor
            loss for the batch
        """
        pass

    def network_to_device(self, device):
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
        #     self.network = torch.nn.DataParallel(self.network)
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
        model_dict = {
            "model_kwargs": self.model_kwargs,
            "model_state_dict": self.network.state_dict(),
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
        device: str
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
        self.initialize_network()
        self.network.load_state_dict(d["model_state_dict"])

        self.epoch = d["epoch"]

        self.metadata = d["metadata"]

        if "context" in d:
            self.context = d["context"]

        if "event_metadata" in d:
            self.event_metadata = d["event_metadata"]

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
        runtime_limits: object = None,
        checkpoint_epochs: int = None,
        use_wandb=False,
        test_only=False,
        early_stopping=False,
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
            if early_stopping:
                early_stopping = EarlyStopping(patience=7, verbose=True)

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

                if early_stopping:
                    best_model = early_stopping(test_loss, self)
                    if best_model:
                        self.save_model(join(train_dir, "best_model.pt"), save_training_info=False)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break
                print(f"Finished training epoch {self.epoch}.\n")

    def sample(
        self,
        *x,
        batch_size=None,
        num_samples=None,
        get_log_prob=False,
    ):
        """
        Sample from posterior model, conditioned on context x. x is expected to have a
        batch dimension, i.e., to obtain N samples with additional context requires
        x = x_.expand(N, *x_.shape).

        This method takes care of the batching, makes sure that self.network is in
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

        self.network.eval()
        with torch.no_grad():
            if batch_size is None:
                if get_log_prob:
                    samples, log_prob = self.sample_and_log_prob_batch(*x)
                else:
                    samples = self.sample_batch(*x)
            else:
                if num_samples is None:
                    num_samples = batch_size
                samples = []
                if get_log_prob:
                    log_prob = []

                num_batches = math.ceil(len(x[0]) / batch_size) if x else math.ceil(num_samples / batch_size)
                for idx_batch in range(num_batches):
                    if x:
                        lower, upper = idx_batch * batch_size, (idx_batch + 1) * batch_size
                        x_batch = [xi[lower:upper] for xi in x]
                        batch_size = None
                    else:
                        x_batch = x

                    if get_log_prob:
                        samples_batch, log_prob_batch = self.sample_and_log_prob_batch(
                            *x_batch, batch_size=batch_size
                        )
                        samples.append(samples_batch)
                        log_prob.append(log_prob_batch)
                    else:
                        samples.append(self.sample_batch(*x_batch))
                samples = torch.cat(samples, dim=0)
                if get_log_prob:
                    log_prob = torch.cat(log_prob, dim=0)
        if not get_log_prob:
            return samples
        else:
            return samples, log_prob


def train_epoch(pm, dataloader):
    pm.network.train()
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
        pm.network.eval()
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
