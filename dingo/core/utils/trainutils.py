import csv
import os
import time
from os.path import isfile, join
from typing import Literal, Optional

import numpy as np
import torch
import torch.distributed as dist


class AvgTracker:
    def __init__(self):
        self.sum = 0
        self.N = 0
        self.x = None

    def update(self, x, n=1):
        self.sum += x
        self.N += n
        self.x = x

    def get_avg(self):
        if self.N == 0:
            return float("nan")
        return self.sum / self.N


class EarlyStopping:
    """
    Implement early stopping during training, once the validation loss stops decreasing
    for a certain number of epochs (the patience).

    If val_loss > min_val_loss - delta for more than patience epochs, then returns
    early stopping occurs.
    """

    def __init__(
        self,
        patience: int = 5,
        verbose: bool = False,
        delta: float = 0.0,
        metric: Literal["training", "validation"] = "validation",
    ):
        """
        Parameters
        ----------
        patience: int = 5
            Number of epochs to wait before stopping.
        verbose: bool = False
            Whether to print counter increments.
        delta: float = 0.0
            Amount by which loss must decrease in patience epochs.
        metric: Literal["training", "validation"]
            Whether to use the training loss to determine early stopping ("training") or the test loss ("validation")
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        if metric not in ["training", "validation"]:
            raise ValueError(
                "Early Stopping metric must be 'training' or 'validation'."
            )
        self.metric = metric

    def __call__(self, val_loss: float):
        """
        Parameters
        ----------
        val_loss: float
            Value of the validation loss.
        model

        Returns
        -------
        bool
            Whether the current model has the lowest validation loss so-far.
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            return True
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.counter = 0
            return True


class LossInfo:
    def __init__(
        self,
        epoch: int,
        len_dataset: int,
        batch_size_per_grad_update: int,
        mode: str = "Train",
        print_freq: int = 1,
        device: torch.device = torch.device("cuda"),
    ):
        # data for print statements
        self.epoch = epoch
        # iteration = number of optimizer steps
        self.iteration = 0
        self.len_dataset = len_dataset
        self.batch_size_per_grad_update = batch_size_per_grad_update
        self.mode = mode
        self.print_freq = print_freq
        self.device = device
        # track loss
        self.loss_tracker = AvgTracker()
        self.loss = None
        self.cached_losses: list = []
        self.cached_n: list = []
        # Use dist.is_initialized() so that LossInfo is DDP-aware only when a
        # process group has actually been set up.
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.num_gpus = dist.get_world_size()
            self.times = {
                "Dataloader": AvgTracker(),
                "Network": AvgTracker(),
                "Aggregation": AvgTracker(),
            }
        else:
            self.times = {"Dataloader": AvgTracker(), "Network": AvgTracker()}
        self.t = time.time()

    def cache_loss(self, loss: torch.Tensor, n: int) -> None:
        """Cache *loss* from one gradient-accumulation step."""
        self.cached_losses.append(loss.detach())
        self.cached_n.append(n)
        self.update_timer(timer_mode="Network")

    def _reset_cached_losses(self) -> None:
        self.cached_losses = []
        self.cached_n = []

    def update_timer(self, timer_mode: str = "Dataloader") -> None:
        if self.is_ddp:
            dt = torch.tensor(time.time() - self.t, device=self.device)
            dist.barrier()
            dist.reduce(dt, dst=0, op=dist.ReduceOp.MAX)
            dt = dt.item()
        else:
            dt = time.time() - self.t
        self.times[timer_mode].update(dt)
        self.t = time.time()

    def update(self) -> None:
        """Aggregate cached losses across gradient-accumulation steps (and across GPUs in DDP)."""
        self.iteration += 1
        loss = torch.mean(torch.tensor(self.cached_losses, device=self.device))
        n = torch.tensor(sum(self.cached_n), device=self.device, dtype=torch.float32)

        if self.is_ddp:
            # Reduce absolute loss across GPUs so that normalization is correct
            # even when GPUs process different numbers of samples.
            abs_loss = loss * n
            dist.barrier()
            dist.reduce(abs_loss, dst=0)
            dist.reduce(n, dst=0)
            loss = abs_loss / n
            self.update_timer(timer_mode="Aggregation")

        self.loss = loss.item()
        self.loss_tracker.update(self.loss * n.item(), n.item())
        self._reset_cached_losses()

    def get_avg(self) -> float:
        return self.loss_tracker.get_avg()

    def get_iteration(self) -> int:
        """Return the number of optimizer steps performed this epoch."""
        if self.is_ddp:
            # Verify that all ranks performed the same number of steps.
            dist.barrier()
            iteration = torch.tensor(
                self.iteration, device=self.device, dtype=torch.int64
            )
            # all_gather requires output tensors to match the shape of the input.
            gathered = [torch.zeros_like(iteration) for _ in range(self.num_gpus)]
            dist.all_gather(gathered, iteration)
            if not all(torch.equal(gathered[0], t) for t in gathered):
                raise ValueError(
                    f"DDP ranks disagree on iteration count: "
                    f"{[t.item() for t in gathered]}"
                )
        return self.iteration

    def print_info(self, batch_idx: int) -> None:
        if batch_idx % self.print_freq == 0:
            print(
                "{} Epoch: {} [{}/{} ({:.0f}%)]".format(
                    self.mode,
                    self.epoch,
                    min(
                        (batch_idx + 1) * self.batch_size_per_grad_update,
                        self.len_dataset,
                    ),
                    self.len_dataset,
                    min(
                        100.0
                        * (batch_idx + 1)
                        * self.batch_size_per_grad_update
                        / self.len_dataset,
                        100,
                    ),
                ),
                end="\t\t",
            )
            print(f"Loss: {self.loss:.3f} ({self.get_avg():.3f})", end="\t\t")
            td, td_avg = self.times["Dataloader"].x, self.times["Dataloader"].get_avg()
            tn, tn_avg = self.times["Network"].x, self.times["Network"].get_avg()
            print(f"Time Dataloader: {td:.3f} ({td_avg:.3f})", end="\t\t")
            if self.is_ddp:
                ta, ta_avg = (
                    self.times["Aggregation"].x,
                    self.times["Aggregation"].get_avg(),
                )
                print(f"Time Network: {tn:.3f} ({tn_avg:.3f})", end="\t\t")
                print(f"Time Loss Aggregation: {ta:.3f} ({ta_avg:.3f})")
            else:
                print(f"Time Network: {tn:.3f} ({tn_avg:.3f})")


class RuntimeLimits:
    """
    Keeps track of the runtime limits (time limit, epoch limit, max. number
    of epochs for model).

    In DDP training, ``limits_exceeded`` broadcasts the result across all ranks
    so that every process stops at the same epoch even when only one rank (e.g.
    rank 0) detects that the wall-clock limit has been reached.
    """

    def __init__(
        self,
        max_time_per_run: float = None,
        max_epochs_per_run: int = None,
        max_epochs_total: int = None,
        epoch_start: int = None,
        device: torch.device = torch.device("cuda"),
    ):
        """

        Parameters
        ----------
        max_time_per_run: float = None
            maximum time for run, in seconds
            [soft limit, break only after full epoch]
        max_epochs_per_run: int = None
            maximum number of epochs for run
        max_epochs_total: int = None
            maximum total number of epochs for model
        epoch_start: int = None
            start epoch of run
        device: torch.device
            Device used for the DDP all-reduce broadcast.
        """
        self.max_time_per_run = max_time_per_run
        self.max_epochs_per_run = max_epochs_per_run
        self.max_epochs_total = max_epochs_total
        self.epoch_start = epoch_start
        self.time_start = time.time()
        self.device = device
        self.is_ddp = dist.is_initialized()
        if max_epochs_per_run is not None and epoch_start is None:
            raise ValueError("epoch_start required to check max_epochs_per_run.")

    def limits_exceeded(self, epoch: Optional[int] = None) -> bool:
        """
        Check whether any of the runtime limits are exceeded.

        In DDP mode the boolean is broadcast via ``all_reduce`` so all ranks
        agree on whether to stop.

        Parameters
        ----------
        epoch: int = None

        Returns
        -------
        limits_exceeded: bool
            flag whether runtime limits are exceeded and run should be stopped;
            if limits_exceeded = True, this prints a message for the reason
        """
        exceeded = False
        # check time limit for run
        if self.max_time_per_run is not None:
            if time.time() - self.time_start >= self.max_time_per_run:
                print(f"Stop run: Time limit of {self.max_time_per_run} s exceeded.")
                exceeded = True
        # check epoch limit for run
        if self.max_epochs_per_run is not None:
            if epoch is None:
                raise ValueError("epoch required")
            if epoch - self.epoch_start >= self.max_epochs_per_run:
                print(
                    f"Stop run: Epoch limit of {self.max_epochs_per_run} per run reached."
                )
                exceeded = True
        # check total epoch limit
        if self.max_epochs_total is not None:
            if epoch >= self.max_epochs_total:
                print(
                    f"Stop run: Total epoch limit of {self.max_epochs_total} reached."
                )
                exceeded = True

        if self.is_ddp:
            # Broadcast the flag: if *any* rank exceeded a limit, all stop.
            flag = torch.tensor(exceeded, device=self.device, dtype=torch.bool)
            dist.barrier()
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            exceeded = flag.item()

        return exceeded

    def local_limits_exceeded(self, epoch: Optional[int] = None) -> bool:
        """
        Check whether any of the local runtime limits are exceeded. Local runtime
        limits include max_epochs_per_run and max_time_per_run, but not max_epochs_total.

        Parameters
        ----------
        epoch: int = None

        Returns
        -------
        limits_exceeded: bool
            flag whether local runtime limits are exceeded
        """
        # check time limit for run
        if self.max_time_per_run is not None:
            if time.time() - self.time_start >= self.max_time_per_run:
                return True
        # check epoch limit for run
        if self.max_epochs_per_run is not None:
            if epoch is None:
                raise ValueError("epoch required")
            if epoch - self.epoch_start >= self.max_epochs_per_run:
                return True
        # return False if none of the limits is exceeded
        return False


def write_history(
    log_dir,
    epoch,
    train_loss,
    test_loss,
    learning_rates,
    aux=None,
    filename="history.txt",
):
    """
    Writes losses and learning rate history to csv file.

    Parameters
    ----------
    log_dir: str
        directory containing the history file
    epoch: int
        epoch
    train_loss: float
        train_loss of epoch
    test_loss: float
        test_loss of epoch
    learning_rates: list
        list of learning rates in epoch
    aux: list = []
        list of auxiliary information to be logged
    filename: str = 'history.txt'
        name of history file
    """
    if aux is None:
        aux = []
    history_file = join(log_dir, filename)
    if epoch == 1:
        assert not isfile(
            history_file
        ), f"File {history_file} exists, aborting to not overwrite it."

    with open(history_file, "w" if epoch == 1 else "a") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([epoch, train_loss, test_loss, *learning_rates, *aux])


def copyfile(src, dst):
    """
    copy src to dst.
    :param src:
    :param dst:
    :return:
    """
    os.system("cp -p %s %s" % (src, dst))


def save_model(pm, log_dir, model_prefix="model", checkpoint_epochs=None):
    """
    Save model to <model_prefix>_latest.pt in log_dir. Additionally,
    all checkpoint_epochs a permanent checkpoint is saved.

    Parameters
    ----------
    pm:
        model to be saved
    log_dir: str
        log directory, where model is saved
    model_prefix: str = 'model'
        prefix for name of save model
    checkpoint_epochs: int = None
        number of steps between two consecutive model checkpoints
    """
    # save current model
    model_name = join(log_dir, f"{model_prefix}_latest.pt")
    print(f"Saving model to {model_name}.", end=" ")
    pm.save_model(model_name, save_training_info=True)
    print("Done.")

    # potentially copy model to a checkpoint
    if checkpoint_epochs is not None and pm.epoch % checkpoint_epochs == 0:
        model_name_cp = join(log_dir, f"{model_prefix}_{pm.epoch:03d}.pt")
        print(f"Copy model to checkpoint {model_name_cp}.", end=" ")
        copyfile(model_name, model_name_cp)
        print("Done.")
