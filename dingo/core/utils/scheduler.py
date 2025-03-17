from typing import List, Optional
from bisect import bisect_right
import copy
import numpy as np

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LRScheduler,
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    StepLR,
    SequentialLR,
    _check_verbose_deprecated_warning,
)


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
        optimizer: Optimizer,
        schedulers: List[LRScheduler],
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

        # Track whether last scheduler step was per optimizer step
        self.update_level_of_last_step = "epoch"

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


def get_scheduler_from_kwargs(
    optimizer: Optimizer,
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
        "step": StepLR,
        "cosine": CosineAnnealingLR,
        "reduce_on_plateau": ReduceLROnPlateau,
        "sequential": CustomSequentialLR,
        "linear": LinearLR,
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
        num_optimizer_steps = scheduler_kwargs.pop("num_optimizer_steps_per_epoch", 1)
        if update_ever_optimizer_step and num_optimizer_steps > 1:
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
    loss: float = None,
    scheduler_kwargs: dict = None,
    update_level: str = "epoch",
):
    """
    Wrapper for scheduler.step(). If scheduler is ReduceLROnPlateau,
    then scheduler.step(loss) is called, if not, scheduler.step().

    Parameters
    ----------
    scheduler:
        scheduler for learning rate
    loss: float, optional
        validation loss
    scheduler_kwargs: dict
        scheduler arguments for one or multiple schedulers. Each scheduler arguments can contain
        'update_scheduler_every_optimizer_step' (default=False) which determines whether to do a scheduler step every
        optimizer step or every epoch.
    update_level: str, optional
        Describes from where this function is called, either on the epoch level or on the level of an 'optimizer_step'.
    """

    if scheduler_kwargs is None:
        scheduler_kwargs = {}

    def perform_step(sched, metric: Optional[float] = None):
        if metric is not None:
            sched.step(metrics=metric)
        else:
            sched.step()

    # Standard scheduler
    if not isinstance(scheduler, CustomSequentialLR):
        update_per_step = scheduler_kwargs.get("update_every_optimizer_step", False)
        if (update_level == "epoch" and not update_per_step) or (
            update_level == "optimizer_step" and update_per_step
        ):
            if isinstance(scheduler, ReduceLROnPlateau):
                perform_step(scheduler, loss)
            else:
                perform_step(scheduler)
    # Sequential scheduler
    elif isinstance(scheduler, CustomSequentialLR):
        # Get currently active scheduler
        active_scheduler_index = scheduler.get_active_scheduler_index()
        active_scheduler = f"scheduler_{active_scheduler_index}"
        update_per_step = scheduler_kwargs[active_scheduler].get(
            "update_every_optimizer_step", False
        )
        # Check whether we are switching from updates per optimizer step to updates per epoch;
        # In this case, we have to prevent back-to-back scheduler steps and need to skip the first step.
        valid_step = True
        if (
            scheduler.update_level_of_last_step == "optimizer_step"
            and active_scheduler_index > 0
            and update_level == "epoch"
            and not update_per_step
        ):
            # Skip first step
            valid_step = False

        if (update_level == "epoch" and not update_per_step) or (
            update_level == "optimizer_step" and update_per_step
        ):
            if valid_step:
                perform_step(scheduler, loss)
            # Keep track of previous update level
            scheduler.update_level_of_last_step = update_level
