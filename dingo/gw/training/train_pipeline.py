from typing import Optional, Tuple
import argparse
import numpy as np
import os
import queue
import shutil
import sys
import textwrap
import time
import yaml

from copy import deepcopy
from multiprocessing import Value
import ctypes

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.modules import Module

from threadpoolctl import threadpool_limits

from dingo.core.posterior_models.base_model import BasePosteriorModel
from dingo.core.posterior_models.build_model import (
    autocomplete_model_kwargs,
    build_model_from_kwargs,
)
from dingo.core.posterior_models.pretraining_model import (
    check_pretraining_model_compatibility,
    build_pretraining_model_kwargs,
)
from dingo.core.utils import (
    set_requires_grad_flag,
    print_number_of_model_parameters,
    build_train_and_test_loaders,
    document_git_status,
    document_environment,
)
from dingo.core.utils.trainutils import EarlyStopping, RuntimeLimits
from dingo.core.utils.torchutils import (
    document_gpus,
    setup_ddp,
    cleanup_ddp,
    replace_BatchNorm_with_SyncBatchNorm,
    set_seed_based_on_rank,
)
from dingo.gw.dataset import WaveformDataset
from dingo.gw.training.train_builders import (
    build_dataset,
    set_train_transforms,
    build_svd_for_embedding_network,
)


def copy_files_to_local(
    file_path: str,
    local_dir: Optional[str],
    leave_keys_on_disk: bool,
    is_condor: bool = False,
) -> str:
    """
    Copy files to local node if local_dir is provided to minimize network traffic during training.

    Parameters
    ----------
    file_path: str
        Path to file that should be copied.
    local_dir: Optional[str]
        Directory where file should be copied. If None, file will not be copied.
    leave_keys_on_disk: bool
        Whether to leave keys on disk and load them during training. If dataset is not copied and
        leave_keys_on_disk is True, a warning will be raised.
    is_condor: bool
        Whether this is a condor job.

    Returns
    -------
    local_file_path: str
        Modified file path if file was copied to local node, else the original file path.
    """
    local_file_path = file_path
    if local_dir is not None:
        file_name = file_path.split("/")[-1]
        local_file_path = os.path.join(local_dir, file_name)
        print(f"Copying file from {file_path} to {local_file_path}")
        # Copy file
        start_time = time.time()
        try:
            shutil.copy(file_path, local_file_path)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"Path to os.environ['TMPDIR'] is {os.environ['TMPDIR']}")
        elapsed_time = time.time() - start_time
        print("Done. This took {:2.0f}:{:2.0f} min.".format(*divmod(elapsed_time, 60)))
    elif leave_keys_on_disk and is_condor:
        print(
            f"Warning: leave_waveforms_on_disk defaults to True, but local_cache_path is not specified. "
            f"This means that the waveforms will be loaded during training from {local_file_path} ."
            f"This can lead to unexpected long times for data loading during training due to network traffic. "
            f"To prevent this, specify 'local_cache_path = tmp' in the local settings or set "
            f"leave_waveforms_on_disk = False. However, the latter is not recommended for large datasets since "
            f"it can lead to memory issues when loading the entire dataset into RAM. "
        )

    return local_file_path


def prepare_wfd_and_initialization_for_embedding_network(
    train_settings: dict,
    train_dir: str,
    local_settings: dict,
    pretraining: Optional[bool] = False,
    print_output: Optional[bool] = True,
):
    """
    Based on a settings dictionary, initialize a WaveformDataset and parts of the embedding network.
    The latter include:
    - if embedding_type==DenseResidualNet: initializes the embedding network projection stage with SVD V matrices based
    on clean detector waveforms.
    - if pretraining: loads the full pretrained embedding network

    Parameters
    ----------
    train_settings : dict
        Settings which ultimately come from train_settings.yaml file.
    train_dir : str
        This is only used to save diagnostics from the SVD.
    local_settings : dict
        Local settings (e.g., num_workers, device)
    pretraining: bool
        Whether to run pretraining

    Returns
    -------
    (WaveformDataset, initial_weights, BasePosteriorModel)
    """
    data_settings = deepcopy(train_settings["data"])
    # Optionally copy files to local and update path
    data_settings["waveform_dataset_path"] = copy_files_to_local(
        file_path=data_settings["waveform_dataset_path"],
        local_dir=local_settings.get("local_cache_path", None),
        leave_keys_on_disk=local_settings.get("leave_waveforms_on_disk", True),
        is_condor=True if "condor" in local_settings else False,
    )
    wfd = build_dataset(
        data_settings=data_settings,
        leave_waveforms_on_disk=local_settings.get("leave_waveforms_on_disk", True),
    )  # No transforms yet
    initial_weights = {}
    pretrained_embedding_net = None

    # Prepare initial embedding network weights: Build SVD or load pretrained network
    if train_settings["model"].get("embedding_kwargs", None):
        if (
            train_settings["model"]["posterior_model_type"] == "normalizing_flow"
            and train_settings["model"]["embedding_type"] == "DenseResidualNet"
            and not train_settings["model"]["embedding_kwargs"]["svd"].get(
                "no_init", False
            )
        ):
            # Build the SVD for seeding the resnet embedding network.
            if print_output:
                print("\nBuilding SVD for initialization of ResNet embedding network.")

            batch_size = train_settings["training"]["stage_0"]["batch_size"]
            if "condor" in local_settings:
                num_gpus = local_settings["condor"].get("num_gpus", 1)
            else:
                num_gpus = 1
            # Compute batch size per GPU in case of multi-GPU training
            if num_gpus > 1:
                if batch_size % num_gpus != 0:
                    raise ValueError(
                        f"Total batch size {batch_size} is not divisible by the number of GPUs {num_gpus}."
                    )
                batch_size = int(batch_size / num_gpus)
            initial_weights["V_rb_list"] = build_svd_for_embedding_network(
                wfd=wfd,
                data_settings=train_settings["data"],
                asd_dataset_path=train_settings["training"]["stage_0"][
                    "asd_dataset_path"
                ],
                num_workers=local_settings["num_workers"],
                batch_size=batch_size,
                out_dir=train_dir,
                **train_settings["model"]["embedding_kwargs"]["svd"],
            )
        else:
            initial_weights = None
            if print_output:
                print("Building embedding network without SVD initialization.")

        if pretraining:
            train_settings = build_pretraining_model_kwargs(train_settings)

        if not pretraining and "pretraining" in train_settings.keys():
            # Load pretrained embedding network
            pretrained_model_path = os.path.join(
                train_dir, "pretraining", "model_latest.pt"
            )
            if not os.path.isfile(pretrained_model_path):
                raise ValueError(
                    f"No pretrained model found at {pretrained_model_path}. If you want to start pretraining"
                    f"from scratch, delete the pretraining folder in train_dir."
                )
            if print_output:
                print(
                    f"Loading embedding weights from pretrained model at {pretrained_model_path}."
                )
            pm = build_model_from_kwargs(
                filename=pretrained_model_path,
                pretraining=False,
                pretrained_embedding_net=None,
                device=local_settings["device"],
                print_output=print_output,
            )
            # Check whether loaded model has same architecture as specified in train_settings
            check_pretraining_model_compatibility(train_settings, pm.metadata)
            pretrained_embedding_net = pm.network.embedding_net
    else:
        raise ValueError("No embedding_kwargs specified in model.")

    # Now set the transforms for training. We need to do this here so that we can (a)
    # get the data dimensions to configure the network, and (b) save the
    # parameter standardization dict in the PosteriorModel. In principle, (a) could
    # be done without generating data (by careful calculation) and (b) could also
    # be done outside the transform setup. But for now, this is convenient. The
    # transforms will be reset later by initialize_stage().

    set_train_transforms(
        wfd=wfd,
        data_settings=train_settings["data"],
        asd_dataset_path=train_settings["training"]["stage_0"]["asd_dataset_path"],
        print_output=print_output,
    )

    return wfd, initial_weights, pretrained_embedding_net


def prepare_model_new(
    train_settings: dict,
    train_dir: str,
    local_settings: dict,
    wfd: WaveformDataset,
    initial_weights: dict,
    pretraining: bool,
    pretrained_embedding_net,
):
    """
    Based on a settings dictionary, initialize a WaveformDataset and parts of the embedding network.
    The latter include:
    - if embedding_type==DenseResidualNet: initializes the embedding network projection stage with SVD V matrices based
    on clean detector waveforms.
    - if pretraining: loads the full pretrained embedding network

    Parameters
    ----------
    train_settings : dict
        Settings which ultimately come from train_settings.yaml file.
    train_dir : str
        This is only used to save diagnostics from the SVD.
    local_settings : dict
        Local settings (e.g., num_workers, device)
    wfd: WaveformDataset
        The WaveformDataset is required since the model_kwargs are autocompleted based on the dimensions of one
        data point.
    initial_weights: dict
        The weights for the initial layer of the embedding network based on the SVD.
    pretraining: bool
        Whether to run pretraining
    pretrained_embedding_net: torch.nn.module
        If pretraining=True, pretrained embedding network

    Returns
    -------
    (BasePosteriorModel)
    """

    # This modifies the model settings in-place.
    is_gnpe = True if "gnpe_time_shifts" in train_settings["data"] else False
    autocomplete_model_kwargs(
        model_kwargs=train_settings["model"],
        data_sample=wfd[0],
        gnpe=is_gnpe,
    )
    full_settings = {
        "dataset_settings": wfd.settings,
        "train_settings": train_settings,
    }
    print_output = (
        True
        if ("rank" not in local_settings or local_settings.get("rank", None) == 0)
        else False
    )
    if print_output:
        print("\nInitializing new posterior model.")
        print("Complete settings:")
        print(yaml.dump(full_settings, default_flow_style=False, sort_keys=False))

    pm = build_model_from_kwargs(
        settings=full_settings,
        pretraining=pretraining,
        pretrained_embedding_net=pretrained_embedding_net,
        initial_weights=initial_weights,
        device=local_settings["device"],
        print_output=print_output,
    )

    if local_settings.get("wandb", False) and local_settings.get("rank", 0.0) == 0.0:
        try:
            import wandb

            wandb.init(
                config=full_settings,
                dir=train_dir,
                **local_settings["wandb"],
            )
        except ImportError:
            print("WandB is enabled but not installed.")

    return pm


def prepare_training_new(
    train_settings: dict,
    train_dir: str,
    local_settings: dict,
    pretraining: bool = False,
):
    """
    Based on a settings dictionary, initialize a WaveformDataset and PosteriorModel.

    For model type 'nsf+embedding' (the only acceptable type at this point) this also
    initializes the embedding network projection stage with SVD V matrices based on
    clean detector waveforms.

    Parameters
    ----------
    train_settings : dict
        Settings which ultimately come from train_settings.yaml file.
    train_dir : str
        This is only used to save diagnostics from the SVD.
    local_settings : dict
        Local settings (e.g., num_workers, device)
    pretraining: bool
        Whether to run pretraining

    Returns
    -------
    (BasePosteriorModel, WaveformDataset)
    """

    (
        wfd,
        initial_weights,
        pretrained_embedding_net,
    ) = prepare_wfd_and_initialization_for_embedding_network(
        train_settings=train_settings,
        train_dir=train_dir,
        local_settings=local_settings,
        pretraining=pretraining,
    )
    pm = prepare_model_new(
        train_settings=train_settings,
        train_dir=train_dir,
        local_settings=local_settings,
        wfd=wfd,
        initial_weights=initial_weights,
        pretraining=pretraining,
        pretrained_embedding_net=pretrained_embedding_net,
    )

    return pm, wfd


def load_settings_from_ckpt(checkpoint_name: str):
    """Load settings from checkpoint file.

    Parameters
    ----------
    checkpoint_name : str
        Path to checkpoint file

    Returns
    -------
    dict
    """
    pm = build_model_from_kwargs(
        filename=checkpoint_name,
        pretraining=False,
        pretrained_embedding_net=None,
        device="meta",
    )
    return pm.metadata["train_settings"]


def prepare_model_resume(
    checkpoint_name: str,
    local_settings: dict,
    train_dir: str,
    pretraining: Optional[bool] = False,
) -> BasePosteriorModel:
    """
    Loads a PosteriorModel from a checkpoint, as well as the corresponding
    WaveformDataset, in order to continue training. It initializes the saved optimizer
    and scheduler from the checkpoint.

    Parameters
    ----------
    checkpoint_name : str
        File name containing the checkpoint (.pt format).
    local_settings : dict
        Local settings (e.g., num_workers, device)
    train_dir: str
        Path to training directory where the wandb info is saved.
    pretraining: bool
        Whether to resume in pretraining mode

    Returns
    -------
    (BasePosteriorModel)
    """
    print_output = (
        True
        if ("rank" not in local_settings or local_settings.get("rank", None) == 0)
        else False
    )
    pm = build_model_from_kwargs(
        filename=checkpoint_name,
        pretraining=pretraining,
        pretrained_embedding_net=None,
        device=local_settings["device"],
        print_output=print_output,
    )

    if local_settings.get("wandb", False):
        try:
            import wandb

            wandb.init(
                resume="must",
                dir=train_dir,
                **local_settings["wandb"],
            )
        except ImportError:
            print("WandB is enabled but not installed.")

    return pm


def prepare_training_resume(
    checkpoint_name: str,
    local_settings: dict,
    train_dir: str,
    pretraining: Optional[bool] = False,
) -> Tuple[BasePosteriorModel, WaveformDataset]:
    """
    Loads a PosteriorModel from a checkpoint, as well as the corresponding
    WaveformDataset, in order to continue training. It initializes the saved optimizer
    and scheduler from the checkpoint.

    Parameters
    ----------
    checkpoint_name : str
        File name containing the checkpoint (.pt format).
    local_settings : dict
        A dictionary containing the local settings.
    train_dir: str
        The directory where all training information is saved.
    pretraining: bool
        Whether to resume in pretraining mode

    Returns
    -------
    (BasePosteriorModel, WaveformDataset)
    """

    train_settings = load_settings_from_ckpt(checkpoint_name)

    data_settings = deepcopy(train_settings["data"])
    # Optionally copy files to local and update path
    data_settings["waveform_dataset_path"] = copy_files_to_local(
        file_path=data_settings["waveform_dataset_path"],
        local_dir=local_settings.get("local_cache_path", None),
        leave_keys_on_disk=local_settings.get("leave_waveforms_on_disk", True),
        is_condor=True if "condor" in local_settings else False,
    )
    wfd = build_dataset(
        data_settings=data_settings,
        leave_waveforms_on_disk=local_settings.get("leave_waveforms_on_disk", True),
    )

    pm = prepare_model_resume(
        checkpoint_name=checkpoint_name,
        local_settings=local_settings,
        train_dir=train_dir,
        pretraining=pretraining,
    )

    return pm, wfd


def initialize_stage(
    pm: BasePosteriorModel,
    wfd: WaveformDataset,
    stage: dict,
    num_workers: int,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    resume: Optional[bool] = False,
):
    """
    Initializes training based on PosteriorModel metadata and current stage:
        * Builds transforms (based on noise settings for current stage);
        * Builds DataLoaders (optional: for distributed training);
        * At the beginning of a stage (i.e., if not resuming mid-stage), initializes
        a new optimizer and scheduler;
        * Freezes / unfreezes SVD layer of embedding network

    Parameters
    ----------
    pm : BasePosteriorModel
    wfd : WaveformDataset
    stage : dict
        Settings specific to current stage of training
    num_workers : int
    world_size: int = None
        total number of devices required for distributed data parallel training
    rank: int = None
        device rank required for distributed data parallel training
    resume : bool
        Whether training is resuming mid-stage. This controls whether the optimizer and
        scheduler should be re-initialized based on contents of stage dict.

    Returns
    -------
    (train_loader, test_loader, train_sampler)
    """

    train_settings = pm.metadata["train_settings"]

    # Rebuild transforms based on possibly different noise.
    print_output = True if rank is None or rank == 0 else False
    set_train_transforms(
        wfd=wfd,
        data_settings=train_settings["data"],
        asd_dataset_path=stage["asd_dataset_path"],
        print_output=print_output,
    )

    # Convert total batch size into batch size per GPU
    if world_size is not None and world_size > 1:
        total_batch_size = stage["batch_size"]
        if total_batch_size % world_size != 0:
            raise ValueError(
                f"Total batch size {total_batch_size} is not divisible by the number of GPUs {world_size}."
            )
        batch_size_per_gpu = int(total_batch_size / world_size)
    else:
        batch_size_per_gpu = stage["batch_size"]

    # Allows for changes in batch size between stages.
    train_loader, test_loader, train_sampler = build_train_and_test_loaders(
        dataset=wfd,
        train_fraction=train_settings["data"]["train_fraction"],
        batch_size=batch_size_per_gpu,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
    )

    if not resume:
        # New optimizer and scheduler. If we are resuming, these should have been
        # loaded from the checkpoint.
        if print_output:
            print("Initializing new optimizer and scheduler.")
        pm.optimizer_kwargs = stage["optimizer"]
        pm.scheduler_kwargs = stage["scheduler"]
        # Precompute number of optimizer steps per epoch in case a scheduler is updated every optimizer step
        # (instead of every epoch).
        # Warning: The following computation assumes that ...
        # ... the full training data set is used for training
        # ... we use drop_last = False in both the DistributedSampler and in the train Dataloader
        train_size = int(
            train_settings["data"]["train_fraction"] * wfd.settings["num_samples"]
        )
        grad_updates_per_optimizer_step = stage.get(
            "gradient_updates_per_optimizer_step", 1
        )
        num_gpus = world_size if world_size is not None else 1
        # (1) When using drop_last = False (default) in the DistributedSampler, some data points are repeatedly sampled
        # such that each GPU processes the same amount of data in multi-GPU training. The effective size of the dataset
        # has to be a multiple of the number of GPUs and the batch size per GPU:
        effective_samples_per_gpu = int(np.ceil(train_size / num_gpus))
        # (2) When using drop_last = False (default) in the Dataloader, the last batch can be incomplete:
        effective_num_batches_per_gpu = int(
            np.ceil(effective_samples_per_gpu / batch_size_per_gpu)
        )
        # (3) If we perform multiple gradient updates per optimizer step, this results in a factor of
        # grad_updates_per_optimizer_step fewer optimizer steps per GPU because an optimizer step is only performed
        # after grad_updates_per_optimizer_step backward passes. If the effective_batches_per_gpu is not divisible
        # by grad_updates_per_optimizer_step, no optimizer step is performed for the last incomplete gradient update.
        if effective_num_batches_per_gpu % grad_updates_per_optimizer_step != 0:
            print(
                f"The effective number of batches per GPU={effective_num_batches_per_gpu} is not divisible "
                f"by the number of gradient updates per optimizer step={grad_updates_per_optimizer_step}, discarding"
                f"the last {effective_num_batches_per_gpu % grad_updates_per_optimizer_step} batches. "
            )
        num_optimizer_steps = np.floor(
            effective_num_batches_per_gpu / grad_updates_per_optimizer_step
        )
        if print_output:
            print(
                f"Training with an effective batch size of {batch_size_per_gpu * num_gpus * grad_updates_per_optimizer_step} "
                f"on {num_gpus} CPU/GPU(s) with {num_optimizer_steps} optimizer steps per epoch. "
            )

        pm.initialize_optimizer_and_scheduler(num_optimizer_steps=num_optimizer_steps)

    # Freeze/unfreeze RB layer if necessary
    if "freeze_rb_layer" in stage:
        if stage["freeze_rb_layer"]:
            # Multi-GPU training: Changes to the model have to be made before wrapping the model in DDP
            if world_size is not None and world_size > 1:
                raise ValueError(
                    f"Not possible to freeze RB layer during multi-GPU training."
                )
            set_requires_grad_flag(
                pm.network, name_contains="layers_rb", requires_grad=False
            )
        else:
            set_requires_grad_flag(
                pm.network, name_contains="layers_rb", requires_grad=True
            )
    if print_output:
        print_number_of_model_parameters(network=pm.network)

    return train_loader, test_loader, train_sampler


def train_stages(
    pm: BasePosteriorModel,
    wfd: WaveformDataset,
    train_dir: str,
    local_settings: dict,
    global_epoch: ctypes.c_int = Value(ctypes.c_int, 1),
) -> Tuple[bool, bool]:
    """
    Train the network, iterating through the sequence of stages. Stages can change
    certain settings such as the noise characteristics, optimizer, and scheduler settings.

    Parameters
    ----------
    pm : BasePosteriorModel
    wfd : WaveformDataset
    train_dir : str
        Directory for saving checkpoints and train history.
    local_settings : dict

    Returns
    -------
    training_finished: bool
        True if last epoch finished
    resume_training: bool
        True if time limit reached and training should resume
        False otherwise
    """

    train_settings = pm.metadata["train_settings"]
    runtime_limits = RuntimeLimits(
        epoch_start=pm.epoch, device=pm.device, **local_settings["runtime_limits"]
    )
    rank = local_settings.get("rank", None)
    print_bool = True if rank is None or rank == 0 else False

    # Extract list of stages from settings dict
    stages = []
    num_stages = 0
    while True:
        try:
            stages.append(train_settings["training"][f"stage_{num_stages}"])
            num_stages += 1
        except KeyError:
            break
    end_epochs = list(np.cumsum([stage["epochs"] for stage in stages]))

    resume_training = False
    num_starting_stage = np.searchsorted(end_epochs, pm.epoch + 1)
    for n in range(num_starting_stage, num_stages):
        stage = stages[n]

        if pm.epoch == end_epochs[n] - stage["epochs"]:
            if print_bool:
                print(f"\nBeginning training stage {n}. Settings:")
                print(yaml.dump(stage, default_flow_style=False, sort_keys=False))
            train_loader, test_loader, train_sampler = initialize_stage(
                pm=pm,
                wfd=wfd,
                stage=stage,
                num_workers=local_settings["num_workers"],
                world_size=local_settings.get("world_size", None),
                rank=rank,
                resume=False,
            )
        else:
            if print_bool:
                print(f"\nResuming training in stage {n}. Settings:")
                print(yaml.dump(stage, default_flow_style=False, sort_keys=False))
            train_loader, test_loader, train_sampler = initialize_stage(
                pm=pm,
                wfd=wfd,
                stage=stage,
                num_workers=local_settings["num_workers"],
                world_size=local_settings.get("world_size", None),
                rank=rank,
                resume=True,
            )
        early_stopping = None
        if stage.get("early_stopping"):
            try:
                early_stopping = EarlyStopping(**stage["early_stopping"])
            except Exception:
                print(
                    "Early stopping settings invalid. Please pass 'patience', 'delta', 'metric'"
                )
                raise

        runtime_limits.max_epochs_total = end_epochs[n]
        pm.train(
            train_loader=train_loader,
            test_loader=test_loader,
            train_sampler=train_sampler,
            train_dir=train_dir,
            runtime_limits=runtime_limits,
            checkpoint_epochs=local_settings["checkpoint_epochs"],
            use_wandb=local_settings.get("wandb", False),
            test_only=local_settings.get("test_only", False),
            early_stopping=early_stopping,
            gradient_updates_per_optimizer_step=stage.get(
                "gradient_updates_per_optimizer_step", 1
            ),
            automatic_mixed_precision=stage.get("automatic_mixed_precision", False),
            world_size=local_settings.get("world_size", 1),
            global_epoch=global_epoch,
        )
        # if test_only, model should not be saved, and run is complete
        if local_settings.get("test_only", False):
            return True, False

        # Only save model for one device
        if pm.epoch == end_epochs[n] and print_bool:
            save_file = os.path.join(train_dir, f"model_stage_{n}.pt")
            print(f"Training stage complete. Saving to {save_file}.")
            pm.save_model(save_file, save_training_info=True)

        if runtime_limits.local_limits_exceeded(pm.epoch):
            if print_bool:
                print("Local runtime limits reached. Ending program.")
            resume_training = True
            break

    training_finished = False
    if pm.epoch == end_epochs[-1]:
        resume_training = False
        training_finished = True

    return training_finished, resume_training


def run_training(
    train_settings: dict,
    local_settings: dict,
    train_dir: str,
    ckpt_file: str,
    resume: bool,
    pretraining: bool = False,
) -> (bool, bool, int):
    """
    Starts a training run for single-GPU training.

    Parameters
    ----------
    train_settings: dict
        Settings for training
    local_settings: dict
        Local settings
    train_dir: str
        Directory to store training output
    ckpt_file: str
        if resume=True, path to the model checkpoint
    resume: bool
        Whether to resume training from checkpoint
    pretraining: bool=False
        Whether to run pretraining

    Returns
    ----------
    complete: bool
        Whether the training run was completed
    resume: bool
        Whether the training should resume in a new job
    epoch: int
        The epoch number where the training finished
    """
    if not resume:
        pm, wfd = prepare_training_new(
            train_settings=train_settings,
            train_dir=train_dir,
            local_settings=local_settings,
        )
    else:
        pm, wfd = prepare_training_resume(
            checkpoint_name=ckpt_file,
            local_settings=local_settings,
            train_dir=train_dir,
        )
    # Set global epoch
    global_epoch = Value(ctypes.c_int, pm.epoch)  # Shared across processes
    wfd.epoch = global_epoch

    with threadpool_limits(limits=1, user_api="blas"):
        complete, resume = train_stages(
            pm=pm,
            wfd=wfd,
            train_dir=train_dir,
            local_settings=local_settings,
            global_epoch=global_epoch,
        )

    return complete, resume, pm.epoch


def run_multi_gpu_training(
    world_size: int,
    train_settings: dict | None,
    local_settings: dict,
    train_dir: str,
    ckpt_file: str,
    resume: bool,
    pretraining: bool = False,
) -> (bool, bool, int):
    """
    Starts a training run for multi-GPU distributed data parallel (DDP) training.

    Parameters
    ----------
    world_size: int
        The number of GPUs to use for DDP training.
    train_settings: dict
        Settings for training
    local_settings: dict
        Local settings
    train_dir: str
        Directory to store training output
    ckpt_file: str
        if resume=True, path to the model checkpoint
    resume: bool
        Whether to resume training from checkpoint
    pretraining: bool=False
        Whether to run pretraining

    Returns
    ----------
    complete: bool
        Whether the training is finished
    resume: bool
        Whether the training should be resumed
    epoch: int
        The epoch number where the training finished
    """

    initial_weights, pretrained_emb_net, checkpoint_file = None, None, None
    if not resume:
        (
            wfd,
            initial_weights,
            pretrained_emb_net,
        ) = prepare_wfd_and_initialization_for_embedding_network(
            train_settings=train_settings,
            train_dir=train_dir,
            local_settings=local_settings,
        )
    else:
        checkpoint_file = os.path.join(train_dir, ckpt_file)
        train_settings = load_settings_from_ckpt(ckpt_file)
        data_settings = deepcopy(train_settings["data"])
        # Optionally copy files to local and update path
        data_settings["waveform_dataset_path"] = copy_files_to_local(
            file_path=data_settings["waveform_dataset_path"],
            local_dir=local_settings.get("local_cache_path", None),
            leave_keys_on_disk=local_settings.get("leave_waveforms_on_disk", True),
            is_condor=True if "condor" in local_settings else False,
        )
        wfd = build_dataset(
            data_settings=data_settings,
            leave_waveforms_on_disk=local_settings.get("leave_waveforms_on_disk", True),
        )
    if pretrained_emb_net is not None:
        pretraining = True
    else:
        pretraining = False

    # Initialize global epoch
    global_epoch = Value(ctypes.c_int, 0)  # Shared across processes

    # Setup multi-processing queue
    result_queue = mp.Queue()
    processes = []
    # Start one process for each gpu
    for rank in range(world_size):
        p = mp.Process(
            target=run_training_ddp,
            args=(
                rank,
                world_size,
                train_settings,
                local_settings,
                train_dir,
                wfd,
                initial_weights,
                pretraining,
                pretrained_emb_net,
                checkpoint_file,
                resume,
                result_queue,
                global_epoch,
            ),
        )
        p.start()
        processes.append(p)

    # Monitor the processes
    error_occurred = False
    for p in processes:
        # Force parent process to wait for the subprocess to complete/exit with error
        p.join()
        # Check whether process p failed
        if p.exitcode != 0:
            error_occurred = True
            print(
                f"Process {p.pid} failed with exit code {p.exitcode}. Aborting all processes."
            )
            # Terminate all other processes
            for proc in processes:
                if proc.is_alive():
                    proc.terminate()
            break

    # Handle the result queue to see what the specific error was
    results = []
    try:
        while True:
            temp_result = result_queue.get_nowait()  # Avoid blocking
            # contains (rank, complete, resume, epoch, error_message)
            results.append(temp_result)
            if temp_result[-1] is not None:
                print(f"Rank {temp_result[0]} failed with error: {temp_result[-1]}")
            else:
                print(f"Rank {temp_result[0]} finished successfully.")
    except queue.Empty:
        pass

    if error_occurred or len(results) != world_size:
        raise RuntimeError(
            f"One or more processes failed or number of results={len(results)} does not match "
            f"world_size={world_size}, check info.out for details."
        )

    # Collect exit results from all processes after training
    complete, resume, pm_epoch = [], [], []
    for result in results:
        complete.append(result[1])
        resume.append(result[2])
        pm_epoch.append(result[3])
    assert (
        len(set(complete)) == 1
    ), f"Processes do not return the same for whether the training is completed or not: {complete}."
    assert (
        len(set(resume)) == 1
    ), f"Processes do not return the same for whether to resume training: {resume}."
    assert (
        len(set(pm_epoch)) == 1
    ), f"Processes do not return the same epochs: {pm_epoch}."

    return complete[0], resume[0], pm_epoch[0]


def run_training_ddp(
    rank: int,
    world_size: int,
    train_settings: dict,
    local_settings: dict,
    train_dir: str,
    wfd: WaveformDataset,
    initial_weights: dict,
    pretraining: bool,
    pretrained_emb_net: Module,
    ckpt_file: str,
    resume: bool,
    result_queue: mp.Queue = None,
    global_epoch: int = Value(ctypes.c_int, 0),
) -> (bool, int):
    """
    Initializes each GPU process of the distributed data parallel (DDP) training.

    Parameters
    ----------
    rank: int
        The rank of the current GPU process.
    world_size: int
        The number of GPUs to use for DDP training.
    train_settings: dict
        Settings for training
    local_settings: dict
        Local settings
    train_dir: str
        Directory to store training output
    wfd: WaveformDataset
        Waveform dataset used for training
    initial_weights: dict
        Initial weights for embedding network
    pretraining: bool
        Whether to use the pretrained embedding network
    pretrained_emb_net: Module
        Pretrained embedding network
    ckpt_file: str
        if resume=True, path to the model checkpoint
    resume: bool
        Whether to resume training from checkpoint
    result_queue:
        The queue to which the results of the run will be stored
    global_epoch: multiprocessing.Value
        Global epoch stored in shared memory

    Returns
    ----------
    complete: bool
        Whether the training run was successful
    epoch: int
        The epoch number where the training finished
    """
    try:
        # Initialize process group
        setup_ddp(rank, world_size)
        # Set seeds for each GPU to different value
        set_seed_based_on_rank(rank)

        # Document GPUs
        if rank == 0:
            document_gpus(train_dir)

        local_settings["device"] = f"cuda:{rank}"
        local_settings["rank"] = rank
        local_settings["world_size"] = world_size

        if not resume:
            pm = prepare_model_new(
                train_settings,
                train_dir,
                local_settings,
                wfd=wfd,
                initial_weights=initial_weights,
                pretraining=pretraining,
                pretrained_embedding_net=pretrained_emb_net,
            )
        else:
            pm = prepare_model_resume(
                checkpoint_name=ckpt_file,
                local_settings=local_settings,
                train_dir=train_dir,
            )
            if rank == 0:
                print(
                    "Warning: If batch norm is used in the model, you have to make sure that the batch norm statistics "
                    "are loaded correctly for mult-GPU training. This has not been tested yet."
                )

        # Replace BatchNorm layers with SyncBatchNorm
        pm.network = replace_BatchNorm_with_SyncBatchNorm(pm.network)
        # Wrap the model with DDP
        pm.network = DDP(pm.network, device_ids=[rank])
        # Set global epoch
        global_epoch.value = pm.epoch
        wfd.epoch = global_epoch

        complete, resume = train_stages(
            pm, wfd, train_dir, local_settings, global_epoch
        )

        if complete and local_settings.get("wandb", False) and rank == 0.0:
            try:
                import wandb

                wandb.finish()
            except ImportError:
                print("wandb not installed. Skipping logging to wandb.")
        # Delete process group
        cleanup_ddp()

    except Exception as e:
        # In case of an error, put the exception in the result queue
        result_queue.put((rank, False, False, 0, str(e)))
        sys.exit(1)  # Exit with a non-zero code to indicate failure

    # Put return info on queue
    result_queue.put((rank, complete, resume, pm.epoch, None))
    sys.exit(0)  # Exit with zero to indicate success


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
        Train a neural network for gravitational-wave single-event inference.
        
        This program can be called in one of two ways:
            a) with a settings file. This will create a new network based on the 
            contents of the settings file.
            b) with a checkpoint file. This will resume training from the checkpoint.
        """
        ),
    )
    parser.add_argument(
        "--settings_file",
        type=str,
        help="YAML file containing training settings.",
    )
    parser.add_argument(
        "--train_dir", required=True, help="Directory for Dingo training output."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint file from which to resume training.",
    )
    parser.add_argument(
        "--exit_command",
        type=str,
        default="",
        help="Optional command to execute after completion of training.",
    )
    parser.add_argument(
        "--pretraining",
        action="store_true",
        help="Whether to pretrain embedding network.",
    )
    args = parser.parse_args()

    # The settings file and checkpoint are mutually exclusive.
    if args.checkpoint is None and args.settings_file is None:
        parser.error("Must specify either a checkpoint file or a settings file.")
    if args.checkpoint is not None and args.settings_file is not None:
        parser.error("Cannot specify both a checkpoint file and a settings file.")

    return args


def train_local():
    args = parse_args()

    os.makedirs(args.train_dir, exist_ok=True)
    if args.pretraining:
        args.train_dir = os.path.join(args.train_dir, "pretraining")
        os.makedirs(args.train_dir, exist_ok=True)

    # Document git
    document_git_status(args.train_dir)
    # Document setup
    document_environment(args.train_dir)
    # Cannot document GPU info here because this results in problems with mp.Process

    resume = True if args.settings_file is None else False

    if not resume:
        print("Beginning new training run.")
        with open(args.settings_file, "r") as fp:
            train_settings = yaml.safe_load(fp)

        # Extract the local settings from train settings file, save it separately. This
        # file can later be modified, and the settings take effect immediately upon
        # resuming.

        local_settings = train_settings.pop("local")
        with open(os.path.join(args.train_dir, "local_settings.yaml"), "w") as f:
            if (
                local_settings.get("wandb", False)
                and "id" not in local_settings["wandb"].keys()
            ):
                try:
                    import wandb

                    local_settings["wandb"]["id"] = wandb.util.generate_id()
                except ImportError:
                    print("wandb not installed, cannot generate run id.")
            yaml.dump(local_settings, f, default_flow_style=False, sort_keys=False)

    else:
        print("Resuming training run.")
        train_settings = None
        with open(os.path.join(args.train_dir, "local_settings.yaml"), "r") as f:
            local_settings = yaml.safe_load(f)

    # Setup multi-GPU training
    if (
        local_settings["device"] == "cuda"
        and "condor" in local_settings
        and "num_gpus" in local_settings["condor"]
        and local_settings["condor"]["num_gpus"] > 1.0
    ):
        # Specify the number of processes (typically the number of GPUs available)
        world_size = local_settings["condor"]["num_gpus"]

        complete, resume, pm_epoch = run_multi_gpu_training(
            world_size=world_size,
            train_settings=train_settings,
            local_settings=local_settings,
            train_dir=args.train_dir,
            ckpt_file=args.checkpoint,
            resume=resume,
            pretraining=args.pretraining,
        )
    else:
        if local_settings["device"] == "cuda":
            document_gpus(args.train_dir)
        complete, resume, pm_epoch = run_training(
            train_settings=train_settings,
            local_settings=local_settings,
            train_dir=args.train_dir,
            ckpt_file=args.checkpoint,
            resume=resume,
            pretraining=args.pretraining,
        )

    if complete:
        if args.exit_command:
            print(
                f"All training stages complete. Executing exit command: {args.exit_command}."
            )
            os.system(args.exit_command)
        else:
            print("All training stages complete.")
    else:
        print("Program terminated due to runtime limit.")
