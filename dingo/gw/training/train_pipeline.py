import argparse
import ctypes
import os
import queue
import shutil
import textwrap
import time
from copy import deepcopy
from multiprocessing import Value
from typing import Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from threadpoolctl import threadpool_limits
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from dingo.core.posterior_models.base_model import BasePosteriorModel
from dingo.core.posterior_models.build_model import (
    autocomplete_model_kwargs,
    build_model_from_kwargs,
)
from dingo.core.utils import (
    build_train_and_test_loaders,
    print_number_of_model_parameters,
    set_requires_grad_flag,
)
from dingo.core.utils.torchutils import (
    cleanup_ddp,
    document_gpus,
    replace_BatchNorm_with_SyncBatchNorm,
    set_seed_based_on_rank,
    setup_ddp,
)
from dingo.core.utils.trainutils import EarlyStopping, RuntimeLimits
from dingo.gw.dataset import WaveformDataset
from dingo.gw.training.train_builders import (
    build_dataset,
    build_svd_for_embedding_network,
    set_train_transforms,
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
        print(f"Copying file to {local_file_path}")
        # Copy file
        start_time = time.time()
        shutil.copy(file_path, local_file_path)
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


def prepare_training_new(
    train_settings: dict, train_dir: str, local_settings: dict
) -> Tuple[BasePosteriorModel, WaveformDataset]:
    """
    Based on a settings dictionary, initialize a WaveformDataset and PosteriorModel.

    For the standard DenseResNet embedding network type (the only acceptable type at this point) this also
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

    Returns
    -------
    (BasePosteriorModel, WaveformDataset)
    """
    wfd, initial_weights = _prepare_wfd_and_initial_weights(
        train_settings=train_settings,
        train_dir=train_dir,
        local_settings=local_settings,
    )

    # This modifies the model settings in-place.
    autocomplete_model_kwargs(train_settings["model"], wfd[0])
    full_settings = {
        "dataset_settings": wfd.settings,
        "train_settings": train_settings,
    }

    print("\nInitializing new posterior model.")
    print("Complete settings:")
    print(yaml.dump(full_settings, default_flow_style=False, sort_keys=False))

    pm = build_model_from_kwargs(
        settings=full_settings,
        initial_weights=initial_weights,
        device=local_settings["device"],
    )

    if local_settings.get("wandb", False):
        try:
            import wandb

            wandb.init(
                config=full_settings,
                dir=train_dir,
                **local_settings["wandb"],
            )
        except ImportError:
            print("WandB is enabled but not installed.")

    return pm, wfd


def prepare_training_resume(
    checkpoint_name: str, local_settings: dict, train_dir: str
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
        Local settings (e.g., num_workers, device)
    train_dir: str
        Path to training directory where the wandb info is saved.

    Returns
    -------
    (BasePosteriorModel, WaveformDataset)
    """

    pm = build_model_from_kwargs(
        filename=checkpoint_name, device=local_settings["device"]
    )
    data_settings = deepcopy(pm.metadata["train_settings"]["data"])
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

    return pm, wfd


def initialize_stage(
    pm: BasePosteriorModel,
    wfd: WaveformDataset,
    stage: dict,
    num_workers: int,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    resume: bool = False,
) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler]]:
    """
    Initializes training based on PosteriorModel metadata and current stage:
        * Builds transforms (based on noise settings for current stage);
        * Builds DataLoaders (with DistributedSampler for DDP);
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
    world_size : int, optional
        Total number of DDP processes (GPUs).
    rank : int, optional
        Rank of this DDP process.
    resume : bool
        Whether training is resuming mid-stage. This controls whether the optimizer and
        scheduler should be re-initialized based on contents of stage dict.

    Returns
    -------
    (train_loader, test_loader, train_sampler)
        *train_sampler* is ``None`` in single-GPU mode.
    """

    train_settings = pm.metadata["train_settings"]
    print_output = rank is None or rank == 0

    # Rebuild transforms based on possibly different noise.
    set_train_transforms(
        wfd,
        train_settings["data"],
        stage["asd_dataset_path"],
        print_output=print_output,
    )

    # Convert total batch size to per-GPU batch size for DDP.
    if world_size is not None and world_size > 1:
        total_batch_size = stage["batch_size"]
        if total_batch_size % world_size != 0:
            raise ValueError(
                f"Total batch size {total_batch_size} is not divisible by "
                f"the number of GPUs {world_size}."
            )
        batch_size_per_gpu = total_batch_size // world_size
    else:
        batch_size_per_gpu = stage["batch_size"]

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
        pm.initialize_optimizer_and_scheduler()

    # Freeze/unfreeze RB layer if necessary
    if "freeze_rb_layer" in stage:
        if stage["freeze_rb_layer"]:
            if world_size is not None and world_size > 1:
                raise ValueError(
                    "Cannot freeze the RB layer during DDP training — modify "
                    "the network before wrapping it with DDP."
                )
            set_requires_grad_flag(
                pm.network, name_contains="layers_rb", requires_grad=False
            )
        else:
            set_requires_grad_flag(
                pm.network, name_contains="layers_rb", requires_grad=True
            )

    if print_output:
        print_number_of_model_parameters(pm.network)

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
    global_epoch : multiprocessing.Value
        Shared epoch counter, updated each epoch and forwarded to ``pm.train``.

    Returns
    -------
    (training_finished, resume_training) : (bool, bool)
    """

    train_settings = pm.metadata["train_settings"]
    runtime_limits = RuntimeLimits(
        epoch_start=pm.epoch,
        device=pm.device,
        **local_settings["runtime_limits"],
    )

    rank = local_settings.get("rank", None)
    world_size = local_settings.get("world_size", None)
    print_primary = rank is None or rank == 0

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
            if print_primary:
                print(f"\nBeginning training stage {n}. Settings:")
                print(yaml.dump(stage, default_flow_style=False, sort_keys=False))
            train_loader, test_loader, train_sampler = initialize_stage(
                pm,
                wfd,
                stage,
                local_settings["num_workers"],
                world_size=world_size,
                rank=rank,
                resume=False,
            )
        else:
            if print_primary:
                print(f"\nResuming training in stage {n}. Settings:")
                print(yaml.dump(stage, default_flow_style=False, sort_keys=False))
            train_loader, test_loader, train_sampler = initialize_stage(
                pm,
                wfd,
                stage,
                local_settings["num_workers"],
                world_size=world_size,
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
            train_dir=train_dir,
            train_sampler=train_sampler,
            runtime_limits=runtime_limits,
            checkpoint_epochs=local_settings["checkpoint_epochs"],
            use_wandb=local_settings.get("wandb", False),
            test_only=local_settings.get("test_only", False),
            early_stopping=early_stopping,
            gradient_updates_per_optimizer_step=stage.get(
                "gradient_updates_per_optimizer_step", 1
            ),
            automatic_mixed_precision=stage.get("automatic_mixed_precision", False),
            world_size=world_size if world_size is not None else 1,
            global_epoch=global_epoch,
        )

        # if test_only, model should not be saved, and run is complete
        if local_settings.get("test_only", False):
            return True, False

        if pm.epoch == end_epochs[n] and print_primary:
            save_file = os.path.join(train_dir, f"model_stage_{n}.pt")
            print(f"Training stage complete. Saving to {save_file}.")
            pm.save_model(save_file, save_training_info=True)

        if runtime_limits.local_limits_exceeded(pm.epoch):
            if print_primary:
                print("Local runtime limits reached. Ending program.")
            resume_training = True
            break

    training_finished = pm.epoch == end_epochs[-1]
    return training_finished, resume_training


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
    args = parser.parse_args()

    # The settings file and checkpoint are mutually exclusive.
    if args.checkpoint is None and args.settings_file is None:
        parser.error("Must specify either a checkpoint file or a settings file.")
    if args.checkpoint is not None and args.settings_file is not None:
        parser.error("Cannot specify both a checkpoint file and a settings file.")

    return args


def get_num_gpus(local_settings: dict) -> int:
    """
    Return the number of GPUs to use for training.

    The canonical location is ``local_settings["num_gpus"]`` (default: 1).

    For backward compatibility, ``local_settings["condor"]["num_gpus"]`` is
    accepted as a fallback when ``local_settings["num_gpus"]`` is absent, but
    its use is deprecated and will be removed in a future release.
    """
    if "num_gpus" in local_settings:
        return int(local_settings["num_gpus"])
    condor_num_gpus = local_settings.get("condor", {}).get("num_gpus")
    if condor_num_gpus is not None:
        import warnings

        warnings.warn(
            "Specifying 'num_gpus' under 'local.condor' is deprecated. "
            "Move it to 'local.num_gpus' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return int(condor_num_gpus)
    return 1


def run_training(
    train_settings: Optional[dict],
    local_settings: dict,
    train_dir: str,
    ckpt_file: Optional[str],
    resume: bool,
) -> Tuple[bool, bool, int]:
    """
    Single-GPU training entry point.

    Parameters
    ----------
    train_settings : dict or None
        Full training settings (from YAML).  ``None`` when resuming.
    local_settings : dict
    train_dir : str
    ckpt_file : str or None
        Checkpoint path used when *resume* is True.
    resume : bool

    Returns
    -------
    (complete, resume, epoch) : (bool, bool, int)
    """
    if not resume:
        pm, wfd = prepare_training_new(train_settings, train_dir, local_settings)
    else:
        pm, wfd = prepare_training_resume(ckpt_file, local_settings, train_dir)

    global_epoch = Value(ctypes.c_int, pm.epoch)
    wfd.epoch = global_epoch

    with threadpool_limits(limits=1, user_api="blas"):
        complete, resume_flag = train_stages(
            pm=pm,
            wfd=wfd,
            train_dir=train_dir,
            local_settings=local_settings,
            global_epoch=global_epoch,
        )

    return complete, resume_flag, pm.epoch


def run_training_ddp(
    rank: int,
    world_size: int,
    train_settings: Optional[dict],
    local_settings: dict,
    train_dir: str,
    wfd: WaveformDataset,
    initial_weights: Optional[dict],
    ckpt_file: Optional[str],
    resume: bool,
    result_queue: mp.Queue,
    global_epoch: ctypes.c_int,
) -> None:
    """
    Worker function executed by each DDP process.

    Each GPU runs this function in its own process, identified by *rank*.
    Results (complete flag, resume flag, epoch) are put into *result_queue*
    so the parent process can collect them.

    Parameters
    ----------
    rank : int
    world_size : int
    train_settings : dict or None
    local_settings : dict
    train_dir : str
    wfd : WaveformDataset
        Dataset object pre-built in the parent process and shared with workers.
    initial_weights : dict or None
        SVD-based initial weights for the embedding network.
    ckpt_file : str or None
    resume : bool
    result_queue : mp.Queue
    global_epoch : multiprocessing.Value
    """
    try:
        setup_ddp(rank, world_size)
        set_seed_based_on_rank(rank)

        if rank == 0:
            document_gpus(train_dir)

        local_settings = dict(local_settings)  # avoid mutating the original
        local_settings["device"] = f"cuda:{rank}"
        local_settings["rank"] = rank
        local_settings["world_size"] = world_size

        if not resume:
            full_settings = {
                "dataset_settings": wfd.settings,
                "train_settings": train_settings,
            }
            print_output = rank == 0
            if print_output:
                print("\nInitializing new posterior model.")
                print("Complete settings:")
                print(
                    yaml.dump(full_settings, default_flow_style=False, sort_keys=False)
                )

            pm = build_model_from_kwargs(
                settings=full_settings,
                initial_weights=initial_weights,
                device=local_settings["device"],
            )

            if rank == 0 and local_settings.get("wandb", False):
                try:
                    import wandb

                    wandb.init(
                        config=full_settings,
                        dir=train_dir,
                        **local_settings["wandb"],
                    )
                except ImportError:
                    print("WandB is enabled but not installed.")
        else:
            pm = build_model_from_kwargs(
                filename=ckpt_file, device=local_settings["device"]
            )
            if rank == 0 and local_settings.get("wandb", False):
                try:
                    import wandb

                    wandb.init(
                        resume="must",
                        dir=train_dir,
                        **local_settings["wandb"],
                    )
                except ImportError:
                    print("WandB is enabled but not installed.")

        pm.network = replace_BatchNorm_with_SyncBatchNorm(pm.network)
        pm.network = DDP(pm.network, device_ids=[rank])

        global_epoch.value = pm.epoch
        wfd.epoch = global_epoch

        with threadpool_limits(limits=1, user_api="blas"):
            complete, resume_flag = train_stages(
                pm=pm,
                wfd=wfd,
                train_dir=train_dir,
                local_settings=local_settings,
                global_epoch=global_epoch,
            )

        if complete and local_settings.get("wandb", False) and rank == 0:
            try:
                import wandb

                wandb.finish()
            except ImportError:
                pass

        cleanup_ddp()
        result_queue.put((rank, complete, resume_flag, pm.epoch, None))

    except Exception as exc:
        import traceback

        result_queue.put((rank, False, False, 0, traceback.format_exc()))
        raise


def run_multi_gpu_training(
    world_size: int,
    train_settings: Optional[dict],
    local_settings: dict,
    train_dir: str,
    ckpt_file: Optional[str],
    resume: bool,
) -> Tuple[bool, bool, int]:
    """
    Multi-GPU DDP training entry point.

    Spawns one process per GPU, each running :func:`run_training_ddp`.

    Parameters
    ----------
    world_size : int
        Number of GPUs.
    train_settings : dict or None
    local_settings : dict
    train_dir : str
    ckpt_file : str or None
    resume : bool

    Returns
    -------
    (complete, resume, epoch) : (bool, bool, int)
    """
    initial_weights = None

    if not resume:
        # Build the dataset and compute SVD-based initialisation *once* in the
        # parent process so it can be shared across all worker processes.
        wfd, initial_weights = _prepare_wfd_and_initial_weights(
            train_settings=train_settings,
            train_dir=train_dir,
            local_settings=local_settings,
        )
        autocomplete_model_kwargs(
            model_kwargs=train_settings["model"],
            data_sample=wfd[0],
        )
    else:
        d = torch.load(ckpt_file, map_location="cpu")
        train_settings = d["metadata"]["train_settings"]
        data_settings = deepcopy(train_settings["data"])
        data_settings["waveform_dataset_path"] = copy_files_to_local(
            file_path=data_settings["waveform_dataset_path"],
            local_dir=local_settings.get("local_cache_path", None),
            leave_keys_on_disk=local_settings.get("leave_waveforms_on_disk", True),
            is_condor="condor" in local_settings,
        )
        wfd = build_dataset(
            data_settings=data_settings,
            leave_waveforms_on_disk=local_settings.get("leave_waveforms_on_disk", True),
        )

    global_epoch = Value(ctypes.c_int, 0)
    result_queue = mp.Queue()
    processes = []

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
                ckpt_file,
                resume,
                result_queue,
                global_epoch,
            ),
        )
        p.start()
        processes.append(p)

    error_occurred = False
    for p in processes:
        p.join()
        if p.exitcode != 0:
            error_occurred = True
            print(f"Process {p.pid} exited with code {p.exitcode}.")
            for proc in processes:
                if proc.is_alive():
                    proc.terminate()
            break

    results = []
    try:
        while True:
            item = result_queue.get_nowait()
            results.append(item)
            if item[-1] is not None:
                print(f"Rank {item[0]} failed:\n{item[-1]}")
            else:
                print(f"Rank {item[0]} finished successfully.")
    except queue.Empty:
        pass

    if error_occurred or len(results) != world_size:
        raise RuntimeError(
            f"DDP training failed: {len(results)}/{world_size} processes reported results."
        )

    complete_flags = [r[1] for r in results]
    resume_flags = [r[2] for r in results]
    epochs = [r[3] for r in results]

    assert (
        len(set(complete_flags)) == 1
    ), f"Inconsistent 'complete' flags: {complete_flags}"
    assert len(set(resume_flags)) == 1, f"Inconsistent 'resume' flags: {resume_flags}"
    assert len(set(epochs)) == 1, f"Inconsistent epoch counts: {epochs}"

    return complete_flags[0], resume_flags[0], epochs[0]


def _prepare_wfd_and_initial_weights(
    train_settings: dict,
    train_dir: str,
    local_settings: dict,
) -> Tuple[WaveformDataset, Optional[dict]]:
    """
    Build the WaveformDataset and, if applicable, compute SVD-based initial
    weights for the embedding network.

    This is called once in the parent process before spawning DDP workers so
    that the expensive SVD computation and dataset loading happen only once.

    Returns
    -------
    (wfd, initial_weights)
    """
    data_settings = deepcopy(train_settings["data"])
    data_settings["waveform_dataset_path"] = copy_files_to_local(
        file_path=data_settings["waveform_dataset_path"],
        local_dir=local_settings.get("local_cache_path", None),
        leave_keys_on_disk=local_settings.get("leave_waveforms_on_disk", True),
        is_condor="condor" in local_settings,
    )
    wfd = build_dataset(
        data_settings=data_settings,
        leave_waveforms_on_disk=local_settings.get("leave_waveforms_on_disk", True),
    )

    initial_weights = {}
    model_kwargs = train_settings["model"]

    if (
        model_kwargs.get("embedding_kwargs")
        and "svd" in model_kwargs["embedding_kwargs"]
        and not model_kwargs["embedding_kwargs"]["svd"].get("no_init", False)
    ):
        batch_size = train_settings["training"]["stage_0"]["batch_size"]
        print("\nBuilding SVD for initialization of embedding network.")
        initial_weights["V_rb_list"] = build_svd_for_embedding_network(
            wfd=wfd,
            data_settings=train_settings["data"],
            asd_dataset_path=train_settings["training"]["stage_0"]["asd_dataset_path"],
            batch_size=batch_size,
            out_dir=train_dir,
            **model_kwargs["embedding_kwargs"]["svd"],
        )
    else:
        initial_weights = None

    set_train_transforms(
        wfd,
        train_settings["data"],
        train_settings["training"]["stage_0"]["asd_dataset_path"],
    )

    return wfd, initial_weights


def train_local():
    args = parse_args()

    os.makedirs(args.train_dir, exist_ok=True)

    resume = args.checkpoint is not None

    if not resume:
        print("Beginning new training run.")
        with open(args.settings_file, "r") as fp:
            train_settings = yaml.safe_load(fp)

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

    num_gpus = get_num_gpus(local_settings)
    if local_settings.get("device") == "cuda" and num_gpus > 1:
        complete, resume_flag, _ = run_multi_gpu_training(
            world_size=num_gpus,
            train_settings=train_settings,
            local_settings=local_settings,
            train_dir=args.train_dir,
            ckpt_file=args.checkpoint,
            resume=resume,
        )
    else:
        if local_settings.get("device") == "cuda":
            document_gpus(args.train_dir)
        complete, resume_flag, _ = run_training(
            train_settings=train_settings,
            local_settings=local_settings,
            train_dir=args.train_dir,
            ckpt_file=args.checkpoint,
            resume=resume,
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
