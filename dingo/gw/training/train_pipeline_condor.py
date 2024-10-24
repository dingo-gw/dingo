import os
import sys
from os.path import join, isfile
import yaml
import argparse
import shutil
import time

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.modules import Module

from dingo.gw.training import (
    prepare_wfd_and_initialization_for_embedding_network,
    prepare_model_new,
    load_settings_from_ckpt,
    prepare_model_resume,
    prepare_training_new,
    prepare_training_resume,
    train_stages,
)
from dingo.gw.training.train_builders import build_dataset
from dingo.gw.dataset.waveform_dataset import WaveformDataset
from dingo.core.utils.environment import document_environment
from dingo.core.utils.torchutils import (
    document_gpus,
    setup_ddp,
    cleanup_ddp,
    replace_BatchNorm_with_SyncBatchNorm,
    set_seed_based_on_rank,
)


def create_submission_file(
    train_dir: str, condor_settings: dict, filename: str = "submission_file.sub"
):
    """
    Creates submission file and writes it to filename.

    Parameters
    ----------
    train_dir: str
        Path to training directory
    condor_settings: dict
        Condor settings
    filename: str
        Filename of submission file
    """
    lines = []
    lines.append(f'executable = {condor_settings["executable"]}\n')
    lines.append(f'request_cpus = {condor_settings["num_cpus"]}\n')
    lines.append(f'request_memory = {condor_settings["memory_cpus"]}\n')
    lines.append(f'request_gpus = {condor_settings["num_gpus"]}\n')
    lines.append(
        f"requirements = TARGET.CUDAGlobalMemoryMb > "
        f'{condor_settings["memory_gpus"]}\n\n'
    )
    if condor_settings["num_gpus"] == 8:
        # Request full node
        lines.append("use template : FullNode\n")
    elif condor_settings["num_gpus"] >= 6:
        # Still request full nodes because wait times are long
        lines.append(f'use template : FullNode({condor_settings["num_gpus"]})\n')

    lines.append(f'arguments = "{condor_settings["arguments"]}"\n')
    lines.append(f'error = {join(train_dir, "info.err")}\n')
    lines.append(f'output = {join(train_dir, "info.out")}\n')
    lines.append(f'log = {join(train_dir, "info.log")}\n')
    lines.append("queue")

    with open(join(train_dir, filename), "w") as f:
        for line in lines:
            f.write(line)


def copyfile(src, dst):
    os.system("cp -p %s %s" % (src, dst))


def copy_logfiles(log_dir, epoch, name="info", suffixes=(".err", ".log", ".out")):
    for suffix in suffixes:
        src = join(log_dir, name + suffix)
        dest = join(log_dir, name + "_{:03d}".format(epoch) + suffix)
        try:
            copyfile(src, dest)
        except:
            print("Could not copy " + src)


def run_training(
    train_settings: dict,
    local_settings: dict,
    train_dir: str,
    ckpt_file: str,
    resume: bool,
    pretraining: bool = False,
) -> (bool, int):
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
        Whether the training run was successful
    epoch: int
        The epoch number where the training finished
    """
    if not resume:
        pm, wfd = prepare_training_new(train_settings, train_dir, local_settings)
    else:
        pm, wfd = prepare_training_resume(
            checkpoint_name=ckpt_file,
            local_settings=local_settings,
            train_dir=train_dir,
        )

    complete = train_stages(pm, wfd, train_dir, local_settings)

    return complete, pm.epoch


def run_multi_gpu_training(
    world_size: int,
    train_settings: dict | None,
    local_settings: dict,
    train_dir: str,
    ckpt_file: str,
    resume: bool,
    pretraining: bool = False,
) -> (bool, int):
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
        Whether the training run was successful
    epoch: int
        The epoch number where the training finished
    """
    # Copy waveform dataset to local node to minimize network traffic during training
    wfd_path = train_settings["data"]["waveform_dataset_path"]
    file_name = wfd_path.split("/")[-1]
    wfd_path_tmp = join("/tmp", file_name)
    print("Copying waveform dataset to {}".format(wfd_path_tmp))
    start_time = time.time()
    shutil.copy(wfd_path, wfd_path_tmp)
    elapsed_time = time.time() - start_time
    print("Done. This took {:2.0f}:{:2.0f} min.".format(*divmod(elapsed_time, 60)))
    # Overwrite waveform_dataset_path
    train_settings["data"]["waveform_dataset_path"] = wfd_path_tmp

    initial_weights, pretrained_emb_net, checkpoint_file = None, None, None
    if not resume:
        (
            wfd,
            initial_weights,
            pretrained_emb_net,
        ) = prepare_wfd_and_initialization_for_embedding_network(
            train_settings, train_dir, local_settings
        )
    else:
        checkpoint_file = os.path.join(train_dir, ckpt_file)
        train_settings = load_settings_from_ckpt(ckpt_file)
        wfd = build_dataset(train_settings["data"])
    if pretrained_emb_net is not None:
        pretraining = True
    else:
        pretraining = False

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
    while not result_queue.empty():
        temp_result = result_queue.get()
        if temp_result[1] is False:
            print(f"Rank {temp_result[0]} failed with error: {temp_result[3]}")
        else:
            print(f"Rank {temp_result[0]} completed successfully.")

    if error_occurred:
        raise RuntimeError("One or more processes failed, check info.out for details.")

    # Collect exit results from all processes after training
    complete, pm_epoch = [], []
    for _ in range(world_size):
        temp_result = result_queue.get()
        complete.append(temp_result[1])
        pm_epoch.append(temp_result[2])
    assert all(complete) is True, f"Not all processes exited successfully: {complete}."
    assert (
        len(set(pm_epoch)) == 1
    ), f"Processes do not return the same epochs: {pm_epoch}."

    return complete[0], pm_epoch[0]


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

        # Replace BatchNorm layers with SyncBatchNorm
        pm.network = replace_BatchNorm_with_SyncBatchNorm(pm.network)
        # Wrap the model with DDP
        pm.network = DDP(pm.network, device_ids=[rank])

        complete = train_stages(pm, wfd, train_dir, local_settings)

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
        result_queue.put((rank, False, 0, str(e)))
        sys.exit(1)  # Exit with a non-zero code to indicate failure

    # Put return info on queue
    result_queue.put((rank, complete, pm.epoch, None))
    sys.exit(0)  # Exit with zero to indicate success


def train_condor():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dir", required=True, help="Directory for Dingo training output."
    )
    parser.add_argument("--checkpoint", default="model_latest.pt")
    parser.add_argument("--start_submission", action="store_true")
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

    # For condor settings, first try looking for a local settings file. Otherwise,
    # defer to train_settings.yaml.
    # if isfile(join(args.train_dir, 'local_settings.yaml')):
    #     with open(join(args.train_dir, 'local_settings.yaml')) as f:
    #         condor_settings = yaml.safe_load(f)['condor']
    # else:

    if not args.start_submission:
        #
        # TRAIN
        #

        # Document setup
        document_environment(args.train_dir)
        # Cannot document GPU info here because this results in problems with mp.Process

        if not isfile(join(args.train_dir, args.checkpoint)):
            print("Beginning new training run.")
            resume = False
            with open(join(args.train_dir, "train_settings.yaml"), "r") as f:
                train_settings = yaml.safe_load(f)

            # Extract the local settings from train settings file, save it separately.
            # This file can later be modified, and the settings take effect immediately
            # upon resuming.

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
            resume = True
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

            complete, pm_epoch = run_multi_gpu_training(
                world_size,
                train_settings,
                local_settings,
                args.train_dir,
                args.checkpoint,
                resume,
                args.pretraining,
            )
        else:
            document_gpus(args.train_dir)
            complete, pm_epoch = run_training(
                train_settings,
                local_settings,
                args.train_dir,
                args.checkpoint,
                resume,
                args.pretraining,
            )

        print("Copying log files")
        copy_logfiles(args.train_dir, epoch=pm_epoch)

        #
        # PREPARE NEXT SUBMISSION
        #

        if complete:
            print(
                f"Training complete, job will not be resubmitted. Executing exit command: {args.exit_command}."
            )
            if args.exit_command:
                os.system(args.exit_command)
            sys.exit()

        else:
            condor_arguments = f"--train_dir {args.train_dir}"

    else:
        #
        # PREPARE FIRST SUBMISSION
        #

        condor_arguments = f"--train_dir {args.train_dir}"
        if args.checkpoint != "model_latest.pt":
            condor_arguments += f" --checkpoint {args.checkpoint}"

    if args.exit_command:
        condor_arguments += f" --exit_command '{args.exit_command}'"

    submission_file = "submission_file.sub"
    with open(join(args.train_dir, "train_settings.yaml"), "r") as f:
        condor_settings = yaml.safe_load(f)["local"]["condor"]
    condor_settings["arguments"] = condor_arguments
    condor_settings["executable"] = join(
        os.path.dirname(sys.executable), "dingo_train_condor"
    )
    create_submission_file(args.train_dir, condor_settings, submission_file)

    #
    # SUBMIT NEXT CONDOR JOB
    #

    if "bid" in condor_settings:
        # This is a specific setting for the MPI-IS cluster.
        bid = condor_settings["bid"]
        os.system(
            f"condor_submit_bid {bid} " f"{join(args.train_dir, submission_file)}"
        )
    else:
        os.system(f"condor_submit {join(args.train_dir, submission_file)}")


if __name__ == "__main__":
    train_condor()
