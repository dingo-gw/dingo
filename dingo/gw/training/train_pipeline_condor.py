import os
import sys
from os.path import join, isfile
import yaml
import argparse

import torch.multiprocessing as mp

from dingo.gw.training import (
    prepare_training_new,
    prepare_training_resume,
    train_stages,
)
from dingo.core.utils.torchutils import setup_ddp, cleanup_ddp, replace_BatchNorm_with_SyncBatchNorm, set_seed_based_on_rank
from torch.nn.parallel import DistributedDataParallel as DDP


def create_submission_file(train_dir, condor_settings, filename="submission_file.sub"):
    """
    TODO: documentation
    :param train_dir:
    :param filename:
    :return:
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
    lines.append(f'arguments = \"{condor_settings["arguments"]}\"\n')
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
        checkpoint: str,
        resume: bool,
):
    if not resume:
        pm, wfd = prepare_training_new(
            train_settings, train_dir, local_settings
        )
    else:
        ckpt_file = os.path.join(train_dir, checkpoint)
        pm, wfd = prepare_training_resume(
            checkpoint_name=ckpt_file,
            local_settings=local_settings,
            train_dir=train_dir,
        )

    complete = train_stages(pm, wfd, train_dir, local_settings)

    return complete, pm.epoch


def run_training_ddp(
        rank: int,
        world_size: int,
        train_settings: dict,
        local_settings: dict,
        train_dir: str,
        checkpoint: str,
        resume: bool,
):
    # Initialize process group
    setup_ddp(rank, world_size)
    # Set seeds for each GPU to different value
    set_seed_based_on_rank(rank)

    local_settings["device"] = f"cuda:{rank}"
    local_settings["rank"] = rank
    local_settings["world_size"] = world_size

    if not resume:
        pm, wfd = prepare_training_new(
            train_settings, train_dir, local_settings
        )
    else:
        ckpt_file = os.path.join(train_dir, checkpoint)
        pm, wfd = prepare_training_resume(
            checkpoint_name=ckpt_file,
            local_settings=local_settings,
            train_dir=train_dir,
        )

    # Replace BatchNorm layers with SyncBatchNorm
    pm.network = replace_BatchNorm_with_SyncBatchNorm(pm.network)
    # Wrap the model with DDP
    pm.network = DDP(pm.network, device_ids=[rank])

    complete = train_stages(pm, wfd, train_dir, local_settings)

    if complete and local_settings.get("wandb", False) and rank == 0.:
        try:
            import wandb

            wandb.finish()
        except ImportError:
            print("wandb not installed. Skipping logging to wandb.")

    # Delete process group
    cleanup_ddp()

    return complete, pm.epoch


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
            with open(os.path.join(args.train_dir, "local_settings.yaml"), "r") as f:
                local_settings = yaml.safe_load(f)

        # Setup multi-GPU training
        if (local_settings["device"] == "cuda"
                and "condor" in local_settings
                and "num_gpus" in local_settings["condor"]
                and local_settings["condor"]["num_gpus"] > 1.):
            # Specify the number of processes (typically the number of GPUs available)
            world_size = local_settings["condor"]["num_gpus"]

            complete, pm_epoch = mp.spawn(run_training_ddp,
                                args=(
                                    world_size, train_settings, local_settings, args.train_dir, args.checkpoint,
                                    resume),
                                nprocs=world_size,
                                join=True)
        else:
            complete, pm_epoch = run_training(train_settings, local_settings, args.train_dir, args.checkpoint, resume)

        print("Copying log files")
        copy_logfiles(args.train_dir, epoch=pm_epoch)

        #
        # PREPARE NEXT SUBMISSION
        #

        if complete:
            print(f"Training complete, job will not be resubmitted. Executing exit command: {args.exit_command}.")
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
