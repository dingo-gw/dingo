import os
import sys
from os.path import join, isfile
import yaml
import argparse

from dingo.gw.training import (
    prepare_training_new,
    prepare_training_resume,
    train_stages,
)


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
    lines.append(f'arguments = {condor_settings["arguments"]}\n')
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


def train_condor():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dir", required=True, help="Directory for Dingo training output."
    )
    parser.add_argument("--checkpoint", default="model_latest.pt")
    parser.add_argument("--start_submission", action="store_true")
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

            pm, wfd = prepare_training_new(
                train_settings, args.train_dir, local_settings
            )

        else:
            print("Resuming training run.")
            with open(os.path.join(args.train_dir, "local_settings.yaml"), "r") as f:
                local_settings = yaml.safe_load(f)
            pm, wfd = prepare_training_resume(
                join(args.train_dir, args.checkpoint),
                local_settings,
                train_dir=args.train_dir,
            )

        complete = train_stages(pm, wfd, args.train_dir, local_settings)

        print("Copying log files")
        copy_logfiles(args.train_dir, epoch=pm.epoch)

        #
        # PREPARE NEXT SUBMISSION
        #

        if complete:
            print("Training complete, job will not be resubmitted.")
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
