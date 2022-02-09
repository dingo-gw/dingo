import os
import sys
from os.path import join, isfile
import yaml
import argparse

from dingo.gw.dataset.generate_dataset_dag import create_args_string
from dingo.gw.training import (
    prepare_training_new,
    prepare_training_resume,
    train_stages,
)


def create_submission_file(train_dir, d, kwargs_dict, filename="submission_file.sub"):
    """
    TODO: documentation
    :param train_dir:
    :param filename:
    :return:
    """
    lines = []
    lines.append(f'executable = {d["python"]}\n')
    lines.append(f'request_cpus = {d["num_cpus"]}\n')
    lines.append(f'request_memory = {d["memory_cpus"]}\n')
    lines.append(f'request_gpus = {d["num_gpus"]}\n')
    lines.append(
        f"requirements = TARGET.CUDAGlobalMemoryMb > " f'{d["memory_gpus"]}\n\n'
    )
    kwargs_str = create_args_string(kwargs_dict)
    lines.append(f'arguments = {d["train_script"]} {kwargs_str}\n')
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
                yaml.dump(local_settings, f, default_flow_style=False, sort_keys=False)

            pm, wfd = prepare_training_new(
                train_settings, args.train_dir, local_settings
            )

        else:
            print("Resuming training run.")
            with open(os.path.join(args.train_dir, "local_settings.yaml"), "r") as f:
                local_settings = yaml.safe_load(f)
            pm, wfd = prepare_training_resume(
                join(args.train_dir, args.checkpoint), local_settings["device"]
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
            kwargs_dict = {
                "train_dir": args.train_dir,
                "checkpoint": "model_latest.py",
            }

    else:

        #
        # PREPARE FIRST SUBMISSION
        #

        kwargs_dict = {
            "train_dir": args.train_dir,
            "checkpoint": args.checkpoint,
        }

    submission_file = "submission_file.sub"
    with open(join(args.train_dir, "train_settings.yaml"), "r") as f:
        condor_settings = yaml.safe_load(f)["local"]["condor"]
    create_submission_file(
        args.train_dir, condor_settings, kwargs_dict, filename=submission_file
    )

    #
    # SUBMIT NEXT CONDOR JOB
    #

    # There was no 'bid' in the sample settings file.
    bid = condor_settings.get("bid", "")
    os.system(f"condor_submit_bid {bid} " f"{join(args.train_dir, submission_file)}")
