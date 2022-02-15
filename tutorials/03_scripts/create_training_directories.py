import os
from os.path import join
import copy
import yaml
import argparse


def none_or_str(value):
    if value == "None":
        return None
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Start dingo training runs.")
    parser.add_argument(
        "--parent_directory",
        type=str,
        required=True,
        help="Parent directory for training runs. The individual train directories are "
        "created in this directory.",
    )
    parser.add_argument(
        "--base_settings",
        type=str,
        default=None,
        help="Base settings file for training.",
    )
    parser.add_argument(
        "--update_settings",
        type=str,
        default=None,
        help="Individual updated settings for training runs.",
    )
    parser.add_argument(
        "--submission_command",
        type=none_or_str,
        default="dingo_train_condor --start_submission --train_dir <train_dir>",
        help="Submission command for training. By default, <train_dir> will be replaced "
        "with the actual train directory.",
    )

    args = parser.parse_args()

    if args.base_settings is None:
        args.base_settings = join(args.parent_directory, "base_settings.yaml")
    if args.update_settings is None:
        args.update_settings = join(args.parent_directory, "update_settings.yaml")

    return args


def merge_dicts_recursive(base_dict, update_dict, parent_keys=None):
    if parent_keys is None:
        parent_keys = []
    if update_dict is not None:
        for key, val in update_dict.items():
            assert (
                key in base_dict
            ), f"No element {'->'.join(parent_keys + [key])} in base dict."
            if type(val) == dict:
                merge_dicts_recursive(base_dict[key], val, parent_keys + [key])
            else:
                base_dict[key] = val


def create_new_train_dir_with_settings(base_settings, update_settings_run, train_dir):
    new_settings = copy.deepcopy(base_settings)
    merge_dicts_recursive(new_settings, update_settings_run)
    os.makedirs(train_dir, exist_ok=True)
    with open(join(train_dir, "train_settings.yaml"), "w") as fp:
        yaml.dump(new_settings, fp, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    args = parse_args()

    # load base settings file
    with open(args.base_settings, "r") as fp:
        base_settings = yaml.safe_load(fp)

    # load update settings file
    with open(args.update_settings, "r") as fp:
        update_settings = yaml.safe_load(fp)

    runs = update_settings.keys()
    for run in runs:
        train_dir = join(args.parent_directory, str(run))
        print(f"Creating train directory {str(run)}")
        # create train dir with updated settings file
        create_new_train_dir_with_settings(
            base_settings,
            update_settings[run],
            train_dir,
        )
        # optionally submit training run
        if args.submission_command is not None:
            cmd = args.submission_command.replace("<train_dir>", train_dir)
            os.system(cmd)
