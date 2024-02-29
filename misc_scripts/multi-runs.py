from typing import Dict, List, Optional
import argparse
import itertools
from functools import reduce
import operator
import os
import sys
from os.path import join

import yaml

""" This script was copied from 
    https://github.com/dingo-gw/dingo-development/blob/continuous_flows/continuous_flows/training/multi-runs.py
    and refactored.
    It works similarly to /misc_scripts/create_training_directories.py except that every run and its parameters have
    to be specified manually. However, this script automatically creates all combinations of specified hyperparameters
    and loops over them. It should be possible to combine both scripts.
    An example for the sweep_settings.yaml file can be found in /train_dir_parent/sweep_settings.yaml
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir", required=True, help="Base directory for creating the runs"
    )
    parser.add_argument(
        "--sweep_settings",
        default=None,
        help="Path to sweep_settings file which overrides the train_settings.yaml in the base_dir",
    )
    parser.add_argument(
        "--overwrite_wandb_name",
        default=True,
        help="Whether to overwrite the wandb name with the name of the experiment folder. Default True.",
    )
    parser.add_argument(
        "--create_evaluation_dir",
        action="store_true",
        help="Whether to create an evaluation directory within the training directories."
    )
    parser.add_argument(
        "--condor",
        action="store_true",
        help="Whether to submit the script with condor."
    )

    args = parser.parse_args()

    return args


def replace_values(target_dict: Dict, combination_dict: Dict):
    updated_dict = (
        target_dict.copy()
    )  # Create a copy to avoid modifying the original target_dict

    for key, value in combination_dict.items():
        # Traverse the nested dictionary using the key path
        current_dict = updated_dict
        for sub_key in key[:-1]:
            current_dict = current_dict[sub_key]

        # Replace the value in the nested dictionary
        current_dict[key[-1]] = value

    return updated_dict


def extract_final_layers(d, key_path: List = None):
    if key_path is None:
        key_path = []
    final_layers = {}

    for key, value in d.items():
        current_key_path = key_path + [key]

        # If the value is a dictionary, recurse and update the key_path
        if isinstance(value, dict):
            final_layers.update(extract_final_layers(value, current_key_path))
        else:
            # If the final layer is reached, store it in the final_layers dict with the key path as the key
            final_key = tuple(current_key_path)
            final_layers[final_key] = value

    return final_layers


def create_combinations(final_layer: Dict):
    # Extract lists from the final layer dictionary
    lists = [value for value in final_layer.values()]
    keys = [key for key in final_layer.keys()]

    # Create an outer product of the lists using itertools.product
    outer_product = itertools.product(*lists)

    # Combine the outer product with the keys to create a list of dictionaries
    combinations_dicts = [dict(zip(keys, values)) for values in outer_product]

    return combinations_dicts


def construct_train_command(
    train_dir: str, dir_with_settings_file: Optional[str], use_condor: bool
):
    arguments = f" --train_dir={train_dir}"

    if use_condor:
        train_script_path = sys.executable.replace("python", "") + "dingo_train_condor"
        arguments += f" --start_submission"
    else:
        train_script_path = sys.executable.replace("python", "") + "dingo_train"
        arguments += (
            f" --settings={join(dir_with_settings_file, 'train_settings.yaml')}"
        )
    command = train_script_path + arguments

    return command


def add_settings_to_modify_jointly(list_of_combination_dicts: List, joint_settings_dict: Dict, sweep_settings_dict: Dict):
    def get_by_path(root, items):
        """Access a nested object in root by item sequence."""
        return reduce(operator.getitem, items, root)

    # Check whether keys of joint_dict exist in combinations

    # Loop over keys in joint_dict
    for k, v in joint_settings_dict.items():
        # Get ordering of values for k in sweep_settings
        try:
            sweep_vals_k = get_by_path(sweep_settings_dict, k)
        except:
            raise ValueError(f"Key within 'joint_modification' which was read in as {k} does not exist in sweep_settings.")
        # Loop over combination dict
        for combination_dict in list_of_combination_dicts:
            # Get index of value to set
            ind_k = sweep_vals_k.index(combination_dict[k])
            # Set v_val[ind_k] for all elements of v
            for v_key, v_val in v.items():
                combination_dict[v_key] = v_val[ind_k]
    return list_of_combination_dicts


if __name__ == "__main__":
    args = parse_args()
    base_dir = args.base_dir

    if not args.sweep_settings:
        print(
            f"No sweep settings provided. Training a model based on the settings in {base_dir}."
        )

        with open(join(base_dir, "train_settings.yaml")) as f:
            train_settings = yaml.safe_load(f)

        command = construct_train_command(
            train_dir=base_dir,
            dir_with_settings_file=base_dir,
            use_condor=args.condor,
        )
        os.system(command)

    else:
        # We combinatorially combine the sweep settings and run a model for each combination. sweep_settings.yaml file
        # should contain a subset of the keys in the train_settings.yaml file. The values of the sweep_settings.yaml
        # should be lists of values that should be combined.

        with open(args.sweep_settings, "r") as f:
            sweep_settings = yaml.safe_load(f)

        # If the sweep settings contain the key "joint_modification", extract the information
        final_layer_dicts_joint = None
        if "joint_modification" in sweep_settings.keys():
            params_to_jointly_modify = sweep_settings.pop("joint_modification")
            final_layer_dicts_joint = {tuple(k.split("/")): extract_final_layers(v) for k, v in params_to_jointly_modify.items()}

        with open(join(base_dir, "train_settings.yaml"), "r") as f:
            base_settings = yaml.safe_load(f)

        final_layer_dict = extract_final_layers(sweep_settings)
        combination_dicts = create_combinations(final_layer_dict)

        # Add the joint settings
        if final_layer_dicts_joint is not None:
            combination_dicts = add_settings_to_modify_jointly(
                combination_dicts,
                final_layer_dicts_joint,
                sweep_settings
            )

        for i, settings in enumerate(combination_dicts):
            # set wandb run name to folder name
            folder_name = str(i).zfill(3)
            if args.overwrite_wandb_name and "wandb" in base_settings["local"].keys():
                print(f"Wandb name is overwritten by {folder_name}.")
                settings[("local", "wandb", "name")] = folder_name

            # create new dictionary that defaults to the base settings and replaces
            # keys that are contained in the run-specific settings:
            new_settings = replace_values(base_settings, settings)

            train_dir = join(base_dir, folder_name)
            os.makedirs(train_dir, exist_ok=True)
            if args.create_evaluation_dir:
                eval_dir = join(train_dir, "evaluation")
                os.makedirs(eval_dir, exist_ok=True)

            with open(join(train_dir, "train_settings.yaml"), "w") as outfile:
                yaml.dump(new_settings, outfile, default_flow_style=False)

            command = construct_train_command(
                train_dir=train_dir,
                dir_with_settings_file=train_dir,
                use_condor=args.condor,
            )
            os.system(command)
