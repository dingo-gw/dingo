import argparse
import os
import textwrap

import yaml

from dingo.populations.population_models import build_population_model
from dingo.populations.training.transform_builders import set_train_transforms


def prepare_training_new(train_settings: dict, train_dir: str, local_settings: dict):
    # (1) Build population forward models (train and test). These use different halves
    # of the base population events.

    population_model_train = build_population_model(
        train_settings["data"], mode="train"
    )
    population_model_test = build_population_model(train_settings["data"], mode="test")

    # (2) Build dataloaders
    set_train_transforms(population_model_train, train_settings["data"])

    # (3) Build model
    # autocomplete_model_kwargs(train_settings["model"], population_model_train[0])

    full_settings = {}

    # Return dataloaders and model
    return


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
        Train a neural network for population inference.

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
        "--train_dir", required=True, help="Directory for training output."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint file from which to resume training.",
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

    if args.settings_file is not None:
        print("Beginning new training run.")
        with open(args.settings_file, "r") as fp:
            train_settings = yaml.safe_load(fp)

        # Extract the local settings from train settings file, save it separately. This
        # file can later be modified, and the settings take effect immediately upon
        # resuming.

        local_settings = train_settings.pop("local")
        with open(os.path.join(args.train_dir, "local_settings.yaml"), "w") as f:
            yaml.dump(local_settings, f, default_flow_style=False, sort_keys=False)

        prepare_training_new(train_settings, args.train_dir, local_settings)

    # else:
    #     print("Resuming training run.")
    #     with open(os.path.join(args.train_dir, "local_settings.yaml"), "r") as f:
    #         local_settings = yaml.safe_load(f)
    #     pm, wfd = prepare_training_resume(
    #         args.checkpoint, local_settings, args.train_dir
    #     )
    #
    # with threadpool_limits(limits=1, user_api="blas"):
    #     complete = train(pm, wfd, args.train_dir, local_settings)
    #
    # if complete:
    #     print("All training stages complete.")
    # else:
    #     print("Program terminated due to runtime limit.")


if __name__ == "__main__":
    train_local()
