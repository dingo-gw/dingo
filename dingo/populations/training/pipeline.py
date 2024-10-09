import argparse
import os
import textwrap

import torch
import yaml
import multiprocessing
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader

from dingo.core.models import PosteriorModel
from dingo.core.utils import (
    fix_random_seeds,
    get_number_of_model_parameters,
    RuntimeLimits,
)
from dingo.populations.training.population_dataset import (
    PopulationDataset,
    construct_population_dataset,
)
from dingo.populations.training.transform_builders import set_train_transforms


# Allow use of TF32 datatype if available. This may speed up training, although
# precision is somewhat reduced.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def train(
    train_dir: str,
    local_settings: dict,
    resume=False,
    train_settings: dict = None,
    checkpoint=None,
):
    if resume:
        pm = PosteriorModel(model_filename=checkpoint, device=local_settings["device"])
        train_settings = pm.metadata["train_settings"]

    # (1) Prepare training data

    # Build population forward models (train and test). These use different halves of
    # the base population events.
    population_model_train = construct_population_dataset(
        device=local_settings["device"], 
        mode="train", 
        **train_settings["data"],
    )
    population_model_test = construct_population_dataset(
        device=local_settings["device"], 
        mode="test", 
        **train_settings["data"],
    )

    set_train_transforms(population_model_train, train_settings["data"])
    set_train_transforms(population_model_test, train_settings["data"])

    train_loader = DataLoader(
        population_model_train,
        batch_size=train_settings["training"]["batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=local_settings["num_workers"],
        worker_init_fn=fix_random_seeds,
    )
    test_loader = DataLoader(
        population_model_test,
        batch_size=train_settings["training"]["batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=local_settings["num_workers"],
        worker_init_fn=fix_random_seeds,
    )

    # (2) Prepare neural model

    if not resume:
        # Configure additional network settings based on the embedding size and number
        # of hyperparameters.
        model_settings = train_settings["model"]
        model_settings["transformer_kwargs"][
            "d_dingo_encoding"
        ] = population_model_train.embedding_size
        model_settings["nsf_kwargs"][
            "input_dim"
        ] = population_model_train.num_hyperparameters

        # (3) Build model
        full_settings = {
            "base_settings": population_model_train.base_settings,
            "train_settings": train_settings,
        }
        pm = PosteriorModel(metadata=full_settings, device=local_settings["device"])

        pm.optimizer_kwargs = train_settings["training"]["optimizer"]
        pm.scheduler_kwargs = train_settings["training"]["scheduler"]
        pm.initialize_optimizer_and_scheduler()

    num_params_encoder = get_number_of_model_parameters(pm.model.embedding_net)
    num_params_flow = get_number_of_model_parameters(pm.model.flow)
    print(
        f"Number of parameters:\n  Encoder: {num_params_encoder:.1e}\n  Flow: "
        f"{num_params_flow:.1e}\n  Total:  {num_params_encoder+num_params_flow:.1e}"
    )

    # (3) Train

    runtime_limits = RuntimeLimits(
        epoch_start=pm.epoch, **local_settings["runtime_limits"]
    )
    runtime_limits.max_epochs_total = train_settings["training"]["epochs"]
    pm.train(
        train_loader,
        test_loader,
        train_dir=train_dir,
        runtime_limits=runtime_limits,
        checkpoint_epochs=local_settings["checkpoint_epochs"],
        use_wandb=local_settings.get("wandb", False),
        test_only=local_settings.get("test_only", False),
    )

    population_model_train.save_stats(train_dir)

    if pm.epoch == train_settings["training"]["epochs"]:
        save_file = os.path.join(train_dir, f"model_complete.pt")
        print(f"Training complete. Saving to {save_file}.")
        pm.save_model(save_file, save_training_info=True)
        return True
    if runtime_limits.local_limits_exceeded(pm.epoch):
        print("Local runtime limits reached. Ending program.")
        return False


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
        resume = False
        checkpoint = None
        with open(args.settings_file, "r") as fp:
            train_settings = yaml.safe_load(fp)

        # Extract the local settings from train settings file, save it separately. This
        # file can later be modified, and the settings take effect immediately upon
        # resuming.

        local_settings = train_settings.pop("local")
        with open(os.path.join(args.train_dir, "local_settings.yaml"), "w") as f:
            yaml.dump(local_settings, f, default_flow_style=False, sort_keys=False)

    else:
        print("Resuming training run.")
        resume = True
        checkpoint = args.checkpoint
        train_settings = None
        with open(os.path.join(args.train_dir, "local_settings.yaml"), "r") as f:
            local_settings = yaml.safe_load(f)

    with threadpool_limits(limits=1, user_api="blas"):
        complete = train(
            train_dir=args.train_dir,
            local_settings=local_settings,
            resume=resume,
            train_settings=train_settings,
            checkpoint=checkpoint,
        )

    if complete:
        print("Training complete.")
    else:
        print("Program terminated due to runtime limit.")


if __name__ == "__main__":
    train_local()
