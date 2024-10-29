import argparse
import os
import textwrap

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

from dingo.core.posterior_models.base_model import Base
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
    get_number_of_model_parameters,
    build_train_and_test_loaders,
)
from dingo.core.utils.trainutils import RuntimeLimits
from dingo.core.utils.environment import document_environment
from dingo.core.utils.torchutils import document_gpus
from dingo.gw.dataset import WaveformDataset
from dingo.gw.training.train_builders import (
    build_dataset,
    set_train_transforms,
    build_svd_for_embedding_network,
)


def prepare_wfd_and_initialization_for_embedding_network(
    train_settings: dict,
    train_dir: str,
    local_settings: dict,
    pretraining: bool = False,
    print_output: bool = True,
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
    (WaveformDataset, dict, torch.nn.module)
    """

    # No transforms yet
    wfd = build_dataset(
        train_settings["data"],
        copy_to_tmp=local_settings.get("copy_waveform_dataset_to_tmp", False),
    )
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
            initial_weights["V_rb_list"] = build_svd_for_embedding_network(
                wfd,
                train_settings["data"],
                train_settings["training"]["stage_0"]["asd_dataset_path"],
                num_workers=local_settings["num_workers"],
                batch_size=train_settings["training"]["stage_0"]["batch_size"],
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
        wfd,
        train_settings["data"],
        train_settings["training"]["stage_0"]["asd_dataset_path"],
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
    (WaveformDataset, dict, torch.nn.module)
    """

    # This modifies the model settings in-place.
    autocomplete_model_kwargs(
        model_kwargs=train_settings["model"],
        data_sample=wfd[0],
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
    (WaveformDataset, Base)
    """

    (
        wfd,
        initial_weights,
        pretrained_embedding_net,
    ) = prepare_wfd_and_initialization_for_embedding_network(
        train_settings,
        train_dir,
        local_settings,
        pretraining,
    )
    pm = prepare_model_new(
        train_settings,
        train_dir,
        local_settings,
        wfd,
        initial_weights,
        pretraining,
        pretrained_embedding_net,
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
    pretraining: bool = False,
):
    """
    Loads a PosteriorModel from a checkpoint in order to continue training. It initializes the saved optimizer
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
    Base
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
    pretraining: bool = False,
):
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
    (Base, WaveformDataset)
    """

    train_settings = load_settings_from_ckpt(checkpoint_name)
    wfd = build_dataset(
        train_settings["data"],
        copy_to_tmp=local_settings.get("copy_waveform_dataset_to_tmp", False),
    )

    pm = prepare_model_resume(
        checkpoint_name,
        local_settings,
        train_dir,
        pretraining,
    )

    return pm, wfd


def initialize_stage(
    pm,
    wfd,
    stage,
    num_workers,
    world_size: int = None,
    rank: int = None,
    resume=False,
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
    pm : Base
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
        wfd,
        train_settings["data"],
        stage["asd_dataset_path"],
        print_output=print_output,
    )

    # Convert total batch size into batch size per GPU
    if world_size is not None and world_size > 1:
        total_batch_size = stage["batch_size"]
        if total_batch_size % world_size != 0:
            raise ValueError(
                f"Total batch size {total_batch_size} is not divisible by the number of GPUs {world_size}."
            )
        stage["batch_size"] = int(total_batch_size / world_size)

    # Allows for changes in batch size between stages.
    train_loader, test_loader, train_sampler = build_train_and_test_loaders(
        wfd,
        train_settings["data"]["train_fraction"],
        stage["batch_size"],
        num_workers,
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
        # ... the batch size is the batch size per GPU
        train_size = int(
            train_settings["data"]["train_fraction"] * wfd.settings["num_samples"]
        )
        grad_updates_per_optimizer_step = stage.get(
            "gradient_updates_per_optimizer_step", 1
        )
        num_gpus = world_size if world_size is not None else 1
        num_optimizer_steps = int(
            np.ceil(
                train_size
                / (stage["batch_size"] * num_gpus * grad_updates_per_optimizer_step)
            )
        )
        pm.initialize_optimizer_and_scheduler(num_optimizer_steps=num_optimizer_steps)

    # Freeze/unfreeze RB layer if necessary
    if "freeze_rb_layer" in stage:
        if stage["freeze_rb_layer"]:
            set_requires_grad_flag(
                pm.network, name_contains="layers_rb", requires_grad=False
            )
        else:
            set_requires_grad_flag(
                pm.network, name_contains="layers_rb", requires_grad=True
            )
    n_grad = get_number_of_model_parameters(pm.network, (True,))
    n_nograd = get_number_of_model_parameters(pm.network, (False,))
    if print_output:
        print(f"Fixed parameters: {n_nograd}\nLearnable parameters: {n_grad}\n")

    return train_loader, test_loader, train_sampler


def train_stages(pm, wfd, train_dir, local_settings):
    """
    Train the network, iterating through the sequence of stages. Stages can change
    certain settings such as the noise characteristics, optimizer, and scheduler settings.

    Parameters
    ----------
    pm : Base
    wfd : WaveformDataset
    train_dir : str
        Directory for saving checkpoints and train history.
    local_settings : dict

    Returns
    -------
    bool
        True if all stages are complete
        False otherwise
    """

    train_settings = pm.metadata["train_settings"]
    runtime_limits = RuntimeLimits(
        epoch_start=pm.epoch, **local_settings["runtime_limits"]
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

    num_starting_stage = np.searchsorted(end_epochs, pm.epoch + 1)
    for n in range(num_starting_stage, num_stages):
        stage = stages[n]

        if pm.epoch == end_epochs[n] - stage["epochs"]:
            if print_bool:
                print(f"\nBeginning training stage {n}. Settings:")
                print(yaml.dump(stage, default_flow_style=False, sort_keys=False))
            train_loader, test_loader, train_sampler = initialize_stage(
                pm,
                wfd,
                stage,
                local_settings["num_workers"],
                world_size=local_settings.get("world_size", None),
                rank=rank,
                resume=False,
            )
        else:
            if print_bool:
                print(f"\nResuming training in stage {n}. Settings:")
                print(yaml.dump(stage, default_flow_style=False, sort_keys=False))
            train_loader, test_loader, train_sampler = initialize_stage(
                pm,
                wfd,
                stage,
                local_settings["num_workers"],
                world_size=local_settings.get("world_size", None),
                rank=rank,
                resume=True,
            )

        runtime_limits.max_epochs_total = end_epochs[n]
        pm.train(
            train_loader,
            test_loader,
            train_sampler,
            train_dir=train_dir,
            runtime_limits=runtime_limits,
            checkpoint_epochs=local_settings["checkpoint_epochs"],
            use_wandb=local_settings.get("wandb", False),
            test_only=local_settings.get("test_only", False),
            gradient_updates_per_optimizer_step=stage.get(
                "gradient_updates_per_optimizer_step", 1
            ),
            automatic_mixed_precision=stage.get("automatic_mixed_precision", False),
            world_size=local_settings.get("world_size", 1),
        )
        # if test_only, model should not be saved, and run is complete
        if local_settings.get("test_only", False):
            return True

        if pm.epoch == end_epochs[n]:
            # Only save model on one device
            if rank is None or rank == 0:
                save_file = os.path.join(train_dir, f"model_stage_{n}.pt")
                print(f"Training stage complete. Saving to {save_file}.")
                pm.save_model(save_file, save_training_info=True)
        if runtime_limits.local_limits_exceeded(pm.epoch) and print_bool:
            print("Local runtime limits reached. Ending program.")
            break

    if pm.epoch == end_epochs[-1]:
        return True
    else:
        return False


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

    # Document setup
    document_environment(args.train_dir)
    document_gpus(args.train_dir)

    if args.settings_file is not None:
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

        pm, wfd = prepare_training_new(
            train_settings, args.train_dir, local_settings, args.pretraining
        )

    else:
        print("Resuming training run.")
        with open(os.path.join(args.train_dir, "local_settings.yaml"), "r") as f:
            local_settings = yaml.safe_load(f)
        pm, wfd = prepare_training_resume(
            args.checkpoint, local_settings, args.train_dir, args.pretraining
        )

    with threadpool_limits(limits=1, user_api="blas"):
        complete = train_stages(pm, wfd, args.train_dir, local_settings)

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
