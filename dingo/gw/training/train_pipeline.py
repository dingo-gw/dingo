from typing import Optional, Tuple
import logging
import os

import hydra
import numpy as np
import torchvision
import yaml
import shutil
import time
from copy import deepcopy

from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from threadpoolctl import threadpool_limits

from dingo.core.posterior_models.build_model import build_model_from_kwargs
from dingo.core.utils.hydra_utils import instantiate_with_runtime_dependencies
from dingo.gw.training.train_builders import (
    set_train_transforms,
    build_svd_for_embedding_network,
)
from dingo.gw.gwutils import get_standardization_dict
from dingo.core.utils.trainutils import RuntimeLimits
from dingo.core.utils import (
    set_requires_grad_flag,
    get_number_of_model_parameters,
    build_train_and_test_loaders,
)
from dingo.core.utils.trainutils import EarlyStopping
from dingo.gw.dataset import WaveformDataset
from dingo.core.posterior_models import BasePosteriorModel

logger = logging.getLogger(__name__)
logging.captureWarnings(True)


def _standardization_is_missing(standardization: dict) -> bool:
    if standardization in (None, "???"):
        return True
    return (
        standardization.get("mean") in (None, "???")
        or standardization.get("std") in (None, "???")
    )


def _populate_parameter_standardization(
    wfd: WaveformDataset,
    data_settings: dict,
    asd_dataset_config: dict,
) -> None:
    if not _standardization_is_missing(data_settings.get("standardization")):
        logger.info("Using previously-calculated parameter standardizations.")
        _sync_parameter_standardization_transform_configs(data_settings)
        return

    logger.info("Calculating new parameter standardizations.")
    asd_dataset = instantiate(asd_dataset_config, domain_update=wfd.domain.domain_dict)
    runtime_dependencies = {
        "domain": wfd.domain,
        "asd_dataset": asd_dataset,
    }
    transforms = [
        instantiate_with_runtime_dependencies(transform_config, runtime_dependencies)
        for transform_config in data_settings["standardization_transforms"]
    ]
    selected_parameters = (
        data_settings["inference_parameters"] + data_settings["context_parameters"]
    )
    data_settings["standardization"] = get_standardization_dict(
        instantiate(data_settings["extrinsic_prior"]),
        wfd,
        selected_parameters,
        torchvision.transforms.Compose(transforms),
    )
    _sync_parameter_standardization_transform_configs(data_settings)


def _sync_parameter_standardization_transform_configs(data_settings: dict) -> None:
    for transform_config in data_settings["transforms"]:
        if transform_config.get("_target_", "").endswith(
            "SelectStandardizeRepackageParameters"
        ):
            transform_config["standardization_dict"] = data_settings["standardization"]


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
        logger.info(f"Copying file to {local_file_path}")
        # Copy file
        start_time = time.time()
        shutil.copy(file_path, local_file_path)
        elapsed_time = time.time() - start_time
        logger.info(
            "Done. This took {:2.0f}:{:2.0f} min.".format(*divmod(elapsed_time, 60))
        )
    elif leave_keys_on_disk and is_condor:
        logger.warning(
            f"leave_waveforms_on_disk defaults to True, but local_cache_path is not specified. "
            f"This means that the waveforms will be loaded during training from {local_file_path}. "
            f"This can lead to unexpected long times for data loading during training due to network traffic. "
            f"To prevent this, specify 'local_cache_path = tmp' in the local settings or set "
            f"leave_waveforms_on_disk = False. However, the latter is not recommended for large datasets since "
            f"it can lead to memory issues when loading the entire dataset into RAM."
        )

    return local_file_path


def prepare_training_new(
    train_settings: dict, train_dir: str, local_settings: dict
) -> Tuple[BasePosteriorModel, WaveformDataset]:
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

    Returns
    -------
    (BasePosteriorModel, WaveformDataset)
    """
    data_settings = deepcopy(train_settings["data"])
    # Optionally copy files to local and update path
    data_settings["waveform_dataset"]["file_name"] = copy_files_to_local(
        file_path=data_settings["waveform_dataset"]["file_name"],
        local_dir=local_settings.get("local_cache_path", None),
        leave_keys_on_disk=local_settings.get("leave_waveforms_on_disk", True),
        is_condor=True if "condor" in local_settings else False,
    )
    wfd = instantiate(
        data_settings["waveform_dataset"],
        leave_waveforms_on_disk=local_settings.get("leave_waveforms_on_disk", True),
    )  # No transforms yet
    train_settings["data"] = data_settings
    initial_weights = {}

    _populate_parameter_standardization(
        wfd,
        train_settings["data"],
        train_settings["training"]["stage_0"]["asd_dataset"],
    )

    # The embedding network is assumed to have an SVD projection layer. If other types
    # of embedding networks are added in the future, update this code.

    svd_settings = (train_settings["model"].get("embedding_kwargs") or {}).get("svd")
    if svd_settings and svd_settings.get("num_training_samples", 0) > 0:
        # First, build the SVD for seeding the embedding network.
        logger.info("\nBuilding SVD for initialization of embedding network.")
        initial_weights["V_rb_list"] = build_svd_for_embedding_network(
            wfd,
            train_settings["data"],
            train_settings["training"]["stage_0"]["asd_dataset"],
            num_workers=local_settings["num_workers"],
            batch_size=train_settings["training"]["stage_0"]["batch_size"],
            out_dir=train_dir,
            **train_settings["model"]["embedding_kwargs"]["svd"],
        )

    # Now set the transforms for training. We need to do this here so that we can (a)
    # get the data dimensions to configure the network, and (b) save the
    # parameter standardization dict in the PosteriorModel. In principle, (a) could
    # be done without generating data (by careful calculation) and (b) could also
    # be done outside the transform setup. But for now, this is convenient. The
    # transforms will be reset later by initialize_stage().

    set_train_transforms(
        wfd,
        train_settings["data"],
        train_settings["training"]["stage_0"]["asd_dataset"],
    )

    data_sample = wfd[0]
    train_settings["model"]["embedding_kwargs"]["input_dims"] = list(
        data_sample[1].shape
    )
    train_settings["model"]["posterior_kwargs"]["input_dim"] = len(data_sample[0])
    try:
        gnpe_proxy_dim = len(data_sample[2])
        train_settings["model"]["embedding_kwargs"]["added_context"] = True
        train_settings["model"]["posterior_kwargs"]["context_dim"] = (
            train_settings["model"]["embedding_kwargs"]["output_dim"] + gnpe_proxy_dim
        )
    except IndexError:
        train_settings["model"]["embedding_kwargs"]["added_context"] = False
        train_settings["model"]["posterior_kwargs"]["context_dim"] = train_settings[
            "model"
        ]["embedding_kwargs"]["output_dim"]
    full_settings = {
        "dataset_settings": wfd.settings,
        "train_settings": train_settings,
    }

    logger.info("\nInitializing new posterior model.")
    logger.info("Complete settings:")
    logger.info(yaml.dump(full_settings, default_flow_style=False, sort_keys=False))

    pm = build_model_from_kwargs(
        settings=full_settings,
        initial_weights=initial_weights or None,
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
            logger.warning("WandB is enabled but not installed.")

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
    data_settings["waveform_dataset"]["file_name"] = copy_files_to_local(
        file_path=data_settings["waveform_dataset"]["file_name"],
        local_dir=local_settings.get("local_cache_path", None),
        leave_keys_on_disk=local_settings.get("leave_waveforms_on_disk", True),
        is_condor=True if "condor" in local_settings else False,
    )
    wfd = instantiate(
        data_settings["waveform_dataset"],
        leave_waveforms_on_disk=local_settings.get("leave_waveforms_on_disk", True),
    )

    if local_settings.get("wandb", False):
        try:
            import wandb

            wandb.init(
                resume="allow",
                dir=train_dir,
                **local_settings["wandb"],
            )
        except ImportError:
            logger.warning("WandB is enabled but not installed.")

    return pm, wfd


def initialize_stage(
    pm: BasePosteriorModel,
    wfd: WaveformDataset,
    stage: dict,
    num_workers: int,
    resume: bool = False,
):
    """
    Initializes training based on PosteriorModel metadata and current stage:
        * Builds transforms (based on noise settings for current stage);
        * Builds DataLoaders;
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
    resume : bool
        Whether training is resuming mid-stage. This controls whether the optimizer and
        scheduler should be re-initialized based on contents of stage dict.

    Returns
    -------
    (train_loader, test_loader)
    """

    train_settings = pm.metadata["train_settings"]

    # Rebuild transforms based on possibly different noise.
    set_train_transforms(wfd, train_settings["data"], stage["asd_dataset"])

    # Allows for changes in batch size between stages.
    train_loader, test_loader = build_train_and_test_loaders(
        wfd,
        train_settings["data"]["train_fraction"],
        stage["batch_size"],
        num_workers,
    )

    if not resume:
        # New optimizer and scheduler. If we are resuming, these should have been
        # loaded from the checkpoint.
        logger.info("Initializing new optimizer and scheduler.")
        pm.optimizer_kwargs = stage["optimizer"]
        pm.scheduler_kwargs = stage["scheduler"]
        pm.initialize_optimizer_and_scheduler()

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
    logger.info(f"Fixed parameters: {n_nograd}\nLearnable parameters: {n_grad}\n")

    return train_loader, test_loader


def train_stages(
    pm: BasePosteriorModel, wfd: WaveformDataset, train_dir: str, local_settings: dict
) -> bool:
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
            logger.info(f"\nBeginning training stage {n}. Settings:")
            logger.info(yaml.dump(stage, default_flow_style=False, sort_keys=False))
            train_loader, test_loader = initialize_stage(
                pm, wfd, stage, local_settings["num_workers"], resume=False
            )
        else:
            logger.info(f"\nResuming training in stage {n}. Settings:")
            logger.info(yaml.dump(stage, default_flow_style=False, sort_keys=False))
            train_loader, test_loader = initialize_stage(
                pm, wfd, stage, local_settings["num_workers"], resume=True
            )
        early_stopping = None
        if stage.get("early_stopping"):
            try:
                early_stopping = EarlyStopping(**stage["early_stopping"])
            except Exception:
                logger.warning(
                    "Early stopping settings invalid. Please pass 'patience', 'delta', 'metric'"
                )
                raise

        runtime_limits.max_epochs_total = end_epochs[n]
        pm.train(
            train_loader,
            test_loader,
            train_dir=train_dir,
            runtime_limits=runtime_limits,
            checkpoint_epochs=local_settings["checkpoint_epochs"],
            use_wandb=local_settings.get("wandb", False),
            test_only=local_settings.get("test_only", False),
            early_stopping=early_stopping,
        )
        # if test_only, model should not be saved, and run is complete
        if local_settings.get("test_only", False):
            return True

        if pm.epoch == end_epochs[n]:
            save_file = os.path.join(train_dir, f"model_stage_{n}.pt")
            logger.info(f"Training stage complete. Saving to {save_file}.")
            pm.save_model(save_file, save_training_info=True)
        if runtime_limits.local_limits_exceeded(pm.epoch):
            logger.info("Local runtime limits reached. Ending program.")
            break

    if pm.epoch == end_epochs[-1]:
        return True
    else:
        return False


def _resolve_training_input_paths(train_settings: dict) -> None:
    train_settings["data"]["waveform_dataset"]["file_name"] = to_absolute_path(
        train_settings["data"]["waveform_dataset"]["file_name"]
    )
    for stage in train_settings["training"].values():
        if isinstance(stage, dict) and "asd_dataset" in stage:
            stage["asd_dataset"]["file_name"] = to_absolute_path(
                stage["asd_dataset"]["file_name"]
            )


@hydra.main(
    version_base="1.3",
    config_path="../../../configs",
    config_name="train",
)
def train_local(cfg: DictConfig):
    settings = OmegaConf.to_container(cfg, resolve=True)
    checkpoint = settings.pop("checkpoint")
    train_dir = settings.pop("train_dir")
    exit_command = settings.pop("exit_command")

    os.makedirs(train_dir, exist_ok=True)

    if checkpoint is None:
        logger.info("Beginning new training run.")
        train_settings = settings
        _resolve_training_input_paths(train_settings)

        # Extract the local settings from train settings file, save it separately. This
        # file can later be modified, and the settings take effect immediately upon
        # resuming.

        local_settings = train_settings.pop("local")
        with open(os.path.join(train_dir, "local_settings.yaml"), "w") as f:
            if (
                local_settings.get("wandb", False)
                and "id" not in local_settings["wandb"].keys()
            ):
                try:
                    import wandb

                    local_settings["wandb"]["id"] = wandb.util.generate_id()
                except ImportError:
                    logger.warning("wandb not installed, cannot generate run id.")
            yaml.dump(local_settings, f, default_flow_style=False, sort_keys=False)

        pm, wfd = prepare_training_new(train_settings, train_dir, local_settings)

    else:
        checkpoint = to_absolute_path(checkpoint)
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        logger.info("Resuming training run.")
        with open(os.path.join(train_dir, "local_settings.yaml"), "r") as f:
            local_settings = yaml.safe_load(f)
        pm, wfd = prepare_training_resume(checkpoint, local_settings, train_dir)

    with threadpool_limits(limits=1, user_api="blas"):
        complete = train_stages(pm, wfd, train_dir, local_settings)

    if complete:
        if exit_command:
            logger.info(
                f"All training stages complete. Executing exit command: {exit_command}."
            )
            os.system(exit_command)
        else:
            logger.info("All training stages complete.")
    else:
        logger.info("Program terminated due to runtime limit.")
