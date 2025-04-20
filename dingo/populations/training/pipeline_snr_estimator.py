import os
import numpy as np
import yaml
import argparse
import textwrap


from threadpoolctl import threadpool_limits

from dingo.core.nn.nsf import autocomplete_model_kwargs_nsf
from dingo.gw.training.train_builders import (
    build_dataset,
    build_train_and_test_loaders,
    set_train_transforms,
    build_svd_for_embedding_network,
)
from dingo.core.models.posterior_model import PosteriorModel
from dingo.core.utils.trainutils import RuntimeLimits
from dingo.core.utils import (
    set_requires_grad_flag,
    get_number_of_model_parameters,
    build_train_and_test_loaders,
)
from dingo.gw.dataset import WaveformDataset
from dingo.populations.training.pipeline_embedding_emulator import (
    train_stages
)
from dingo.populations.models_training.models import (
    SNREstimator
)

from dingo.gw.transforms import (
    AddWhiteNoiseComplex,
    SelectStandardizeRepackageParameters
)

def complete_train_settings_snr(model_kwargs, single_event_model):
    
    model_kwargs['input_dim'] = single_event_model.metadata['train_settings']['model']['embedding_net_kwargs']['output_dim']
    model_kwargs['output_dim'] = 1

def set_train_transforms_snr_estimate(wfd, train_settings):

    snr_mean = train_settings['data']['standardization']['mean']['snr']
    snr_std = train_settings['data']['standardization']['std']['snr']
    for t in wfd.transform.transforms:
        if isinstance(t, AddWhiteNoiseComplex):
            t.store_snr = True
        if isinstance(t, SelectStandardizeRepackageParameters):
            t.mean['matched_filter_snr'] = snr_mean
            t.std['matched_filter_snr'] = snr_std
        
def prepare_training_new(train_settings: dict, train_dir: str, local_settings: dict):
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
    (WaveformDataset, PosteriorModel)
    """
    device = local_settings['device']

    # put either model on cpu, or move all tensors to cuda
    pm_embeddings = PosteriorModel(train_settings['data']['posterior_model'], device=device)

    # for debug purposes
    # pm_embeddings.metadata['train_settings']['data']['waveform_dataset_path'] = '/mnt/lustre2/gravitational_waves/kleyde/dingo_population/waveform_datasets/waveform_test.hdf5'

    # important! Overwrite inference parameters with SNR here
    pm_embeddings.metadata['train_settings']["data"]['inference_parameters'] = ['matched_filter_snr']
    wfd = build_dataset(pm_embeddings.metadata['train_settings']["data"])  # No transforms yet

    # This is the only case that exists so far, but we leave it open to develop new
    # model types.
    if train_settings["model"]["type"] == "nn-dense-res":

        set_train_transforms(
            wfd,
            pm_embeddings.metadata['train_settings']["data"],
            pm_embeddings.metadata['train_settings']["training"]["stage_0"]["asd_dataset_path"],
        )

        set_train_transforms_snr_estimate(
            wfd,
            train_settings
        )

        complete_train_settings_snr(train_settings["model"], pm_embeddings)
        
        full_settings = {
            "dataset_settings": wfd.settings,
            "train_settings": train_settings,
        }

    else:
        raise ValueError('Model type must be "nn-dense-res".')

    print("\nInitializing new posterior model.")
    print("Complete settings:")
    print(yaml.dump(full_settings, default_flow_style=False, sort_keys=False))

    print("full_settings", full_settings["train_settings"]["model"])
    
    model = SNREstimator(
        metadata=full_settings,
        device=device,
        pm_single_event=pm_embeddings
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
            print("WandB is enabled but not installed.")

    return model, wfd


def prepare_training_resume(checkpoint_name, local_settings, train_dir):
    """
    Loads a PosteriorModel from a checkpoint, as well as the corresponding
    WaveformDataset, in order to continue training. It initializes the saved optimizer
    and scheduler from the checkpoint.

    Parameters
    ----------
    checkpoint_name : str
        File name containing the checkpoint (.pt format).
    

    Returns
    -------
    (PosteriorModel, WaveformDataset)
    """

    device = local_settings['device']

    # TODO: 
    pm = SNREstimator(
        model_filename=checkpoint_name,
        device=device,
        pm_single_event=None
    )

    pm.add_pm_single_event()

    # important! Overwrite inference parameters with SNR here
    pm.pm_single_event.metadata['train_settings']["data"]['inference_parameters'] = ['matched_filter_snr']
    wfd = build_dataset(pm.pm_single_event.metadata['train_settings']['data'])
    set_train_transforms(
        wfd,
        pm.pm_single_event.metadata['train_settings']['data'],
        pm.pm_single_event.metadata['train_settings']['training']["stage_0"]["asd_dataset_path"],
    )
    set_train_transforms_snr_estimate(
        wfd,
        pm.metadata['train_settings']
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

    return pm, wfd


def initialize_stage(pm, wfed, stage, num_workers, resume=False):
    """
    Initializes training based on PosteriorModel metadata and current stage:
        * Builds transforms (based on noise settings for current stage);
        * Builds DataLoaders;
        * At the beginning of a stage (i.e., if not resuming mid-stage), initializes
        a new optimizer and scheduler;
        * Freezes / unfreezes SVD layer of embedding network

    Parameters
    ----------
    pm : PosteriorModel
    wfed : WaveformDataset
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
    # set_train_transforms(wfd, train_settings["data"], stage["asd_dataset_path"])

    # Allows for changes in batch size between stages.
    train_loader, test_loader = build_train_and_test_loaders(
        wfed,
        train_settings["data"]["train_fraction"],
        stage["batch_size"],
        num_workers,
    )

    if not resume:
        # New optimizer and scheduler. If we are resuming, these should have been
        # loaded from the checkpoint.
        print("Initializing new optimizer and scheduler.")
        pm.optimizer_kwargs = stage["optimizer"]
        pm.scheduler_kwargs = stage["scheduler"]
        pm.initialize_optimizer_and_scheduler()

    # Freeze/unfreeze RB layer if necessary
    if "freeze_rb_layer" in stage:
        if stage["freeze_rb_layer"]:
            set_requires_grad_flag(
                pm.model, name_contains="layers_rb", requires_grad=False
            )
        else:
            set_requires_grad_flag(
                pm.model, name_contains="layers_rb", requires_grad=True
            )
    n_grad = get_number_of_model_parameters(pm.model, (True,))
    n_nograd = get_number_of_model_parameters(pm.model, (False,))
    print(f"Fixed parameters: {n_nograd}\nLearnable parameters: {n_grad}\n")

    return train_loader, test_loader

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
            if local_settings.get("wandb", False) and "id" not in local_settings["wandb"].keys():
                try:
                    import wandb
                    local_settings["wandb"]["id"] = wandb.util.generate_id()
                except ImportError:
                    print("wandb not installed, cannot generate run id.")
            yaml.dump(local_settings, f, default_flow_style=False, sort_keys=False)

        pm, wfd = prepare_training_new(train_settings, args.train_dir, local_settings)

    else:
        print("Resuming training run.")
        with open(os.path.join(args.train_dir, "local_settings.yaml"), "r") as f:
            local_settings = yaml.safe_load(f)
        pm, wfd = prepare_training_resume(
            args.checkpoint, local_settings, args.train_dir
        )

    with threadpool_limits(limits=1, user_api="blas"):
        complete = train_stages(pm, wfd, args.train_dir, local_settings)

    if complete:
        print("All training stages complete.")
    else:
        print("Program terminated due to runtime limit.")