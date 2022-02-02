import os

os.environ["OMP_NUM_THREADS"] = str(1)
os.environ["MKL_NUM_THREADS"] = str(1)

import time
import torch
import numpy as np
import copy
import yaml
import argparse
import textwrap
from torch.utils.data import DataLoader

from dingo.core.nn.nsf import autocomplete_model_kwargs_nsf
from dingo.gw.SVD import SVDBasis
from dingo.api.train_setup_new import (
    build_dataset,
    build_train_and_test_loaders,
    set_train_transforms,
)
from dingo.core.models.posterior_model_new import PosteriorModel
from dingo.gw.transforms import (
    AddWhiteNoiseComplex,
    RepackageStrainsAndASDS,
    SelectStandardizeRepackageParameters,
    UnpackDict,
)
from dingo.core.utils import set_requires_grad_flag, get_number_of_model_parameters
from dingo.gw.dataset import WaveformDataset


def build_svd_for_embedding_network(
    wfd: WaveformDataset,
    data_settings: dict,
    asd_dataset_path: str,
    size: int,
    num_training_samples: int,
    num_validation_samples: int,
    num_workers: int = 0,
    batch_size: int = 1000,
    out_dir=None,
):
    """
    Construct SVD matrices V based on clean waveforms in each interferometer. These
    will be used to seed the weights of the initial projection part of the embedding
    network.

    It first generates a number of training waveforms, and then produces the SVD.

    Parameters
    ----------
    wfd : WaveformDataset
    data_settings : dict
    asd_dataset_path : str
        Training waveforms will be whitened with respect to these ASDs.
    size : int
        Number of basis elements to include in the SVD projection.
    num_training_samples : int
    num_validation_samples : int
    num_workers : int
    batch_size : int
    out_dir : str
        SVD performance diagnostics are saved here.

    Returns
    -------
    list of numpy arrays
        The V matrices for each interferometer. They are ordered as in data_settings[
        'detectors'].
    """
    # Building the transforms can alter the data_settings dictionary. We do not want
    # the construction of the SVD to impact this, so begin with a fresh copy of this
    # dictionary.
    data_settings = copy.deepcopy(data_settings)

    # Fix the luminosity distance to a standard value, just in order to generate the SVD.
    data_settings["extrinsic_prior"]["luminosity_distance"] = "100.0"

    # Build the dataset, but with certain transforms omitted. In particular, we want to
    # build the SVD based on zero-noise waveforms. They should still be whitened though.
    set_train_transforms(
        wfd,
        data_settings,
        asd_dataset_path,
        omit_transforms=[
            AddWhiteNoiseComplex,
            RepackageStrainsAndASDS,
            SelectStandardizeRepackageParameters,
            UnpackDict,
        ],
    )

    print("Generating waveforms for embedding network SVD initialization.")
    time_start = time.time()
    ifos = list(wfd[0]["waveform"].keys())
    waveform_len = len(wfd[0]["waveform"][ifos[0]])
    num_waveforms = num_training_samples + num_validation_samples
    waveforms = {
        ifo: np.empty((num_waveforms, waveform_len), dtype=np.complex128)
        for ifo in ifos
    }
    loader = DataLoader(
        wfd,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=lambda _: np.random.seed(
            int(torch.initial_seed()) % (2 ** 32 - 1)
        ),
    )
    for idx, data in enumerate(loader):
        strain_data = data["waveform"]
        lower = idx * batch_size
        n = min(batch_size, num_waveforms - lower)
        for ifo, strains in strain_data.items():
            waveforms[ifo][lower : lower + n] = strains[:n]
        if lower + n == num_waveforms:
            break
    print(f"...done. This took {time.time() - time_start:.0f} s.")

    print("Generating SVD basis for ifo:")
    time_start = time.time()
    basis_dict = {}
    for ifo in ifos:
        basis = SVDBasis()
        basis.generate_basis(waveforms[ifo][:num_training_samples], size)
        basis_dict[ifo] = basis
        print(f"...{ifo} done.")
    print(f"...this took {time.time() - time_start:.0f} s.")

    if out_dir is not None:
        print(f"Testing SVD basis matrices, saving stats to {out_dir}")
        for ifo, basis in basis_dict.items():
            basis.test_basis(
                waveforms[ifo][num_training_samples:],
                outfile=os.path.join(out_dir, f"SVD_{ifo}_stats.npy"),
            )
    print("Done")

    # Return V matrices in standard order.
    return [basis_dict[ifo].V for ifo in data_settings["detectors"]]


def prepare_training_new(train_settings: dict, train_dir: str):
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

    Returns
    -------
    (WaveformDataset, PosteriorModel)
    """

    wfd = build_dataset(train_settings["data"])  # No transforms yet
    initial_weights = {}

    # This is the only case that exists so far, but we leave it open to develop new
    # model types.
    if train_settings["model"]["type"] == "nsf+embedding":

        # First, build the SVD for seeding the embedding network.
        print("\nBuilding SVD for initialization of embedding network.")
        initial_weights["V_rb_list"] = build_svd_for_embedding_network(
            wfd,
            train_settings["data"],
            train_settings["training"]["stage_0"]["asd_dataset_path"],
            num_workers=train_settings["local"]["num_workers"],
            batch_size=train_settings["training"]["stage_0"]["batch_size"],
            out_dir=train_dir,
            **train_settings["model"]["embedding_net_kwargs"]["svd"],
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
            train_settings["training"]["stage_0"]["asd_dataset_path"],
        )

        # This modifies the model settings in-place.
        autocomplete_model_kwargs_nsf(train_settings["model"], wfd[0])
        full_settings = {
            "dataset_settings": wfd.settings,
            "train_settings": train_settings,
        }

    else:
        raise ValueError('Model type must be "nsf+embedding".')

    print("\nInitializing new posterior model.")
    print("Complete settings:")
    print(yaml.dump(full_settings, default_flow_style=False, sort_keys=False))

    pm = PosteriorModel(
        metadata=full_settings,
        initial_weights=initial_weights,
    )

    return pm, wfd


def prepare_training_resume(checkpoint_name):
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

    pm = PosteriorModel(model_filename=checkpoint_name)
    pm.initialize_optimizer_and_scheduler()

    wfd = build_dataset(pm.metadata["train_settings"]["data"])

    return pm, wfd


def initialize_stage(pm, wfd, stage, resume=False):
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
    wfd : WaveformDataset
    stage : dict
        Settings specific to current stage of training
    resume : bool
        Whether training is resuming mid-stage. This controls whether the optimizer and
        scheduler should be re-initialized based on contents of stage dict.

    Returns
    -------
    (train_loader, test_loader)
    """

    train_settings = pm.metadata["train_settings"]

    # Rebuild transforms based on possibly different noise.
    set_train_transforms(wfd, train_settings["data"], stage["asd_dataset_path"])

    # Allows for changes in batch size between stages.
    train_loader, test_loader = build_train_and_test_loaders(
        wfd,
        train_settings["data"]["train_fraction"],
        stage["batch_size"],
        train_settings["local"]["num_workers"],
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


def train_stages(pm, wfd, train_dir):
    """
    Train the network, iterating through the sequence of stages. Stages can change
    certain settings such as the noise characteristics, optimizer, and scheduler settings.

    Parameters
    ----------
    pm : PosteriorModel
    wfd : WaveformDataset
    train_dir : str
        Directory for saving checkpoints and train history.
    """

    train_settings = pm.metadata["train_settings"]

    # Extract list of stages from settings dict
    stages = []
    num_stages = 0
    while True:
        try:
            stages.append(train_settings["training"][f"stage_{num_stages}"])
            num_stages += 1
        except KeyError:
            break
    end_epochs = np.cumsum([stage["epochs"] for stage in stages])

    num_starting_stage = np.searchsorted(end_epochs, pm.epoch + 1)
    for n in range(num_starting_stage, num_stages):
        stage = stages[n]

        if pm.epoch == end_epochs[n] - stage["epochs"]:
            print(f"\nBeginning training stage {n}. Settings:")
            print(yaml.dump(stage, default_flow_style=False, sort_keys=False))
            train_loader, test_loader = initialize_stage(pm, wfd, stage, resume=False)

        else:
            print(f"\nResuming training in stage {n}. Settings:")
            print(yaml.dump(stage, default_flow_style=False, sort_keys=False))
            train_loader, test_loader = initialize_stage(pm, wfd, stage, resume=True)

        runtime_limits_kwargs = train_settings["local"]["runtime_limits"].copy()
        runtime_limits_kwargs["max_epochs_total"] = end_epochs[n]
        pm.train(
            train_loader,
            test_loader,
            train_dir=train_dir,
            runtime_limits_kwargs=runtime_limits_kwargs,
            checkpoint_epochs=train_settings["local"]["checkpoint_epochs"],
        )

        save_file = os.path.join(train_dir, f"model_stage_{n}.pt")
        print(f"Training stage complete. Saving to {save_file}.")
        pm.save_model(save_file, save_training_info=True)


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


def main():

    args = parse_args()

    os.makedirs(args.train_dir, exist_ok=True)

    if args.settings_file is not None:
        print("Beginning new training run.")
        with open(args.settings_file, "r") as fp:
            train_settings = yaml.safe_load(fp)
        pm, wfd = prepare_training_new(train_settings, args.train_dir)

    else:
        print("Resuming training run.")
        pm, wfd = prepare_training_resume(args.checkpoint)

    train_stages(pm, wfd, args.train_dir)
