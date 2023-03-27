import copy

import torch

from dingo.core.utils import build_train_and_test_loaders
from dingo.core.utils.trainutils import RuntimeLimits
import numpy as np
import pandas as pd
import argparse

from dingo.core.models import PosteriorModel


class SampleDataset(torch.utils.data.Dataset):
    """
    Dataset class for unconditional density estimation.
    This is required, since the training method of dingo.core.models.PosteriorModel
    expects a tuple of (theta, *context) as output of the DataLoader, but here we have
    no context, so len(context) = 0. This SampleDataset therefore returns a tuple
    (theta, ) instead of just theta.
    """

    def __init__(self, data):
        self.data = data
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """Return the data and labels at the given index as a tuple of length 1."""
        return (self.data[index],)


def train_unconditional_density_estimator(
    result,
    settings: dict,
    train_dir: str,
):
    """
    Train unconditional density estimator for a given set of samples.

    Parameters
    ----------
    samples: pd.DataFrame
        DataFrame containing the samples to train the density estimator on.
    settings: dict
        Dictionary containing the settings for the density estimator.
    train_dir: str
        Path to the directory where the trained model should be saved.

    Returns
    -------
    model: PosteriorModel
        trained density estimator
    """
    samples = result.samples
    # Process samples: select parameters, normalize, and convert to torch tensor
    if "parameters" in settings["data"] and settings["data"]["parameters"]:
        parameters = settings["data"]["parameters"]
    else:
        parameters = list(samples.keys())
    samples = np.array(samples[parameters])
    num_samples, num_params = samples.shape
    mean, std = np.mean(samples, axis=0), np.std(samples, axis=0)
    settings["data"]["standardization"] = {
        "mean": {param: mean[i].item() for i, param in enumerate(parameters)},
        "std": {param: std[i].item() for i, param in enumerate(parameters)},
    }
    settings["data"]["unconditional"] = True
    # normalized torch samples
    samples_torch = torch.from_numpy((samples - mean) / std).float()

    # set up density estimation network
    settings["model"]["input_dim"] = num_params
    settings["model"]["context_dim"] = None
    model = PosteriorModel(
        metadata={"train_settings": settings, "base": copy.deepcopy(result.metadata)},
        device=settings["training"]["device"],
    )
    model.optimizer_kwargs = settings["training"]["optimizer"]
    model.scheduler_kwargs = settings["training"]["scheduler"]
    model.initialize_optimizer_and_scheduler()

    # Store context and event metadata to keep a record, even though it will not be used
    # in training.
    model.context = result.context
    model.event_metadata = result.event_metadata

    # set up dataloaders
    train_loader, test_loader = build_train_and_test_loaders(
        SampleDataset(samples_torch),
        settings["training"]["train_fraction"],
        settings["training"]["batch_size"],
        settings["training"]["num_workers"],
    )

    # train model
    runtime_limits = RuntimeLimits(
        epoch_start=0,
        max_epochs_total=settings["training"]["epochs"],
    )
    model.train(
        train_loader,
        test_loader,
        train_dir=train_dir,
        runtime_limits=runtime_limits,
    )

    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unconditional density estimation for dingo samples."
    )
    parser.add_argument(
        "--settings",
        type=str,
        required=True,
        help="Path to settings file.",
    )
    return parser.parse_args()
