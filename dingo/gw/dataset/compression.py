import copy
import logging
from functools import partial
from typing import Dict

import numpy as np
import pandas as pd
from bilby.gw.prior import BBHPriorDict
from torchvision.transforms import Compose

from dingo.core.utils.misc import call_func_strict_output_dim
from dingo.gw.SVD import SVDBasis
from dingo.gw.dataset.waveform_dataset import WaveformDataset
from dingo.gw.waveform_generator import WaveformGenerator

logger = logging.getLogger(__name__)


def train_svd_basis(dataset: WaveformDataset, size: int, n_train: int) -> SVDBasis:
    """
    Train (and optionally validate) an SVD basis.
    """
    train_data = np.vstack([val[:n_train] for val in dataset.polarizations.values()])
    test_data = np.vstack([val[n_train:] for val in dataset.polarizations.values()])
    test_parameters = pd.concat(
        [
            # I would like to save the polarization, but saving the dataframe with
            # string columns causes problems. Fix this later.
            # dataset.parameters.iloc[n_train:].assign(polarization=pol)
            dataset.parameters.iloc[n_train:]
            for pol in dataset.polarizations
        ]
    )
    test_parameters.reset_index(drop=True, inplace=True)

    logger.info("Building SVD basis.")
    basis = SVDBasis()
    basis.generate_basis(train_data, size)

    assert np.allclose(basis.V[: dataset.domain.min_idx], 0)

    if test_data.size != 0:
        basis.compute_test_mismatches(
            test_data, parameters=test_parameters, verbose=True
        )

    return basis


def load_svd_basis(file_name: str) -> SVDBasis:
    """Load an existing SVD basis for Hydra-configured compression."""
    return SVDBasis(file_name=file_name)


def train_svd_basis_from_waveforms(
    size: int,
    num_training_samples: int,
    num_validation_samples: int = 0,
    *,
    waveform_generator: WaveformGenerator,
    prior: BBHPriorDict,
    num_processes: int,
    settings: Dict,
    compression_transforms: list,
    compression_settings: list,
) -> SVDBasis:
    """Train an SVD basis from waveforms generated with preceding compression steps."""
    from dingo.gw.dataset.generate_dataset import generate_parameters_and_polarizations

    waveform_generator.transform = Compose(compression_transforms)

    num_samples = num_training_samples + num_validation_samples
    func = partial(
        generate_parameters_and_polarizations,
        waveform_generator,
        prior,
        num_processes=num_processes,
    )
    parameters, polarizations = call_func_strict_output_dim(func, num_samples)

    svd_dataset_settings = copy.deepcopy(settings)
    svd_dataset_settings["num_samples"] = len(parameters)
    svd_dataset_settings["compression"] = copy.deepcopy(compression_settings) or None

    # We build a WaveformDataset containing the SVD-training waveforms because when
    # constructed, it automatically zeroes waveforms below f_min.
    svd_dataset = WaveformDataset(
        dictionary={
            "parameters": parameters,
            "polarizations": polarizations,
            "settings": svd_dataset_settings,
        }
    )
    return train_svd_basis(svd_dataset, size, num_training_samples)
