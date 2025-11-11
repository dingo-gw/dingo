import copy
from os.path import join

import torchvision

import numpy as np
import pandas as pd
import torch.utils.data
import yaml
from bilby.core.prior import Constraint
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
from bilby.gw.prior import BBHPriorDict
from scipy.spatial import cKDTree

from dingo.core.dataset import DingoDataset
from dingo.populations.population_models import build_population_model
from dingo.populations.models_training.models import (
    SNREstimator,
    EmbeddingEmulator
) 
from dingo.gw.injection import Injection

from bilby.core.prior import Uniform, PriorDict
from dingo.populations.utilities import (
    build_bilby_prior_dict,
    calculate_mean_and_std
)
from dingo.gw.transforms import (
    SelectStandardizeRepackageParameters,
    UnpackDict
)

# TODO: remove this script? 

class PopulationDataset(torch.utils.data.Dataset):
    """
    Training dataset for population posterior that is based on stored embeddings.

    Contains a PopulationModel and a EventEmbeddingsDataset.
    """

    def __init__(
        self,
        population_model_name,
        population_prior_dict,
        mode,
        size,
        train_fraction=0.9,
        minimum_population_size=10,
        maximum_population_size=100,
        factor_event_generation=30,
        kwargs_selection_cut=None,
        embedding_emulator_metadata=None,
        **kwargs,  # These are ignored (particularly 'standardization'). Not clean.
    ):
        super().__init__()

        self.population_model_name = population_model_name
        self.population_prior = build_bilby_prior_dict(population_prior_dict)
        
        self.minimum_population_size = minimum_population_size
        self.maximum_population_size = maximum_population_size
        self.factor_event_generation = factor_event_generation

        # compute the number of events drawn each iteration for
        # population
        self.size_all_events = self.factor_event_generation * self.maximum_population_size

        self.embedding_emulator_metadata = embedding_emulator_metadata
        self.settings_pm_single_event = self.embedding_emulator_metadata['settings_pm_single_event']
        
        # if non-empty dictionary, events that do not pass this cut are directly discarded
        self.kwargs_selection_cut = kwargs_selection_cut

        self.mode = mode

        # Depending on whether we are in train or test mode, we take even or odd
        # elements of the base population (half for each).
        if mode == "train":
            # TODO, separate here? 
            self.size = int(size * train_fraction)
        elif mode == "test":
            self.size = int(size * (1 - train_fraction))
        else:
            raise ValueError(f"Mode {mode} not recognized.")

        # TODO replace here later with better metadata
        inj = Injection.from_posterior_model_metadata(self.settings_pm_single_event)
        self.population_model = build_population_model(
            population_model_name, self.population_prior, inj.prior
        )
        self.inference_parameters = self.get_inference_parameters()

        self.hyperparameters = None

        self.num_hyperparameters = len(self.population_prior)

        self.transform = None

    def get_inference_parameters(self):

        return self.embedding_emulator_metadata['train_settings']['data']['params_for_embedding']

    def __getitem__(self, idx):
        size = np.random.randint(
            low=self.minimum_population_size, high=self.maximum_population_size + 1
        )
        hp = self.population_prior.sample()
        
        generate_event_func = self.population_model.get_event_generator(hp, self.kwargs_selection_cut)

        is_training = self.mode in ["train", "test"]
        parameters = generate_event_func(size=self.size_all_events, buffer_factor=55, train=is_training)
        
        # Prepare output, consisting of hyperparameters and an array of embeddings.
        sample = {
            "hyperparameters": hp,
            "parameters": parameters,
            "size": size,
        }
        if self.transform is not None:
            # Transform could prepare for NN.
            return self.transform(sample)
        else:
            return sample

    def __len__(self):
        return self.size

    @property
    def embedding_size(self):
        return self.embedding_emulator_metadata['train_settings']['model']['input_dim']

    @property
    def base_settings(self):
        return self.embedding_emulator_metadata

    def init_epoch(self):
        # TODO: delete?
        pass

    def hyperparameter_mean_std(self):
        mean, std = {}, {}

        print(self.population_prior)

        for k in self.population_prior.keys():
            mean[k], std[k] = calculate_mean_and_std(self.population_prior[k])

        return {"mean": mean, "std": std}


def construct_population_dataset(embedding_emulator_path, device, **kwargs):
    
    embedding_emulator = EmbeddingEmulator(model_filename=embedding_emulator_path, device=device)
    
    return PopulationDataset(
        embedding_emulator_metadata=embedding_emulator.metadata, 
        **kwargs
    )


# def build_transforms(embedding_emulator):

#     inference_parameters = embedding_emulator.get_inference_parameters()
#     standardization_dict = embedding_emulator.metadata['settings_pm_single_event']["train_settings"]["data"]["standardization"]

#     transform_post = SelectStandardizeRepackageParameters(
#         {"inference_parameters": inference_parameters},
#         standardization_dict,
#         inverse=False,
#         as_type="dict",
#     )

#     transforms = [
#         transform_post,
#         UnpackDict(inference_parameters),
#     ]

#     return torchvision.transforms.Compose(transforms)