import copy
from os.path import join

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
from dingo.populations.models_training.embedding_sampler import EmbeddingSampler
from dingo.populations.models_training.snr_estimator import SNREstimator
from dingo.gw.injection import Injection

from bilby.core.prior import Uniform, PriorDict
from dingo.populations.utilities import (
    build_bilby_prior_dict,
    calculate_mean_and_std
)


class PopulationDataset(torch.utils.data.Dataset):
    """
    Training dataset for population posterior that is based on stored embeddings.

    Contains a PopulationModel and a EventEmbeddingsDataset.
    """

    def __init__(
        self,
        embedding_emulator,
        snr_model,
        population_model_name,
        population_prior_dict,
        snr_threshold,
        minimum_population_size,
        maximum_population_size,
        mode=None,
        kwargs_selection_cut={},
        size=None,
        train_fraction=0.9,
        **kwargs,  # These are ignored (particularly 'standardization'). Not clean.
    ):
        super().__init__()

        # TODO: Should we build the train transforms from within this class? Might be
        #  better for treating the arguments.

        self.embedding_emulator = embedding_emulator
        self.snr_model = snr_model

        self.population_prior = build_bilby_prior_dict(population_prior_dict)

        # if non-empty dictionary, events that do not pass this cut are directly discarded
        self.kwargs_selection_cut = kwargs_selection_cut

        # Depending on whether we are in train or test mode, we take even or odd
        # elements of the base population (half for each).
        if mode == "train":
            # TODO, separate here? 
            self.size = int(size * train_fraction)
        elif mode == "test":
            self.size = int(size * (1 - train_fraction))
        else:
            raise ValueError(f"Mode {mode} not recognized.")

        # TODO, write here checks that the priors of the embedding emulator 
        # is not smaller than the population prior

        # TODO replace here later with better metadata
        inj = Injection.from_posterior_model_metadata(self.embedding_emulator.pm_single_event.metadata)
        self.population_model = build_population_model(
            population_model_name, self.population_prior, inj.prior
        )

        self.snr_threshold = snr_threshold
        self.minimum_population_size = minimum_population_size
        self.maximum_population_size = maximum_population_size

        self.hyperparameters = None

        self.num_hyperparameters = len(self.population_prior)

        self.transform = None

    def __getitem__(self, idx):
        size = np.random.randint(
            low=self.minimum_population_size, high=self.maximum_population_size + 1
        )
        hp = self.population_prior.sample()
        generate_event_func = self.population_model.get_event_generator(hp, self.kwargs_selection_cut)

        # Keep generating events until the desired size is reached.
        embeddings = torch.zeros((size, self.embedding_size))
        n = 0
        tries = 0

        while n < size:

            # TODO write s.t. generate event function can take a batch_size
            p = generate_event_func()
            emb = self.embedding_emulator.sample(p, batch_size=1)

            mf_snr = self.snr_model(emb)

            if mf_snr >= self.snr_threshold:

                embeddings[n] = emb
                n += 1
            
            # found = (mf_snr >= self.snr_threshold)
            # n_found = len(found)
            
            # idx = slice(n,n+n_found)
            # embeddings[idx] = emb[found,:]   
            
            tries += 1
            
            if tries / (n + 1) > 10000:
                raise ValueError("Sampling efficiency < 0.01%")

        # Prepare output, consisting of hyperparameters and an array of embeddings.
        sample = {
            "hyperparameters": hp,
            "embeddings": embeddings,
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
        return self.embedding_emulator.model_kwargs['input_dim']

    @property
    def base_settings(self):
        return self.embedding_emulator.metadata

    def init_epoch(self):
        # TODO: delete?
        pass

    def hyperparameter_mean_std(self):
        mean, std = {}, {}

        print(self.population_prior)

        for k in self.population_prior.keys():
            mean[k], std[k] = calculate_mean_and_std(self.population_prior[k])

        return {"mean": mean, "std": std}


def construct_population_dataset(embedding_emulator_path, snr_model_path, device, **kwargs):
    
    embedding_emulator = EmbeddingSampler(embedding_emulator_path, device='cpu')
    embedding_emulator.add_pm_single_event()

    snr_model = SNREstimator(snr_model_path, device='cpu')
    snr_model.model.eval()

    return PopulationDataset(
        embedding_emulator, 
        snr_model, 
        **kwargs
    )
