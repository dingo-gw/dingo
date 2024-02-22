import copy

import numpy as np
import pandas as pd
import torch.utils.data
from bilby.core.prior import Constraint
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
from bilby.gw.prior import BBHPriorDict
from scipy.spatial import cKDTree

from dingo.core.dataset import DingoDataset
from dingo.populations.population_models import PowerLawPopulation

DATA_KEYS = ["parameters", "embeddings"]


class EventEmbeddingsDataset(DingoDataset):
    """
    Stores a population of GW events. This consists of a set of parameters and
    associated single-event embeddings. The population parameters are typically draws
    from a prior. This is intended to hold a very large population covering the full
    range of possible event parameters, which will later be sub-sampled for specific
    population hyperparameters.

    The class contains functionality for nearest-neighbor sub-sampling.
    """

    dataset_type = "population_dataset"

    def __init__(
        self,
        file_name=None,
        dictionary=None,
    ):
        self.parameters = None
        self.embeddings = None
        self.prior = None
        self.search_parameters_std = None
        self.mean = None
        self.std = None
        self.tree = None
        super().__init__(
            file_name=file_name, dictionary=dictionary, data_keys=DATA_KEYS
        )

    def build_prior(self):
        self.prior = BBHPriorDict(copy.deepcopy(self.settings.get("prior", {})))

    @property
    def embedding_size(self):
        if self.embeddings is not None:
            return self.embeddings.shape[-1]

    def initialize_nearest_neighbors(self, search_parameters):
        # Extend parameters as necessary, e.g., component masses from chirp mass and
        # mass ratio.
        if "mass_1" in search_parameters and "mass_1" not in self.parameters:
            mass_1, mass_2 = chirp_mass_and_mass_ratio_to_component_masses(
                self.parameters["chirp_mass"], self.parameters["mass_ratio"]
            )
            self.parameters["mass_1"] = mass_1
            self.parameters["mass_2"] = mass_2

        self.mean = self.parameters.mean().to_dict()
        self.std = self.parameters.std().to_dict()

        # Standardize the search parameters since Euclidean distance is used for
        # measuring nearest neighbor proximity.
        self.search_parameters_std = pd.DataFrame()
        for p in search_parameters:
            self.search_parameters_std[p] = (
                self.parameters[p] - self.mean[p]
            ) / self.std[p]

        self.tree = cKDTree(self.search_parameters_std)

    def store_log_prior(self):
        if self.prior is None:
            self.build_prior()
        param_keys = [k for k, v in self.prior.items() if not isinstance(v, Constraint)]
        theta = self.parameters[param_keys]
        self.parameters["log_prior"] = self.prior.ln_prob(theta, axis=0)

    def sample_nearest(self, desired_parameters):
        # The cKDTree is based on standardized parameters.
        desired_std = np.array(
            [
                (desired_parameters[k] - self.mean[k]) / self.std[k]
                for k in self.search_parameters_std
            ]
        )
        d, i = self.tree.query(desired_std)
        # TODO: Raise error if d > threshold distance? This would indicate that the
        #  coverage of the base population is too sparse. Alternatively, generate new
        #  samples for the base population.
        return self.parameters.iloc[i].to_dict(), self.embeddings[i]

    def restrict_to_subpopulation(self, s):
        """
        Restricts to a subpopulation of events, as defined by the slice s. This is
        useful for defining training and eval sets.

        Parameters
        ----------
        s : slice
            Slicing object to define subpopulation
        """
        self.parameters = self.parameters.iloc[s].reset_index()
        self.embeddings = self.embeddings[s].copy()
        self.settings["size"] = len(self.parameters)


class PopulationDataset(torch.utils.data.Dataset):
    """
    Training dataset for population posterior that is based on stored embeddings.

    Contains a PopulationModel and a EventEmbeddingsDataset.
    """

    def __init__(
        self,
        event_embeddings_path,
        population_model,
        population_prior,
        snr_threshold,
        minimum_population_size,
        maximum_population_size,
        size,
        train_fraction,
        mode=None,
    ):
        super().__init__()

        self.event_embeddings = EventEmbeddingsDataset(file_name=event_embeddings_path)

        # Depending on whether we are in train or test mode, we take even or odd
        # elements of the base population (half for each).
        if mode == "train":
            self.event_embeddings.restrict_to_subpopulation(slice(0, None, 2))
            self.size = int(size * train_fraction)
        elif mode == "test":
            self.event_embeddings.restrict_to_subpopulation(slice(1, None, 2))
            self.size = int(size * (1 - train_fraction))

        self.event_embeddings.build_prior()

        if population_model == "power_law":
            minimum_distance = self.event_embeddings.prior[
                "luminosity_distance"
            ].minimum
            maximum_distance = self.event_embeddings.prior[
                "luminosity_distance"
            ].maximum
            self.population_model = PowerLawPopulation(
                population_prior, minimum_distance, maximum_distance
            )
        else:
            raise NotImplementedError(
                f"Population model {population_model} is not " f"implemented."
            )

        self.event_embeddings.initialize_nearest_neighbors(
            search_parameters=self.population_model.event_parameters
        )

        self.snr_threshold = snr_threshold
        self.minimum_population_size = minimum_population_size
        self.maximum_population_size = maximum_population_size

        self.hyperparameters = None
        self.sample_hyperparameters()

        self.transform = None

    def __getitem__(self, idx):
        size = np.random.randint(
            low=self.minimum_population_size, high=self.maximum_population_size + 1
        )
        p = self.hyperparameters.iloc[idx]
        generate_event_func = self.population_model.get_event_generator(p)

        # Keep generating events until the desired size is reached.
        embeddings = np.empty((size, self.embedding_size))
        n = 0
        tries = 0
        while n < size:
            p = generate_event_func()
            p_nearest, emb = self.event_embeddings.sample_nearest(p)
            if p_nearest["matched_filter_snr"] >= self.snr_threshold:
                embeddings[n] = emb
                n += 1
            tries += 1
            if tries / (n + 1) > 100:
                raise ValueError("Sampling efficiency < 1%")

        # Prepare output, consisting of hyperparameters and an array of embeddings.
        sample = {
            "hyperparameters": self.hyperparameters.iloc[idx].to_dict(),
            "embeddings": embeddings,
        }
        if self.transform is not None:
            # Transform could prepare for NN.
            return self.transform(sample)
        else:
            return sample

    def __len__(self):
        return len(self.hyperparameters)

    @property
    def embedding_size(self):
        return self.event_embeddings.embedding_size

    @property
    def num_hyperparameters(self):
        return self.hyperparameters.shape[-1]

    @property
    def base_settings(self):
        return self.event_embeddings.settings

    def init_epoch(self):
        self.sample_hyperparameters()

    def sample_hyperparameters(self):
        print("Generating new set of population hyperparameters for dataset.")
        self.hyperparameters = pd.DataFrame(
            self.population_model.prior.sample(self.size)
        )

    def hyperparameter_mean_std(self):
        mean = self.hyperparameters.mean().to_dict()
        std = self.hyperparameters.std().to_dict()
        return {"mean": mean, "std": std}
