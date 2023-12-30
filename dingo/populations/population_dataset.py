import copy

import numpy as np
import pandas as pd
from bilby.core.prior import Constraint
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
from bilby.gw.prior import BBHPriorDict
from scipy.spatial import cKDTree

from dingo.core.dataset import DingoDataset

DATA_KEYS = ["parameters", "embeddings"]


class PopulationDataset(DingoDataset):
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

    def initialize_nearest_neighbors(self, search_parameters):
        # Extend parameters as necessary
        if "mass_1" in search_parameters and "mass_1" not in self.parameters:
            mass_1, mass_2 = chirp_mass_and_mass_ratio_to_component_masses(
                self.parameters["chirp_mass"], self.parameters["mass_ratio"]
            )
            self.parameters["mass_1"] = mass_1
            self.parameters["mass_2"] = mass_2

        self.mean = self.parameters.mean().to_dict()
        self.std = self.parameters.std().to_dict()

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

    def sample_nearest_subpopulation(self, desired_parameters):
        """
        Sample a subpopulation based on proximity to desired parameters.

        Parameters
        ----------
        desired_parameters : dict[np.array]
            Desired subpopulation parameters

        Returns
        -------
        dict[np.array]
        """
        desired_std = []
        for k in self.search_parameters_std:
            desired_std.append((desired_parameters[k] - self.mean[k]) / self.std[k])
        _, nearest = self.tree.query(np.vstack(desired_std).T)
        # Impose a check that the distances aren't too far?
        # parameters = self.parameters.iloc[nearest]
        embeddings = self.embeddings[nearest]
        return embeddings
