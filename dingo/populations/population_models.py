import copy

import numpy as np
import pandas as pd
import torch.utils.data
from astropy.cosmology import FlatLambdaCDM
from bilby.core.prior import PriorDict, PowerLaw, Constraint
from bilby.gw.conversion import generate_mass_parameters
from bilby.gw.prior import UniformComovingVolume, UniformSourceFrame
from pycbc.cosmology import DistToZ

from dingo.populations.population_dataset import BasePopulationDataset


class PowerLawPopulation(torch.utils.data.Dataset):
    def __init__(
        self,
        base_population_path,
        population_prior,
        snr_threshold,
        minimum_population_size,
        maximum_population_size,
        size,
        train_fraction,
        mode=None,
    ):
        super().__init__()

        self.base_population = BasePopulationDataset(file_name=base_population_path)

        # Depending on whether we are in train or test mode, we take even or odd
        # elements of the base population (half for each).
        if mode == "train":
            self.base_population.restrict_to_subpopulation(slice(0, None, 2))
            self.size = int(size * train_fraction)
        elif mode == "test":
            self.base_population.restrict_to_subpopulation(slice(1, None, 2))
            self.size = int(size * (1 - train_fraction))

        self.base_population.initialize_nearest_neighbors(
            search_parameters=["mass_1", "mass_2", "luminosity_distance"]
        )

        # Maybe use ConditionalPriorDict
        self.population_prior = PriorDict(copy.deepcopy(population_prior))

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
        generate_event_func = self.get_event_generator(idx)

        # Keep generating events until the desired size is reached.
        embeddings = np.empty((size, self.embedding_size))
        n = 0
        tries = 0
        while n < size:
            p = generate_event_func()
            p_nearest, emb = self.base_population.sample_nearest(p)
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
        return self.base_population.embedding_size

    @property
    def num_hyperparameters(self):
        return self.hyperparameters.shape[-1]

    @property
    def base_settings(self):
        return self.base_population.settings

    def init_epoch(self):
        self.sample_hyperparameters()

    def sample_hyperparameters(self):
        print("Generating new set of population hyperparameters for dataset.")
        self.hyperparameters = pd.DataFrame(self.population_prior.sample(self.size))

    def get_event_generator(self, population_idx):
        p = self.hyperparameters.iloc[population_idx]
        if self.base_population.prior is None:
            self.base_population.build_prior()
        minimum_distance = self.base_population.prior["luminosity_distance"].minimum
        maximum_distance = self.base_population.prior["luminosity_distance"].maximum
        cosmology = FlatLambdaCDM(Om0=0.3, H0=p["hubble_constant"])
        prior = PriorDict(
            {
                "mass_1_source": PowerLaw(
                    alpha=-p["alpha"],
                    minimum=p["minimum_mass"],
                    maximum=p["maximum_mass"],
                ),
                "mass_2_source": PowerLaw(
                    alpha=p["beta"],
                    minimum=p["minimum_mass"],
                    maximum=p["maximum_mass"],
                ),
                "luminosity_distance": UniformSourceFrame(
                    minimum=minimum_distance,
                    maximum=maximum_distance,
                    cosmology=cosmology,
                    name="luminosity_distance",
                ),
                "mass_ratio": Constraint(minimum=0.125, maximum=1.0),
            },
            conversion_function=lambda x: generate_mass_parameters(x, source=True),
        )

        # We use the PyCBC class DistToZ, which is much faster than using the astropy
        # function for z(d_L) directly, since it interpolates.
        dist_to_z = DistToZ(cosmology=cosmology)

        # We return the generating function for event parameters for two reasons:
        # (1) Because of selection effects, we don't know a priori how many events we
        # have to generate.
        # (2) Some of the objects (construction of prior, cosmology, DistToZ) are a bit
        # slow to construct, so we should avoid doing so repeatedly for each set of
        # hyperparameters.
        def generation_func():
            s = prior.sample()
            s["redshift"] = dist_to_z.get_redshift(s["luminosity_distance"])
            for k in ["mass_1", "mass_2"]:
                s[k] = s[k + "_source"] * (1 + s["redshift"])
            return s

        return generation_func

    def hyperparameter_mean_std(self):
        mean = self.hyperparameters.mean().to_dict()
        std = self.hyperparameters.std().to_dict()
        return {"mean": mean, "std": std}


def build_population_model(settings, mode=None):
    population_model = settings["population_model"]
    kwargs = {k: v for k, v in settings.items() if k != "population_model"}
    if population_model == "power_law":
        return PowerLawPopulation(**kwargs, mode=mode)
