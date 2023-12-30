import copy

import pandas as pd
import torch.utils.data
from astropy.cosmology import FlatLambdaCDM
from bilby.core.prior import PriorDict, PowerLaw, Constraint
from bilby.gw.conversion import generate_mass_parameters
from bilby.gw.prior import UniformComovingVolume
from pycbc.cosmology import DistToZ

from dingo.populations.population_dataset import PopulationDataset


class PowerLawPopulation(torch.utils.data.Dataset):
    def __init__(self, base_population_path, population_prior, snr_threshold):
        super().__init__()
        self.base_population = PopulationDataset(file_name=base_population_path)
        self.base_population.initialize_nearest_neighbors(
            search_parameters=["mass_1", "mass_2", "luminosity_distance"]
        )

        # Maybe use ConditionalPriorDict
        self.population_prior = PriorDict(copy.deepcopy(population_prior))

        self.snr_threshold = snr_threshold
        self.size = None
        self.hyperparameters = None
        self.transform = None

    def __getitem__(self, idx):
        size = 10  # How do we choose the size? It can be variable.
        exact_population = self.generate_population(idx, size)
        embeddings = self.base_population.sample_nearest_subpopulation(exact_population)
        sample = {
            "hyperparameters": self.hyperparameters.iloc[idx].to_dict(),
            "embeddings": embeddings,
        }
        if self.transform is not None:
            # Transform could perform SNR cuts, prepare for NN.
            return self.transform(sample)
        else:
            return sample

    def __len__(self):
        return len(self.hyperparameters)

    def sample_hyperparameters(self, num_samples):
        self.hyperparameters = pd.DataFrame(self.population_prior.sample(num_samples))

    def generate_population(self, population_idx, size):
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
                "luminosity_distance": UniformComovingVolume(
                    minimum=minimum_distance,
                    maximum=maximum_distance,
                    cosmology=cosmology,
                    name="luminosity_distance",
                ),
                "mass_ratio": Constraint(minimum=0.125, maximum=1.0),
            },
            conversion_function=lambda x: generate_mass_parameters(x, source=True),
        )
        samples = prior.sample(size)

        # We use the PyCBC class DistToZ, which is much faster than using the astropy
        # function for z(d_L) directly, since it interpolates.
        dist_to_z = DistToZ(cosmology=cosmology)
        samples["redshift"] = dist_to_z.get_redshift(samples["luminosity_distance"])
        # samples["redshift"] = luminosity_distance_to_redshift(
        #     samples["luminosity_distance"], cosmology=cosmology
        # )
        for k in ["mass_1", "mass_2"]:
            samples[k] = samples[k + "_source"] * (1 + samples["redshift"])
        return samples


def build_population_model(settings):
    population_model = settings["population_model"]
    kwargs = {k: v for k, v in settings.items() if k != "population_model"}
    if population_model == "power_law":
        return PowerLawPopulation(**kwargs)
