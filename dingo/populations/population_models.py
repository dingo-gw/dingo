import copy

from astropy.cosmology import FlatLambdaCDM
from bilby.core.prior import PriorDict, PowerLaw, Constraint
from bilby.gw.conversion import generate_mass_parameters
from bilby.gw.prior import UniformSourceFrame
from pycbc.cosmology import DistToZ


class PowerLawPopulation(object):
    """
    Describes the prior $p(\Lambda)$ and likelihood $p(\theta | \Lambda)$ of a power
    law population model.
    """

    def __init__(self, prior, minimum_distance, maximum_distance):
        self.prior = PriorDict(copy.deepcopy(prior))
        self.minimum_distance = minimum_distance
        self.maximum_distance = maximum_distance
        self.event_parameters = ["mass_1", "mass_2", "luminosity_distance"]

    def get_event_generator(self, p):
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
                    minimum=self.minimum_distance,
                    maximum=self.maximum_distance,
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


# def build_population_model(settings, mode=None):
#     population_model = settings["population_model"]
#     kwargs = {k: v for k, v in settings.items() if k != "population_model"}
#     if population_model == "power_law":
#         return PowerLawPopulation(**kwargs, mode=mode)
