import copy

from astropy.cosmology import FlatLambdaCDM
from bilby.core.prior import PriorDict, PowerLaw, Constraint
from bilby.gw.conversion import (
    generate_mass_parameters,
    component_masses_to_mass_ratio,
    component_masses_to_chirp_mass,
)
from bilby.gw.prior import UniformSourceFrame
from pycbc.cosmology import DistToZ


class PowerLawPopulation(object):
    """
    Describes the prior $p(\Lambda)$ and likelihood $p(\theta | \Lambda)$ of a power
    law population model.
    """

    model_type = "power_law"
    event_parameters = ["mass_1", "mass_2", "luminosity_distance"]

    def __init__(self, prior, minimum_distance, maximum_distance):
        self.prior = PriorDict(copy.deepcopy(prior))
        self.minimum_distance = minimum_distance
        self.maximum_distance = maximum_distance

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
            s['ln_prob'] = prior.ln_prob(s)
            s["redshift"] = dist_to_z.get_redshift(s["luminosity_distance"])
            for k in ["mass_1", "mass_2"]:
                s[k] = s[k + "_source"] * (1 + s["redshift"])
            # IMPORTANT: We want all the mass parameters in order to avoid any problems
            # combining with parameters sampled from the prior. We are leaving off
            # things like total mass, symmetric mass ratio, because they don't tend to
            # occur in our code. But this is something to watch for.
            s["chirp_mass"] = component_masses_to_chirp_mass(s["mass_1"], s["mass_2"])
            s["mass_ratio"] = component_masses_to_mass_ratio(s["mass_1"], s["mass_2"])
            return s

        return generation_func

class PowerLawPeakPopulation(object):
    """
    Describes the prior $p(\Lambda)$ and likelihood $p(\theta | \Lambda)$ of a power
    law+peak population model.
    """

    model_type = "power_law_peak"
    event_parameters = ["mass_1", "mass_2", "luminosity_distance"]

    def __init__(self, prior, minimum_distance, maximum_distance):
        self.prior = PriorDict(copy.deepcopy(prior))
        self.minimum_distance = minimum_distance
        self.maximum_distance = maximum_distance

    def get_event_generator(self, p):
        cosmology = FlatLambdaCDM(Om0=0.3, H0=p["hubble_constant"])
        prior = PriorDict(
            {
                "mass_1_source_q": PowerLawPeak(
                    alpha=-p["alpha"],
                    minimum=p["minimum_mass"],
                    maximum=p["maximum_mass"],
                    delta=p["delta"],
                    lam=p["lam"],
                    mu=p["mu"],
                    sigma=p["sigma"],
                    beta=p["beta"],
                    qlow=0.125,
                ),
                "luminosity_distance": UniformSourceFrame(
                    minimum=self.minimum_distance,
                    maximum=self.maximum_distance,
                    cosmology=cosmology,
                    name="luminosity_distance",
                ),
            },
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
            s0 = prior.sample()
            s={}
            s["mass_1_source"]=s0["mass_1_source_q"][0]
            s["mass_2_source"]=s0["mass_1_source_q"][1]*s0["mass_1_source_q"][0]
            s["luminosity_distance"] = s0["luminosity_distance"]
            s["redshift"] = dist_to_z.get_redshift(s0["luminosity_distance"])
            for k in ["mass_1", "mass_2"]:
                s[k] = s[k + "_source"] * (1 + s["redshift"])
            # IMPORTANT: We want all the mass parameters in order to avoid any problems
            # combining with parameters sampled from the prior. We are leaving off
            # things like total mass, symmetric mass ratio, because they don't tend to
            # occur in our code. But this is something to watch for.
            s["chirp_mass"] = component_masses_to_chirp_mass(s["mass_1"], s["mass_2"])
            s["mass_ratio"] = component_masses_to_mass_ratio(s["mass_1"], s["mass_2"])
            return s

        return generation_func


def build_population_model(population_model, population_prior, event_model_prior):
    if population_model == "power_law":
        minimum_distance = event_model_prior["luminosity_distance"].minimum
        maximum_distance = event_model_prior["luminosity_distance"].maximum
        return PowerLawPopulation(
            population_prior,
            minimum_distance=minimum_distance,
            maximum_distance=maximum_distance,
        )
    elif population_model == "power_law_peak":
        minimum_distance = event_model_prior["luminosity_distance"].minimum
        maximum_distance = event_model_prior["luminosity_distance"].maximum
        return PowerLawPeakPopulation(
            population_prior,
            minimum_distance=minimum_distance,
            maximum_distance=maximum_distance,
        )
    else:
        raise NotImplementedError(
            f"Population model {population_model} is not " f"implemented."
        )
