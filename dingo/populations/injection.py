import numpy as np

from dingo.gw.injection import Injection as EventInjection
from dingo.populations.population_models import PowerLawPopulation


# This duplicates some code from the PopulationDataset class. It would be nice to find
# a clean way to avoid duplication.


class Injection(object):
    """
    Produces injections of GW populations. Each population consists of a choice of
    hyperparameters and a collection of strain data sets corresponding to events drawn
    from that population. The population hyperparameters and population size can either
    be specified or chosen at random.

    The Injection class can be instantiated based on a population posterior model. In
    that case, the associated population forward model is used for generating the
    population, and the strain data sets are made to be consistent with the generative
    process used for the event embedding network.
    """

    def __init__(
        self,
        population_model,
        population_prior,
        snr_threshold,
        minimum_population_size,
        maximum_population_size,
        event_injection,
    ):
        self.snr_threshold = snr_threshold
        self.minimum_population_size = minimum_population_size
        self.maximum_population_size = maximum_population_size
        self.event_injection = event_injection

        if population_model == "power_law":
            self.population_model = PowerLawPopulation(
                population_prior,
                self.event_injection.prior["luminosity_distance"].minimum,
                self.event_injection.prior["luminosity_distance"].maximum,
            )
        else:
            raise NotImplementedError(
                f"Population model {population_model} is not yet " f"implemented."
            )

    @classmethod
    def from_posterior_model_metadata(cls, metadata):
        data_settings = metadata["train_settings"]["data"]
        event_model_metadata = metadata["base_settings"]["full_event_model_metadata"]
        event_injection = EventInjection.from_posterior_model_metadata(
            event_model_metadata
        )
        return cls(
            population_model=data_settings["population_model"],
            population_prior=data_settings["population_prior"],
            snr_threshold=data_settings["snr_threshold"],
            maximum_population_size=data_settings["maximum_population_size"],
            minimum_population_size=data_settings["minimum_population_size"],
            event_injection=event_injection,
        )

    def injection(self, hyperparameters, population_size):
        """
        Generate an injection.

        Parameters
        ----------
        hyperparameters : dict
            Population hyperparameters used for the injection.
        population_size : int
            Number of events in the population.

        Returns
        -------
        dict
            keys:
                hyperparameters
                parameters
                strain
                asd
        """
        generate_event_func = self.population_model.get_event_generator(hyperparameters)

        # Keep generating events until the desired size is reached.
        events = []
        tries = 0
        while len(events) < population_size:
            # Sample event parameters from the population likelihood, and fall back to
            # the usual GW event prior for the remaining parameters.
            p = self.event_injection.prior.sample()
            p.update(generate_event_func())
            p = {k: float(v) for k, v in p.items()}
            event = self.event_injection.injection(p, store_snr=True)
            if event["parameters"]["matched_filter_snr"] >= self.snr_threshold:
                events.append(event)
            tries += 1
            if tries / (len(events) + 1) > 100:
                raise ValueError("Sampling efficiency < 1%")

        # Collate the events and return along with hyperparameters.
        injection = {"hyperparameters": hyperparameters}
        for k in ["waveform", "parameters", "asds"]:
            injection[k] = {}
            for k2 in events[0][k]:
                injection[k][k2] = np.array([e[k][k2] for e in events])
        return injection

    def random_injection(self):
        """
        Generate a random injection.

        Returns
        -------
        dict
            keys:
                hyperparameters
                parameters
                strain
                asd
        """
        population_size = np.random.randint(
            low=self.minimum_population_size, high=self.maximum_population_size + 1
        )
        hyperparameters = self.population_model.prior.sample()
        return self.injection(hyperparameters, population_size)
