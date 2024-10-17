import numpy as np

from dingo.gw.injection import Injection as EventInjection
from dingo.populations.population_models import build_population_model

from dingo.populations.training.population_dataset import build_bilby_prior_dict


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
        population_model_name,
        population_prior_dict,
        mf_snr_threshold,
        minimum_population_size,
        maximum_population_size,
        event_injection,
    ):

        self.population_prior = build_bilby_prior_dict(population_prior_dict)

        self.mf_snr_threshold = mf_snr_threshold
        self.minimum_population_size = minimum_population_size
        self.maximum_population_size = maximum_population_size
        self.event_injection = event_injection

        # self.population_model = build_population_model(
        #     population_model=population_model_name,
        #     population_prior=population_prior,
        #     event_model_prior=self.event_injection.prior,
        # )
        self.population_model = build_population_model(
            population_model_name, self.population_prior, event_injection.prior
        )

    @classmethod
    def from_posterior_model_metadata(cls, metadata):
        data_settings = metadata["train_settings"]["data"]
        event_model_metadata = metadata['embedding_emulator_metadata']['settings_pm_single_event']
        event_injection = EventInjection.from_posterior_model_metadata(
            event_model_metadata
        )
        return cls(
            population_model_name=data_settings["population_model_name"],
            population_prior_dict=data_settings["population_prior_dict"],
            mf_snr_threshold=data_settings["mf_snr_threshold"],
            maximum_population_size=data_settings["maximum_population_size"],
            minimum_population_size=data_settings["minimum_population_size"],
            event_injection=event_injection,
        )

    def injection(self, hyperparameters, population_size, store_total_generated_number=False):
        """
        Generate an injection.

        Parameters
        ----------
        hyperparameters : dict
            Population hyperparameters used for the injection.
        population_size : int
            Number of events in the population.
        store_total_generated_number : boolean
            Whether the total amount of signals is saved.

        Returns
        -------
        dict
            keys:
                hyperparameters
                parameters
                strain
                asd
                total_generated_events (optional, depending on whether store_total_generated_number is true)
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
            if tries / (len(events) + 1) > 10000:
                raise ValueError("Sampling efficiency < 0.01%")

        # Collate the events and return along with hyperparameters.
        injection = {"hyperparameters": hyperparameters}

        if(store_total_generated_number):
            injection['total_generated_events'] = tries

        for k in ["waveform", "parameters", "asds"]:
            injection[k] = {}
            for k2 in events[0][k]:
                injection[k][k2] = np.array([e[k][k2] for e in events])
        return injection

    def random_injection(self, population_size=None):
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
        if population_size is None:
            population_size = np.random.randint(
                low=self.minimum_population_size, high=self.maximum_population_size + 1
            )
        hyperparameters = self.population_model.prior.sample()
        return self.injection(hyperparameters, population_size)
