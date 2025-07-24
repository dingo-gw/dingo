import numpy as np
import torch

from dingo.gw.injection import Injection as EventInjection
from dingo.populations.population_models import build_population_model


from dingo.core.models.posterior_model import PosteriorModel
from dingo.gw.inference.gw_samplers import GWSampler

from dingo.populations.models_training.models import (
    SNREstimator,
    EmbeddingEmulator
) 

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
        detection_model=None,
    ):

        self.population_prior = build_bilby_prior_dict(population_prior_dict)

        self.mf_snr_threshold = mf_snr_threshold
        self.minimum_population_size = minimum_population_size
        self.maximum_population_size = maximum_population_size
        self.event_injection = event_injection

        if detection_model is None:
            self.detection_model = "mf_snr"

        self.population_model = build_population_model(
            population_model_name, self.population_prior, event_injection.prior
        )

        self.use_embedding_emulator = False

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

    def add_pm_single_event(self, pm_single_event_path, device):

        self.pm_single_event = PosteriorModel(pm_single_event_path, device=device)
        self.pm_single_event.set_embedding_only()
        self.pm_single_event.model.eval()

        self.event_sampler = GWSampler(model=self.pm_single_event)

    def add_snr_model(self, snr_model_path, device):

        self.snr_model = SNREstimator(snr_model_path, device=device)
        self.snr_model.model.eval()

    def add_embedding_emulator(self, embedding_emulator):
        
        self.embedding_emulator = embedding_emulator
        self.embedding_emulator.model.eval()

        self.use_embedding_emulator = True

        # we have to use the snr model, since the SNR is not known
        self.detection_model = "snr_model"
        
        print('Using embedding emulator')

    def injection(self, hyperparameters, population_size, store_total_generated_number=False, train=True):
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
            new_p = generate_event_func(size=1, train=train)
            p.update(new_p)
            p = {k: float(v) for k, v in p.items()}

            if not self.use_embedding_emulator:
                event = self.event_injection.injection(p, store_snr=True)
                if self.is_detected(event=event):
                    events.append(event)
            else:
                embeddings = self.get_embeddings_from_emulator(p, device=self.embedding_emulator.device)
                if self.is_detected(embedding=embeddings):
                    events.append(embeddings)
            tries += 1
            if tries / (len(events) + 1) > 50000:
                print('hyperparameters: ', hyperparameters)
                raise ValueError("Sampling efficiency < 0.002%")

        # Collate the events and return along with hyperparameters.
        injection = {"hyperparameters": hyperparameters}

        if(store_total_generated_number):
            injection['total_generated_events'] = tries

        if not self.use_embedding_emulator:
            for k in ["waveform", "parameters", "asds"]:
                injection[k] = {}
                for k2 in events[0][k]:
                    injection[k][k2] = np.array([e[k][k2] for e in events])
        else:
            injection["embeddings"] = torch.stack(events)

        return injection

    def is_detected(self, event=None, embedding=None):
        """
        Check whether an event is detected. 
        
        Parameters
        
        """

        if event is None and embedding is None:
            raise ValueError("Either event or embedding must be provided.")
        if event is not None and embedding is not None:
            raise ValueError("Only one of event or embedding can be provided.")

        if event is not None:
            return self.is_detected_from_event(event)
        if embedding is not None:
            return self.is_detected_from_embedding(embedding)

    def is_detected_from_embedding(self, embedding):

        with torch.no_grad():
            mf_snr = self.snr_model(embedding).squeeze()

        return mf_snr >= self.mf_snr_threshold

    def is_detected_from_event(self, event):

        if self.detection_model == "mf_snr":
            return event["parameters"]["matched_filter_snr"] >= self.mf_snr_threshold
        elif self.detection_model == "snr_model":
            data = self.event_sampler.transform_pre(event)

            with torch.no_grad():
                emb = self.pm_single_event.model(data.unsqueeze(0))
                return self.is_detected_from_embedding(embedding=emb)

        else:
            raise ValueError(f"Unknown detection model: {self.detection_model}")

    def get_embeddings_from_emulator(self, params, device='cpu'):
        
        p = params
        p = {k:torch.tensor(v) for k,v in p.items()}
        embeddings = self.embedding_emulator.sample_from_params(params=p, num_samples=1, device=device)

        return embeddings

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
