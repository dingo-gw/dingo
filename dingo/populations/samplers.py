import copy
import pandas as pd
import torch
import torchvision.transforms

from dingo.core.models import PosteriorModel
from dingo.core.transforms import PadMask
from dingo.gw.inference.gw_samplers import GWSampler
from dingo.gw.transforms import (
    SelectStandardizeRepackageParameters,
    StandardizeParameters,
)


# TODO: Place on correct device.


class PopulationSampler(object):
    """
    Sampler class for GW population inference.

    This takes as input a collection of FD strain data sets, and produces samples from
    the population posterior.
    """

    def __init__(
        self, population_posterior_model: PosteriorModel, event_model: PosteriorModel
    ):
        self.population_posterior_model = population_posterior_model
        self.event_model = event_model
        self.metadata = self.population_posterior_model.metadata.copy()
        self.inference_parameters = list(
            self.metadata["train_settings"]["data"]["standardization"]["mean"].keys()
        )

        self.population = None
        self.samples = None

        self.event_sampler = GWSampler(model=self.event_model)
        self.event_model.set_embedding_only()

        self._initialize_transforms()

    def _initialize_transforms(self):
        self.transform_pre = torchvision.transforms.Compose([])

        self.transform_post = SelectStandardizeRepackageParameters(
            {"inference_parameters": self.inference_parameters},
            self.metadata["train_settings"]["data"]["standardization"],
            inverse=True,
            as_type="dict",
        )
        self.transform_post_inv = copy.deepcopy(self.transform_post)
        self.transform_post_inv.inverse = False

    def pass_embeddings_through_transformer(self, embeddings):

         # Pass through the transformer to get the population embedding.
        x = self.transform_pre([embeddings])

        model = self.population_posterior_model.model
        model.eval()
        with torch.no_grad():
            x = [y.unsqueeze(0) for y in x]  # Unsqueeze a batch dimension.
            x = model.embedding_net(*x)

        return x

    def run_sampler(self, num_samples):
        if self.population is None:
            raise ValueError("Set self.population before running sampler.")

        embeddings = self.get_embeddings()

        model = self.population_posterior_model.model
        model.eval()
    
        with torch.no_grad():
            x = self.pass_embeddings_through_transformer(embeddings)

            # TODO: batch this
            y, log_prob = model.flow.sample_and_log_prob(num_samples, context=x)

        samples = {"parameters": y.squeeze(0), "log_prob": log_prob.squeeze(0)}
        samples = self.transform_post(samples)
        result = samples["parameters"]
        result["log_prob"] = samples["log_prob"]

        # Move to CPU if necessary
        if torch.cuda.is_available():
            result = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in result.items()}

        return pd.DataFrame(result)

    def get_log_prob(self, parameters):

        y = self.transform_post_inv({"parameters": parameters})['inference_parameters']
        y = torch.tensor(y).unsqueeze(0)

        if self.population is None:
            raise ValueError("Set self.population before running sampler.")

        embeddings = self.get_embeddings()

        model = self.population_posterior_model.model
        model.eval()

        with torch.no_grad():
            x = self.pass_embeddings_through_transformer(embeddings)
            log_prob = model.flow.log_prob(y, context=x)

        return log_prob

    def get_embeddings(self):

        if "embeddings" in self.population:
            embeddings = self.population["embeddings"]
        else:
            # TODO: Add batch_size option
            data = self.event_sampler.transform_pre(self.population)
            self.event_model.model.eval()
            with torch.no_grad():
                embeddings = self.event_model.model(data)

        return embeddings

    def to_result(self):
        pass
