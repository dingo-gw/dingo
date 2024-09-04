from .base_model import BasePosteriorModel

from dingo.core.nn.nsf import (
    create_nsf_with_rb_projection_embedding_net,
    create_nsf_wrapped,
)


class NormalizingFlowPosteriorModel(BasePosteriorModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize_network(self):
        model_kwargs = {k: v for k, v in self.model_kwargs.items() if k != "type"}
        if self.initial_weights is not None:
            model_kwargs["initial_weights"] = self.initial_weights

        if self.model_kwargs.get("embedding_kwargs", False):
            self.network = create_nsf_with_rb_projection_embedding_net(**model_kwargs)
        else:
            self.network = create_nsf_wrapped(**model_kwargs["posterior_kwargs"])

    def log_prob(self, theta, *context):
        return self.network(theta, *context)

    def sample(self, *context, num_samples: int = 1):
        return self.network.sample(*context, num_samples=num_samples)

    def sample_and_log_prob(self, *context, num_samples: int = 1):
        return self.network.sample_and_log_prob(*context, num_samples=num_samples)

    def loss(self, theta, *context):
        return -self.network(theta, *context).mean()
