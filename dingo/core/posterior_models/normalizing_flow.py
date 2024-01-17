from .base_model import Base

from dingo.core.nn.nsf import (
    create_nsf_with_rb_projection_embedding_net,
    create_nsf_wrapped,
)


class NormalizingFlow(Base):
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

    def log_prob_batch(self, y, *context_data):
        return self.network(y, *context_data)

    def sample_batch(self, *context_data):
        samples = self.network.sample(*context_data)
        return samples

    def sample_and_log_prob_batch(self, *context_data):
        samples, log_probs = self.network.sample_and_log_prob(*context_data)
        return samples, log_probs

    def loss(self, data, *context_data):
        return -self.network(data, *context_data).mean()
