from .base_model import BasePosteriorModel

from dingo.core.nn.nsf import (
    create_nsf_with_rb_projection_embedding_net,
    create_nsf_wrapped,
)


class NormalizingFlowPosteriorModel(BasePosteriorModel):
    """
    Posterior model based on a (discrete) normalizing flow.

    A normalizing flow describes a distribution as a sequence of discrete
    transformations on a parameter space, ultimately taking samples from the base space
    (multivariate standard normal) to the desired distribution. The discrete transforms
    are parametrized functions (e.g., splines), which are designed to be invertible with
    simple Jacobian determinant. The probability density is given by the change of
    variables rule,

    q(theta | d) = pi(f_d^{-1}(theta)) | det J_{f_d^{-1}} |

    where
        pi = N(0,1)^D is the base space distribution
        f_d is the normalizing flow on the D-dimensional space

    The flow f_d is allowed to depend on context information d, which would be
    observational data in the case of posterior estimation. By construction, the flow
    has fast sampling and density evaluation, require just forward passes of the network.

    This class uses normalizing flows from the dingo.core.nn.nsf module (which in turn
    uses glasflow, which is based on nflows). It is intended to construct and hold a
    neural network for estimating the posterior density, as well as saving / loading,
    and training. It also calls the sampling and density evaluation routines from the
    flows.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize_network(self):
        model_kwargs = {
            k: v for k, v in self.model_kwargs.items() if k != "posterior_model_type"
        }
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
