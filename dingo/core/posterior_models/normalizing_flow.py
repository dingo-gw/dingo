from .base_model import NeuralDistribution
from dingo.core.registry import NEURAL_DISTRIBUTIONS

from dingo.core.nn.nsf import FlowWrapper, create_nsf_model


@NEURAL_DISTRIBUTIONS.register("normalizing_flow")
class NormalizingFlowPosteriorModel(NeuralDistribution):
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
        embedding_net = self.build_embedding_net()
        kwargs = self.model_kwargs["distribution"]["kwargs"]
        flow = create_nsf_model(
            input_dim=kwargs["theta_dim"],
            context_dim=kwargs["context_dim"],
            num_flow_steps=kwargs["num_flow_steps"],
            base_transform_kwargs=kwargs["base_transform_kwargs"],
        )
        if embedding_net is None:
            self.network = FlowWrapper(flow)
        else:
            self.network = FlowWrapper(flow, embedding_net, embedding_net.input_keys)

    def log_prob(self, theta, context: dict = None):
        return self.network(theta, context)

    def sample(self, context: dict = None, num_samples: int = 1):
        return self.network.sample(context, num_samples=num_samples)

    def sample_and_log_prob(self, context: dict = None, num_samples: int = 1):
        return self.network.sample_and_log_prob(context, num_samples=num_samples)

    def loss(self, theta, context: dict = None):
        return -self.network(theta, context).mean()
