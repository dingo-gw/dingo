import torch
import numpy as np
from abc import abstractmethod

from torchdiffeq import odeint
from glasflow.nflows.utils.torchutils import repeat_rows, split_leading_dim

from .base_model import BasePosteriorModel

from dingo.core.nn.cfnets import create_cf


class ContinuousFlowPosteriorModel(BasePosteriorModel):
    """
    Class for posterior models based on continuous normalizing flows (CNFs).

    CNFs are parameterized by a vector field v(theta_t, t), that transports a simple
    base distribution (typically a gaussian N(0,1) with same dimension as theta) at
    time t=0 to the target distribution at time t=1. This vector field defines the flow
    via the ODE

                    d/dt f(theta, t) = v(f(theta, t), t).

    The vector field v is parameterized with a neural network. It is impractical to train
    this neural network (and thereby the CNF) directly with log-likelihood maximization,
    as solving the full ODE for each training iteration, requires thousands of
    vector field evaluations.

    Several alternative methods have been developed to make training CNFs more
    efficient. These directly regress on the vector field v (or a scaled version of v,
    such as the score). It has been shown that this can be done on a per-sample basis
    by adding noise to the parameters at various scales t. Specifically, a parameter
    sample theta is transformed as follows.

        t               ~ U[0, 1-eps)                               noise level
        theta_0         ~ N(0, 1)                                   sampled noise
        theta_1         = theta                                     pure sample
        theta_t         = c1(t) * theta_1 + c0(t) * theta_0         noisy sample

    Within that framework, one can employ different methods to learn the vector field v,
    such as flow matching or score matching. These have slightly different coefficients
    c1(t), c2(t) and training objectives.

    This class is intended to construct and hold a neural network for estimating the
    posterior density, as well as saving / loading, and training. It also has
    functionality for sampling and density evaluation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eps = 0
        self.time_prior_exponent = self.model_kwargs["posterior_kwargs"][
            "time_prior_exponent"
        ]
        self.theta_dim = self.metadata["train_settings"]["model"]["posterior_kwargs"][
            "input_dim"
        ]

    def sample_t(self, batch_size):
        t = (1 - self.eps) * torch.rand(batch_size, device=self.device)
        return t.pow(1 / (1 + self.time_prior_exponent))

    def sample_theta_0(self, batch_size):
        """Sample theta_0 from the gaussian prior."""
        return torch.randn(batch_size, self.theta_dim, device=self.device)

    @abstractmethod
    def evaluate_vector_field(self, t, theta_t, *context_data):
        """
        Evaluate the vector field v(t, theta_t, context_data) that generates the flow
        via the ODE

            d/dt f(theta_t, t, context) = v(f(theta_t, t, context), t, context).

        Parameters
        ----------
        t: float
            time (noise level)
        theta_t: torch.Tensor
            noisy parameters, perturbed with noise level t
        *context_data: list[torch.tensor]
            list with context data (GW data)
        """
        pass

    def rhs_of_joint_ode(self, t, theta_and_div_t, *context_data, hutchinson=False):
        """
        Returns the right hand side of the neural ODE that is used to evaluate the
        log_prob of theta samples. This is a joint ODE over the vector field and the
        divergence. By integrating this ODE, one can simultaneously trace the parameter
        sample theta_t and integrate the divergence contribution to the log_prob,
        see e.g., https://arxiv.org/abs/1806.07366 or Appendix C in
        https://arxiv.org/abs/2210.02747.

        Parameters
        ----------
        t: float
            time (noise level)
        theta_and_div_t: torch.Tensor
            concatenated tensor of (theta_t, div).
            theta_t: noisy parameters, perturbed with noise level t
        *context_data: list[torch.tensor]
            list with context data (GW data)

        Returns
        -------
        torch.Tensor
            vector field that generates the flow and its divergence (required for
            likelihood evaluation).
        """
        divergence_fun = compute_divergence
        if hutchinson:
            divergence_fun = compute_hutchinson_divergence

        theta_t = theta_and_div_t[:, :-1].detach()  # extract theta_t
        # context_data = context_data[0].detach().clone()
        t = t.detach()
        with torch.enable_grad():
            theta_t.requires_grad_(True)
            vf = self.evaluate_vector_field(t, theta_t, *context_data)
            div_vf = divergence_fun(vf, theta_t)
        return torch.cat((vf, -div_vf), dim=1)

    def initialize_network(self):
        model_kwargs = {
            k: v for k, v in self.model_kwargs.items() if k != "posterior_model_type"
        }
        if self.initial_weights is not None:
            model_kwargs["initial_weights"] = self.initial_weights
        self.network = create_cf(**model_kwargs)

    def sample(self, *context: torch.Tensor, num_samples: int = None):
        """
        Sample parameters theta from the posterior model,

        theta ~ p(theta | context)

        by solving an ODE forward in time.

        Parameters
        ----------
        context: torch.Tensor
            Context information (typically observed data). Should have a batch
            dimension (even if size B = 1).
        num_samples: int = 1
            Number of samples to generate.

        Returns
        -------
        samples: torch.Tensor
            Shape (B, num_samples, dim(theta))
        """
        context_size = context[0].shape[0]
        theta_0 = self.sample_theta_0(num_samples * context_size)
        context = [repeat_rows(c, num_reps=num_samples) for c in context]

        _, theta_1 = odeint(
            lambda t, theta_t: self.evaluate_vector_field(t, theta_t, *context),
            theta_0,
            self.integration_range,
            atol=1e-7,
            rtol=1e-7,
            method="dopri5",
        )

        theta_1 = split_leading_dim(theta_1, shape=[context_size, num_samples])

        self.network.train()
        return theta_1

    # MD: rename log_prob_batch, extract eps from self.epsilon_ode_integration
    def log_prob(self, theta: torch.Tensor, *context: torch.Tensor, hutchinson=False):
        """
        Evaluate the log posterior density,

        log p(theta | context)

        For this we solve an ODE backwards in time until we reach the initial pure
        noise distribution.

        There are two contributions, the log_prob of theta_0 (which is uniquely
        determined by theta) under the base distribution, and the integrated divergence
        of the vector field.

        Parameters
        ----------
        theta: torch.Tensor
            Parameter values at which to evaluate the density. Should have a batch
            dimension (even if size B = 1).
        context: torch.Tensor
            Context information (typically observed data). Must have context.shape[0] = B.
        hutchinson

        Returns
        -------
        log_prob: torch.Tensor
            Shape (B,)
        """
        theta_and_div_init = torch.cat(
            (theta, torch.zeros((theta.shape[0],), device=theta.device).unsqueeze(1)),
            dim=1,
        )
        _, theta_and_div_0 = odeint(
            lambda t, theta_and_div_t: self.rhs_of_joint_ode(
                t, theta_and_div_t, *context, hutchinson=hutchinson
            ),
            theta_and_div_init,
            torch.flip(
                self.integration_range, dims=(0,)
            ),  # integrate backwards in time, [1-eps, 0]
            atol=1e-7,
            rtol=1e-7,
            options={"norm": norm_without_divergence_component},
            method="dopri5",
        )
        # return log p_0(x_0) - divergence|0
        theta_0, divergence = theta_and_div_0[:, :-1], theta_and_div_0[:, -1]
        log_prior = compute_log_prior(theta_0)
        return (log_prior - divergence).detach()

    def sample_and_log_prob(self, *context: torch.Tensor, num_samples: int = None):
        """
        Sample parameters theta from the posterior model,

        theta ~ p(theta | context)

        and also return the log_prob. This is more efficient than calling sample_batch
        and log_prob_batch separately.

        If d/dt [phi(t), f(t)] = rhs joint with initial conditions [theta_0,
        log p(theta_0)], where theta_0 ~ p_0(theta_0), then [phi(1), f(1)] = [theta_1,
        log p(theta_0) + log p_1(theta_1) - log p(theta_0)] = [theta_1, log p_1(theta_1)].

        Parameters
        ----------
        context: torch.Tensor
            Context information (typically observed data). Should have a batch
            dimension (even if size B = 1).
        num_samples: int = 1
            Number of samples to generate.

        Returns
        -------
        samples, log_prob: torch.Tensor, torch.Tensor
            Shapes (B, num_samples, dim(theta)), (B, num_samples)
        """

        context_size = context[0].shape[0]
        theta_0 = self.sample_theta_0(num_samples * context_size)
        context = [repeat_rows(c, num_reps=num_samples) for c in context]

        log_prior = compute_log_prior(theta_0)
        theta_and_div_init = torch.cat((theta_0, log_prior.unsqueeze(1)), dim=1)

        _, theta_and_div_1 = odeint(
            lambda t, theta_and_div_t: self.rhs_of_joint_ode(
                t, theta_and_div_t, *context
            ),
            theta_and_div_init,
            self.integration_range,  # integrate forwards in time, [0, 1-eps]
            atol=1e-7,
            rtol=1e-7,
            method="dopri5",
            options={"norm": norm_without_divergence_component},
        )

        theta_1, log_prob_1 = theta_and_div_1[:, :-1], theta_and_div_1[:, -1]

        theta_1 = split_leading_dim(theta_1, shape=[context_size, num_samples])
        log_prob_1 = split_leading_dim(log_prob_1, shape=[context_size, num_samples])

        return theta_1, log_prob_1

    @property
    def integration_range(self):
        """
        Integration range for ODE. We integrate in the range [0, 1-self.eps]. For score
        matching, self.eps > 0 is required for stability. For flow matching we can have
        self.eps = 0.
        """
        return torch.tensor([0.0, 1.0 - self.eps]).type(torch.float32).to(self.device)


def compute_divergence(y, x):
    div = 0.0
    with torch.enable_grad():
        y.requires_grad_(True)
        x.requires_grad_(True)
        for i in range(y.shape[-1]):
            div += torch.autograd.grad(
                y[..., i], x, torch.ones_like(y[..., i]), retain_graph=True
            )[0][..., i : i + 1]
        return div


def compute_hutchinson_divergence(y, x):
    with torch.enable_grad():
        x.requires_grad_(True)
        eps = torch.randn_like(y)
        y_cdot_eps = torch.sum(eps * y)
        x.retain_grad()
        y_cdot_eps.backward()
        return torch.sum(eps * x.grad, dim=-1, keepdim=True)


def norm_without_divergence_component(y):
    return torch.linalg.norm(y[:, :-1])


def compute_log_prior(theta_0):
    N = theta_0.shape[1]
    return -N / 2.0 * np.log(2 * np.pi) - torch.sum(theta_0**2, dim=1) / 2.0
