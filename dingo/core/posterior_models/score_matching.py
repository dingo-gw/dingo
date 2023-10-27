# implementing Score-based diffusion based on VP SDE. Potential TODO: other SDEs, and in in separate file
import torch

from .cflow_base import ContinuousFlowsBase


class ScoreDiffusion(ContinuousFlowsBase):
    """
    Class for continuous normalizing flows trained with score matching.

        t               ~ U[0, 1-eps)                               noise level
        theta_0         ~ N(0, 1)                                   sampled noise
        theta_1         = theta                                     pure sample
        theta_t         = c1(t) * theta_1 + c0(t) * theta_0         noisy sample

        eps             > 0
        c0              = sigma(t)
        c1              = alpha(1-t)

        score_target    = theta_0 / sigma_t
        weight          = 1/2 * {score-matching: sigma(t)^2, score-flow: beta(1-t), ...}
        loss            = || score_target - network(theta_t, t) ||
    """

    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        kwargs: dict
            parameters to build the model and the training objective,
            "embedding_kwargs" specify the embedding net used,
            "posterior_kwargs" specifies arguments for the posterior model network and
            the noise properties used for the diffusion (beta_min, beta_max, epsilon).
        """
        super().__init__(**kwargs)
        self.eps = self.model_kwargs["posterior_kwargs"]["epsilon"]
        self.beta_min = self.model_kwargs["posterior_kwargs"]["beta_min"]
        self.beta_max = self.model_kwargs["posterior_kwargs"]["beta_max"]

        likelihood_weighting = self.model_kwargs["posterior_kwargs"].get(
            "likelihood_weighting", "score-matching"
        )
        if likelihood_weighting:
            self.likelihood_weighting = self.get_likelihood_weighting(
                likelihood_weighting
            )

    def loss(self, theta, *context_data):
        """
        Returns the score matching loss for parameters theta conditioned on context.

        Parameters
        ----------
        theta: torch.tensor
            parameters (e.g., binary-black hole parameters)
        *context_data: list[torch.Tensor]
            context data (e.g., gravitational-wave data)

        Returns
        -------
        torch.tensor
            Loss.
        """
        t, theta_t, score = self.get_t_theta_t_score(theta_1=theta)
        pred_score = self.network(t, theta_t, *context_data)

        # TODO: write as MSE loss
        weighting = self.likelihood_weighting(t)
        losses = torch.square(pred_score - score)
        losses = 1 / 2 * torch.sum(losses, dim=1) * weighting
        loss = torch.mean(losses)
        return loss

    def evaluate_vectorfield(self, t, theta_t, *context_data):
        """
        Vectorfield that generates the flow, see Docstring in ContinuousFlowsBase for
        details. For score matching, the vectorfield (or drift function) is computed
        from the predicted score.
        """
        # If t is a number (and thus the same for each element in this batch),
        # expand it as a tensor. This is required for the odeint solver.
        t = t * torch.ones(len(theta_t), device=theta_t.device)
        # for VP: -1/2 beta(1 - t)(score(x, t) - x).
        # Note that this is - ~f(x(1-t), 1-t) in the SDE paper
        score = self.network(t, theta_t, *context_data)
        beta = self.beta(1 - t)
        return -1 / 2 * beta[:, None] * (score - theta_t)

    # TODO: move likelihood_weighting, alpha, beta, mu, sigma into a NoiseScheduler
    def get_likelihood_weighting(self, weighting):
        if weighting == "score-matching":

            def weighting(t):
                return self.sigma(t) ** 2

        elif weighting == "score-flow":

            def weighting(t):
                return self.beta(1 - t)

        else:
            raise NotImplementedError("invalid weighting function")
        return weighting

    def alpha(self, t):
        T = t * self.beta_min + 1 / 2 * (self.beta_max - self.beta_min) * (t ** 2)
        return torch.exp(-1 / 2 * T)

    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def mu(self, t, x_1):
        return self.alpha(1 - t)[:, None].to(x_1.device) * x_1

    def sigma(self, t):
        return torch.sqrt(1 - self.alpha(1 - t) ** 2)

    def get_t_theta_t_score(self, theta_1):
        # sample time in [0, 1 - epsilon]
        t = self.sample_t(len(theta_1))
        theta_0 = self.sample_theta_0(len(theta_1))
        mu, sigma = self.mu(t, theta_1), self.sigma(t)
        theta_t = mu + theta_0 * sigma[:, None]
        score = theta_0 / sigma[:, None]
        return t, theta_t, score


# class ScoreBasedNoiseTransform:
#     def __init__(
#         self,
#         beta_min,
#         beta_max,
#         epsilon,
#     ):
#         self.beta_min = beta_min
#         self.beta_max = beta_max
#         self.epsilon = epsilon
#
#     def __call__(self, theta):
#         # sample time in [0, 1 - epsilon]
#         t = (1 - self.epsilon) * torch.rand(theta.shape[0], device=theta.device)
#         mu, sigma = self.mu(t, theta), self.sigma(t, theta).to(theta.device)
#         z = torch.randn_like(theta, device=theta.device)
#         x = mu + z * sigma[:, None]
#         score = z / sigma[:, None]
#         return x, score, t
#
#     def alpha(self, t):
#         T = t * self.beta_min + 1 / 2 * (self.beta_max - self.beta_min) * (t ** 2)
#         return torch.exp(-1 / 2 * T)
#
#     def beta(self, t):
#         return self.beta_min + t * (self.beta_max - self.beta_min)
#
#     def mu(self, t, x_1):
#         return self.alpha(1 - t)[:, None].to(x_1.device) * x_1
#
#     def sigma(self, t, x_1=None):
#         return torch.sqrt(1 - self.alpha(1 - t) ** 2)

# def get_mu_sigma(self, t, theta_1):
#     alpha_tau = self.alpha(1 - t)
#     mu = alpha_tau[:, None] * theta_1
#     sigma = torch.sqrt(1 - alpha_tau ** 2)[:, None]
#     return mu, sigma

# t = self.sample_t(len(theta))
# theta_0 = self.sample_theta_0(len(theta))
# theta_1 = theta
# mu, sigma = self.get_mu_sigma(t, theta_1)
# theta_t = mu + theta_0 * sigma
# score = theta_0 / sigma
# weighting = self.likelihood_weighting(t, theta)
#
# pred_score = self.network(t, theta_t, *context)
