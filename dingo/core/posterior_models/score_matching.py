import inspect
import textwrap

import torch

from .cflow_base import ContinuousFlowPosteriorModel


class ScoreDiffusionPosteriorModel(ContinuousFlowPosteriorModel):
    __doc__ = (
        inspect.getdoc(ContinuousFlowPosteriorModel)
        + "\n\n"
        + textwrap.dedent(
            """\
                Training with score matching:
            
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
                    
                To specify the score matching model, "posterior_kwargs" should 
                additionally specify the noise properties used for the diffusion (
                beta_min, beta_max, epsilon).
                """
        )
    )

    def __init__(self, **kwargs):
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

        weighting = self.likelihood_weighting(t)
        losses = torch.square(pred_score - score)
        losses = 1 / 2 * torch.sum(losses, dim=1) * weighting
        loss = torch.mean(losses)
        return loss

    def evaluate_vector_field(self, t, theta_t, *context_data):
        """
        Evaluate the vector field v(t, theta_t, context_data) that generates the flow
        via the ODE

            d/dt f(theta_t, t, context) = v(f(theta_t, t, context), t, context).

        For score matching, the vector field (or drift function) is computed
        from the predicted score.

        Parameters
        ----------
        t: float
            time (noise level)
        theta_t: torch.Tensor
            noisy parameters, perturbed with noise level t
        *context_data: list[torch.tensor]
            list with context data (GW data)
        """
        # If t is a number (and thus the same for each element in this batch),
        # expand it as a tensor. This is required for the odeint solver.
        t = t * torch.ones(len(theta_t), device=theta_t.device)
        # for VP: -1/2 beta(1 - t)(score(x, t) - x).
        # Note that this is - ~f(x(1-t), 1-t) in the SDE paper
        score = self.network(t, theta_t, *context_data)
        beta = self.beta(1 - t)
        return -1 / 2 * beta[:, None] * (score - theta_t)

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
        T = t * self.beta_min + 1 / 2 * (self.beta_max - self.beta_min) * (t**2)
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
