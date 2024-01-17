import torch
from torch import nn

from .cflow_base import ContinuousFlowsBase


class FlowMatching(ContinuousFlowsBase):
    """
    Class for continuous normalizing flows trained with flow matching.

        t               ~ U[0, 1-eps)                               noise level
        theta_0         ~ N(0, 1)                                   sampled noise
        theta_1         = theta                                     pure sample
        theta_t         = c1(t) * theta_1 + c0(t) * theta_0         noisy sample

        eps             = 0
        c0              = (1 - (1 - sigma_min) * t)
        c1              = t

        v_target        = theta_1 - (1 - sigma_min) * theta_0
        loss            = || v_target - network(theta_t, t) ||
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eps = 0
        self.sigma_min = self.model_kwargs["posterior_kwargs"]["sigma_min"]

    def evaluate_vectorfield(self, t, theta_t, *context_data):
        """
        Vectorfield that generates the flow, see Docstring in ContinuousFlowsBase for
        details. With flow matching, this vectorfield is learnt directly.
        """
        # If t is a number (and thus the same for each element in this batch),
        # expand it as a tensor. This is required for the odeint solver.
        t = t * torch.ones(len(theta_t), device=theta_t.device)
        return self.network(t, theta_t, *context_data)

    def loss(self, theta, *context_data):
        """
        Calculates loss as the the mean squared error between the predicted vectorfield
        and the vector field for transporting the parameter data to samples from the
        prior.

        Parameters
        ----------
        theta: torch.tensor
            parameters (e.g., binary-black hole parameters)
        *context_data: list[torch.Tensor]
            context data (e.g., gravitational-wave data)

        Returns
        -------
        torch.tensor
            loss tensor
        """
        # Shall we allow for multiple time evaluations for every data, context pair (to improve efficiency)?
        mse = nn.MSELoss()

        t = self.sample_t(len(theta))
        theta_0 = self.sample_theta_0(len(theta))
        theta_1 = theta
        theta_t = ot_conditional_flow(theta_0, theta_1, t, self.sigma_min)
        true_vf = theta - (1 - self.sigma_min) * theta_0

        predicted_vf = self.network(t, theta_t, *context_data)
        loss = mse(predicted_vf, true_vf)
        return loss


def ot_conditional_flow(x_0, x_1, t, sigma_min):
    return (1 - (1 - sigma_min) * t)[:, None] * x_0 + t[:, None] * x_1
