import inspect
import textwrap

import torch
from torch import nn

from .cflow_base import ContinuousFlowPosteriorModel


class FlowMatchingPosteriorModel(ContinuousFlowPosteriorModel):
    __doc__ = (
        inspect.getdoc(ContinuousFlowPosteriorModel)
        + "\n\n"
        + textwrap.dedent(
            """\
        For flow matching, the vector field represents the velocity vector field 
        for a particle trajectory. Training proceeds as follows:

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
        )
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eps = 0
        self.sigma_min = self.model_kwargs["posterior_kwargs"]["sigma_min"]

    def evaluate_vector_field(self, t, theta_t, *context_data):
        """
        Evaluate the vector field v(t, theta_t, context_data) that generates the flow
        via the ODE

            d/dt f(theta_t, t, context) = v(f(theta_t, t, context), t, context).

        For flow matching, the vector field is regressed directly during training.

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
        return self.network(t, theta_t, *context_data)

    def loss(self, theta, *context):
        """
        Calculates loss as the mean squared error between the predicted vector field and
        the vector field for transporting the parameter data to samples from the prior.

        Parameters
        ----------
        theta: torch.Tensor
            Parameter values at which to evaluate the density. Should have a batch
            dimension (even if size B = 1).
        context: torch.Tensor
            Context information (typically observed data). Must have the same leading
            (batch) dimension as theta.

        Returns
        -------
        loss: torch.Tensor
            Mean loss across the batch (a scalar).
        """
        # Shall we allow for multiple time evaluations for every data, context pair (to improve efficiency)?
        mse = nn.MSELoss()

        t = self.sample_t(len(theta))
        theta_0 = self.sample_theta_0(len(theta))
        theta_1 = theta
        theta_t = ot_conditional_flow(theta_0, theta_1, t, self.sigma_min)
        true_vf = theta - (1 - self.sigma_min) * theta_0

        predicted_vf = self.network(t, theta_t, *context)
        loss = mse(predicted_vf, true_vf)
        return loss


def ot_conditional_flow(x_0, x_1, t, sigma_min):
    return (1 - (1 - sigma_min) * t)[:, None] * x_0 + t[:, None] * x_1
