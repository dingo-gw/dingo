"""
The per-event context protocol of the factorized sampler.

Every step of a chain reads the event through a shared context: the data in the
representation the networks were trained on, the per-event metadata, and the
likelihood. This module defines the protocol; the gravitational-wave
implementation is `dingo.gw.inference.context.GWSamplerContext`.
"""

from __future__ import annotations

from typing import Optional, Protocol, Union

import torch


class SamplerContext(Protocol):
    """
    Protocol for the per-event state shared by all steps of a chain.

    A context holds the event data and metadata, and everything derived from them
    that a step may need: the data in the representation the networks were trained
    on (`prepared_data`), and the likelihood. Steps never receive the data directly;
    they read it through the context. Concrete implementations are domain-specific;
    see `dingo.gw.inference.context.GWSamplerContext`.

    Attributes
    ----------
    event_metadata : dict or None
        Per-event metadata, such as the event time and analysis settings.
    device : torch.device or str
        The device the chain runs on. Steps that create new tensors, rather than
        transforming existing ones, create them on this device so that they can be
        combined with the outputs of networks running on a GPU.
    """

    event_metadata: Optional[dict]
    device: Union[torch.device, str]

    def prepared_data(self, conditioning=None) -> torch.Tensor:
        """The event data in the representation the networks condition on.

        Parameters
        ----------
        conditioning : dict[str, torch.Tensor], optional
            Chain columns available to a conditioned factor. Without it, the single
            shared representation is returned, computed once and cached. With it,
            the result has one data row per conditioning row. Only the columns the
            data preparation depends on (for example a heterodyning proxy) affect
            the result; the other columns condition the network alone.

        Returns
        -------
        torch.Tensor
        """
        ...

    def likelihood(self):
        """The likelihood of the event data, for likelihood-based factors (such as
        the synthetic phase) and for importance sampling."""
        ...
