import copy

import torch

from dingo.core.nn.enets_pretraining import create_embedding_with_resnet
from .base_model import BasePosteriorModel


class PretrainingModel(BasePosteriorModel):
    """
    Pretraining model for transformer embedding network.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if (
            self.metadata["train_settings"]["model"]["posterior_kwargs"][
                "loss_function"
            ]
            == "mse"
        ):
            self.loss_fn = torch.nn.MSELoss()
        else:
            ValueError(
                f"Loss objective {self.metadata['pretraining']['loss_objective']} not implemented. Available "
                f"options are 'mse'."
            )

    def initialize_network(self):
        model_kwargs = {
            k: v
            for k, v in self.model_kwargs.items()
            if k != "posterior_model_type" and k != "embedding_type"
        }
        if self.initial_weights is not None:
            model_kwargs["initial_weights"] = self.initial_weights
        if self.embedding_net_builder is not None:
            model_kwargs["embedding_net_builder"] = self.embedding_net_builder
        self.network = create_embedding_with_resnet(**model_kwargs)

    def loss(self, data, *context_data):
        output, logging_info = self.network(context_data)
        return self.loss_fn(output, data), logging_info


def build_pretraining_model_kwargs(train_settings: dict):
    model_kwargs = {
        "posterior_model_type": train_settings["pretraining"]["model"][
            "pretraining_model_type"
        ],
        "posterior_kwargs": train_settings["pretraining"]["model"][
            "pretraining_kwargs"
        ],
        "embedding_type": train_settings["model"]["embedding_type"],
        "embedding_kwargs": train_settings["model"]["embedding_kwargs"],
    }
    pretrain_settings = {
        "data": train_settings["data"],
        "model": model_kwargs,
        "training": train_settings["pretraining"]["training"],
    }
    return pretrain_settings


def check_pretraining_model_compatibility(train_settings: dict, pm_settings: dict):
    pm_train_settings = copy.deepcopy(pm_settings["train_settings"]["model"])
    pm_train_settings["embedding_kwargs"].pop("input_dims")
    if (
        pm_train_settings["posterior_kwargs"]["type"]
        != train_settings["pretraining"]["model"]["pretraining_kwargs"]["type"]
    ):
        raise ValueError(
            f"Model type of pretrained model {pm_train_settings['posterior_kwargs']['type']} is different"
            f"from model type {train_settings['pretraining']['model']['pretraining_kwargs']['type']} in "
            f"train settings file."
        )
    if (
        pm_train_settings["embedding_kwargs"]
        != train_settings["model"]["embedding_kwargs"]
    ):
        raise ValueError(
            f"Embedding net kwargs of pretrained model {pm_train_settings['embedding_kwargs']} is ",
            f"different from kwargs in train settings file "
            f"{train_settings['pretraining']['model']['embedding_net_kwargs']}.",
        )
