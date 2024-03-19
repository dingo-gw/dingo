import copy
import torch

from dingo.core.models.base_model import BaseModel


class PretrainingModel(BaseModel):
    """
    Pretraining model for transformer embedding network.
    """

    def __init__(self,
                 model_filename: str = None,
                 metadata: dict = None,
                 initial_weights: dict = None,
                 device: str = "cuda",
                 load_training_info: bool = True,
                 ):
        pretraining_metadata = None
        if metadata is not None:
            # Initialize new pretraining model
            assert metadata['train_settings']['model']['type'] == "nsf+transformer", ("Model type should be "
                                                                                      "nsf+transformer for pretraining")
            # Construct new train_settings and metadata dict based on combination of pretraining & model arguments
            model_kwargs = {
                'type': metadata['train_settings']['pretraining']['model']['type'],
                'embedding_net_kwargs': metadata['train_settings']['model']['embedding_net_kwargs'],
                'pretraining_net_kwargs': metadata['train_settings']['pretraining']['model'],
            }

            pretrain_settings = {
                'data': metadata['train_settings']['data'],
                'model': model_kwargs,
                'training': metadata['train_settings']['pretraining']['training'],
            }
            pretraining_metadata = {
                'dataset_settings': metadata['dataset_settings'],
                'train_settings': pretrain_settings,
            }

        elif model_filename is not None:
            # Load pretraining model from file
            pretraining_metadata = None

        super().__init__(model_filename, pretraining_metadata, initial_weights, device, load_training_info)

        if self.metadata['train_settings']['training']['loss_objective'] == 'mse':
            self.loss_fn = torch.nn.MSELoss()
        else:
            ValueError(f"Loss objective {metadata['pretraining']['loss_objective']} not implemented. Available "
                       f"options are 'mse'.")

    def loss(self, data, *context_data):
        return self.loss_fn(self.model(context_data), data)


def autocomplete_model_kwargs_embedding_pretraining(train_settings: dict, data_sample):
    """
    Autocomplete the model kwargs from train_settings and data_sample from
    the dataloader:
    (*) set input dimension of embedding net to shape of data_sample[1]
    (*) set input dimension of pretraining net to output dimension of embedding network
    (*) set output dimension of pretraining net to number of posterior parameters
    """
    # set input dims from ifo_list and domain information
    train_settings["model"]["embedding_net_kwargs"]["input_dims"] = list(data_sample[1].shape)
    # set input and output dim for pretraining network
    train_settings["pretraining"]["model"]["input_dim"] = train_settings["model"]["embedding_net_kwargs"]["output_dim"]
    train_settings["pretraining"]["model"]["output_dim"] = data_sample[0].shape[0]


def check_pretraining_model_compatibility(train_settings: dict, pm: PretrainingModel):
    pm_settings = copy.deepcopy(pm.metadata["train_settings"]["model"])
    pm_settings["embedding_net_kwargs"].pop("input_dims")
    if pm_settings["type"] != train_settings["pretraining"]["model"]["type"]:
        raise ValueError(f"Model type of pretrained model {pm_settings['type']} is different from model type",
                         f"{train_settings['pretraining']['model']['type']} in train settings file.")
    if pm_settings["embedding_net_kwargs"] != train_settings["model"]["embedding_net_kwargs"]:
        raise ValueError(f"Embedding net kwargs of pretrained model ",
                         f"{pm.metadata['train_settings']['model']['embedding_net_kwargs']} is different from kwargs "
                         f"in train settings file {train_settings['pretraining']['model']['embedding_net_kwargs']}.")
