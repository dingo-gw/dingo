"""
TODO: Docstring
"""

from typing import Callable
import torch


class PosteriorModel:
    """
    TODO: Docstring

    Methods
    -------

    initialize_model:
        initialize the NDE (including embedding net) as posterior model
    initialize_training:
        initialize for training, that includes storing the epoch, building
        and optimizer and a learning rate scheduler
    initialize_data_loader:
        initialize the data loader used for training
        TODO:
        this should not be required at inference time. All information
        regarding data conditioning should be stored in
        self.data_conditioning_information.

    save_model:
        save the model, including all information required to rebuild it,
        except for the builder function
    load_model:
        load and build a model from a file

    train_model:
        train the model
    inference:
        perform inference
    """

    def __init__(self,
                 model_builder: Callable,
                 model_kwargs: dict = None,
                 model_filename: str = None,
                 ):
        """
        TODO: Docstring
        """
        self.model_builder = model_builder
        self.model_kwargs = model_kwargs

        # build model
        if model_filename is not None:
            self.load_model(model_filename)
        else:
            self.initialize_model()

        # TODO: initialize training and  data loader

    def initialize_model(self):
        """
        Initialize a model for the posterior by calling the
        self.model_builder with self.model_kwargs.

        """
        self.model = self.model_builder(**self.model_kwargs)

    def initialize_training(self,
                            epoch: int = 0,
                            optimizer=None,
                            scheduler=None,
                            ):
        self.epoch = epoch
        # TODO

    def save_model(self,
                   model_filename: str,
                   ):
        """
        Save the posterior model to the disk.

        Parameters
        ----------
        model_filename: str
            filename for saving the model
        save_training_info: bool #TODO: save information for training
            specifies whether information required to proceed with training is
            saved, e.g. optimizer state dict

        """
        model_dict = {
            'model_kwargs': self.model_kwargs,
            'model_state_dict': self.model.state_dict(),
            # 'training_info': None,
            # 'data_conditioning': None,
        }
        torch.save(model_dict, model_filename)



    def load_model(self,
                   model_filename: str,
                   ):
        """
        Load a posterior model from the disk.

        Parameters
        ----------
        model_filename: str
            path to saved model
        save_training_info: bool #TODO: load information for training
            specifies whether information required to proceed with training is
            loaded, e.g. optimizer state dict
        """
        # with torch.load(model_filename) as d:
        #     self.model_kwargs = d['model_kwargs']
        #     self.initialize_model()
        #     self.model.load_state_dict(d['model_state_dict'])

        d = torch.load(model_filename)
        self.model_kwargs = d['model_kwargs']
        self.initialize_model()
        self.model.load_state_dict(d['model_state_dict'])



if __name__ == '__main__':
    from dingo.core.nn.nsf import create_nsf_with_rb_projection_embedding_net
    import os
    from os.path import join

    nsf_kwargs = {
        "input_dim": 4,
        "context_dim": 10,
        "num_flow_steps": 5,
        "base_transform_kwargs": {
            "hidden_dim": 64,
            "num_transform_blocks": 2,
            "activation": "elu",
            "dropout_probability": 0.0,
            "batch_norm": True,
            "num_bins": 8,
            "base_transform_type": "rq-coupling",
        },
    }
    embedding_net_kwargs = {
        'input_dims': (2, 3, 20),
        'n_rb': 10,
        'V_rb_list': None,
        'output_dim': 8,
        'hidden_dims': [32, 16, 8],
        'activation': 'elu',
        'dropout': 0.0,
        'batch_norm': True,
        'added_context': True,
    }

    tmp_dir = './tmp_files'
    os.makedirs(tmp_dir, exist_ok=True)
    model_filename = join(tmp_dir, 'model.pt')

    pm = PosteriorModel(
        model_builder=create_nsf_with_rb_projection_embedding_net,
        model_kwargs={'nsf_kwargs': nsf_kwargs,
                      'embedding_net_kwargs': embedding_net_kwargs},
    )

    pm.save_model(model_filename)

    pm_loaded = PosteriorModel(
        model_builder=create_nsf_with_rb_projection_embedding_net,
        model_filename=model_filename,
    )

    print('Done')
