import pickle
import copy
import os

import numpy as np
import torch, torchvision
from typing import Dict, Union

import dingo
from dingo.pipe.default_settings import DENSITY_RECOVERY_SETTINGS
from dingo.core.models.posterior_model import PosteriorModel, get_model_callable
from dingo.core.nn.enets import DenseResidualNet
from dingo.gw.gwutils import (
    get_standardization_dict,
    get_extrinsic_prior_dict,
)
import dingo.core.utils as utils

from dingo.populations.models_training.snr_estimator import (
    train_epoch_snr_estimator,
    test_epoch_snr_estimator,
)

from dingo.populations.models_training.embedding_emulator import (
    train_epoch_embedding,
    test_epoch_embedding,
    train_epoch_embedding_det,
    test_epoch_embedding_det,
)

from dingo.populations.models_training.pdet_model import (
    train_epoch_pdet_model,
    test_epoch_pdet_model,
)

from dingo.populations.models_training.population_model import (
    train_epoch_population_model,
    test_epoch_population_model,
)

from dingo.gw.transforms import (
    UnpackDict,
    SelectStandardizeRepackageParameters
)

import torch
import time
from threadpoolctl import threadpool_limits

class GenericModel(PosteriorModel):
    def __init__(
        self,
        model_filename: str = None,
        metadata: dict = None,
        initial_weights: dict = None,
        device: str = "cuda",
        load_training_info: bool = True,
        type_pm: str = None,
    ):

        self.train_function = construct_train_function(type_pm)
        self.test_function = construct_test_function(type_pm)

        self.get_model_callable = construct_get_model_callable(type_pm)

        super().__init__(model_filename, metadata, initial_weights, device, load_training_info)

    # need this, since the get_model_callable needs to be overwritten in some cases
    def initialize_model(self):
        """
        Initialize a model for the posterior by calling the
        self.model_builder with self.model_kwargs.

        """
        model_builder = self.get_model_callable(self.model_kwargs["type"])
        model_kwargs = {k: v for k, v in self.model_kwargs.items() if k != "type"}
        if self.initial_weights is not None:
            model_kwargs["initial_weights"] = self.initial_weights
        self.model = model_builder(**model_kwargs)

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        train_dir: str,
        runtime_limits: object = None,
        checkpoint_epochs: int = None,
        use_wandb=False,
        test_only=False,
    ):
        if test_only:
            test_loss = test_epoch_snr_estimator(self, test_loader, self.pm_single_event)
            print(f"test loss: {test_loss:.3f}")
        else:
            while not runtime_limits.limits_exceeded(self.epoch):
                self.epoch += 1

                # Training
                lr = utils.get_lr(self.optimizer)
                with threadpool_limits(limits=1, user_api="blas"):
                    print(f"\nStart training epoch {self.epoch} with lr {lr}")
                    time_start = time.time()
                    train_loss = self.train_function(self, train_loader)
                    train_time = time.time() - time_start

                    print(
                        "Done. This took {:2.0f}:{:2.0f} min.".format(
                            *divmod(train_time, 60)
                        )
                    )

                    # Testing
                    print(f"Start testing epoch {self.epoch}")
                    time_start = time.time()
                    test_loss = self.test_function(self, test_loader)
                    test_time = time.time() - time_start

                    print(
                        "Done. This took {:2.0f}:{:2.0f} min.".format(
                            *divmod(test_time, 60)
                        )
                    )

                # scheduler step for learning rate
                utils.perform_scheduler_step(self.scheduler, test_loss)

                # write history and save model
                utils.write_history(train_dir, self.epoch, train_loss, test_loss, lr)
                utils.save_model(self, train_dir, checkpoint_epochs=checkpoint_epochs)
                
                if use_wandb:
                    try:
                        import wandb
                        wandb.define_metric("epoch")
                        wandb.define_metric("*", step_metric="epoch")
                        wandb.log(
                            {
                                "epoch": self.epoch,
                                "learning_rate": lr[0],
                                "train_loss": train_loss,
                                "test_loss": test_loss,
                                "train_time": train_time,
                                "test_time": test_time,
                            }
                        )
                    except ImportError:
                        print("wandb not installed. Skipping logging to wandb.")

                print(f"Finished training epoch {self.epoch}.\n")

def construct_train_function(type_pm):
    if type_pm == 'snr_estimator':
        return train_epoch_snr_estimator
    elif type_pm == 'embedding_emulator':
        return train_epoch_embedding
    elif type_pm == 'embedding_emulator_det':
        return train_epoch_embedding_det
    elif type_pm == 'population_model':
        return train_epoch_population_model
    elif type_pm == 'pdet_model':
        return train_epoch_pdet_model
    else:
        raise 'Model not known. '

def construct_test_function(type_pm):
    if type_pm == 'snr_estimator':
        return test_epoch_snr_estimator
    elif type_pm == 'embedding_emulator':
        return test_epoch_embedding
    elif type_pm == 'embedding_emulator_det':
        return test_epoch_embedding_det
    elif type_pm == 'population_model':
        return test_epoch_population_model
    elif type_pm == 'pdet_model':
        return test_epoch_pdet_model
    else:
        raise 'Model not known. '

def construct_get_model_callable(type_model):

    if(type_model not in ['snr_estimator', 'pdet_model']):
        return get_model_callable
    else:
        def gmc(type_from_model_kwargs):
            if(type_from_model_kwargs=='nn-dense-res'):
                return build_dense_res_net
            else:
                raise NotImplementedError
    
        return gmc

def build_dense_res_net(**kwargs):
    return DenseResidualNet(**kwargs)


class SNREstimator(GenericModel):
    def __init__(
        self,
        model_filename: str = None,
        metadata: dict = None,
        initial_weights: dict = None,
        device: str = "cuda",
        load_training_info: bool = True,
        pm_single_event: PosteriorModel = None,
    ):  
        type_pm = 'snr_estimator'
        self.mf_snr_threshold = None

        # only add single event model if provided
        if(pm_single_event is not None):
            self.add_pm_single_event(pm_single_event)

        super().__init__(model_filename, metadata, initial_weights, device, load_training_info, type_pm)

    def set_mf_snr_threshold(self, mf_snr_threshold):
        self.mf_snr_threshold = mf_snr_threshold

    def get_pm_single_event(self):
        main_model_path = self.metadata['train_settings']['data']['posterior_model']
        pm_single_event = PosteriorModel(main_model_path, device=str(self.device))

        return pm_single_event

    def add_pm_single_event(self, pm_single_event=None):

        if pm_single_event is None:
            pm_single_event = self.get_pm_single_event()

        pm_single_event.set_embedding_only()
        pm_single_event.model.eval()

        self.pm_single_event = pm_single_event

    def sample(self):
        raise NotImplementedError

    def __call__(self, emb):
        # TODO use transformations
        mean_snr, std_snr = self.get_standardization_snr()

        snr_stand = self.model(emb)
        return snr_stand * std_snr + mean_snr


    def get_standardization_snr(self):
        mean_snr = self.metadata['train_settings']['data']['standardization']['mean']['snr']
        std_snr = self.metadata['train_settings']['data']['standardization']['std']['snr']

        return mean_snr, std_snr

    def apply_selection_to_embeddings(self, emb):

        snr = self(emb).squeeze()
        idx = snr > self.mf_snr_threshold

        # make sure the idx shape is always 2d
        if len(idx.shape) == 1:
            idx = idx.unsqueeze(0)

        return idx
    
def fix_me_get_snr_threshold_from_filename(model_filename):
    device = 'cpu'
    ext = os.path.splitext(model_filename)[-1]
    if ext == ".pt":
        metadata = torch.load(model_filename, map_location=device)["metadata"]
    else:
        raise ValueError("Models should be in .pt format.")
    
    return metadata['train_settings']['data'].get('snr_threshold', None)

class EmbeddingEmulator(GenericModel):
    def __init__(
        self,
        model_filename: str = None,
        metadata: dict = None,
        initial_weights: dict = None,
        device: str = "cuda",
        load_training_info: bool = True,
        pm_single_event = None,
    ):  
        if metadata is not None:
            self.snr_threshold = metadata['train_settings']['data'].get('snr_threshold', None)

        # need to get the snr threshold from metadata for model type
        if model_filename is not None:
            self.snr_threshold = fix_me_get_snr_threshold_from_filename(model_filename)

        if self.snr_threshold is None:
            type_pm = 'embedding_emulator'
        else:
            print(f'Applying SNR threshold {self.snr_threshold}')
            type_pm = 'embedding_emulator_det'

        # only add single event model if provided
        if(pm_single_event is not None):
            self.add_pm_single_event(pm_single_event)

        super().__init__(model_filename, metadata, initial_weights, device, load_training_info, type_pm)
        self.initialize_transform_pre()

    def get_pm_single_event(self):
        main_model_path = self.metadata['train_settings']['data']['posterior_model']
        pm_single_event = PosteriorModel(main_model_path, device=str(self.device))

        return pm_single_event

    def add_pm_single_event(self, pm_single_event=None):

        if pm_single_event is None:
            pm_single_event = self.get_pm_single_event()

        pm_single_event.set_embedding_only()
        pm_single_event.model.eval()

        self.pm_single_event = pm_single_event
        self.initialize_transform_pre()

    def initialize_transform_pre(self):
        params_for_embedding = copy.deepcopy(self.metadata['train_settings']['data']['params_for_embedding'])

        # remove matched filter snr if needed (otherwise this will mess with )
        # the standardization later on
        if 'matched_filter_snr' in params_for_embedding: params_for_embedding.remove('matched_filter_snr')

        self.transform1 = SelectStandardizeRepackageParameters(
            {
                "inference_parameters": params_for_embedding 
            },
            self.metadata['settings_pm_single_event']["train_settings"]["data"]["standardization"],
            inverse=False,
            as_type="dict",
        )
        self.transform2 = UnpackDict(selected_keys=["inference_parameters"])

        self.transform_params_to_array = torchvision.transforms.Compose([self.transform1, self.transform2])

    def sample_from_params(self, params, device=None, num_samples=None):
        
        x = self.transform_params_to_array(dict(parameters=params))[0]
        x = torch.tensor(np.array(x)).squeeze()

        if(num_samples is not None):
            x = x.expand(num_samples, *x.shape)

        return self.sample(x, device=device)

    def sample(self, x=None, batch_size=None, device=None):
        
        if len(x.shape) == 3:
            flattened = True
            batch_shape_pre = x.shape[:-1]
            # flatten along batch dimension
            x = x.reshape(-1, x.shape[-1])
        else:
            flattened = False

        if(device is not None):
            x = x.to(device)
        
        y = super().sample(x, batch_size=batch_size)

        if flattened:
            # unflatten
            y = y.reshape((*batch_shape_pre, -1))

        return y
    
class PdetModel(GenericModel):
    def __init__(
        self,
        model_filename: str = None,
        metadata: dict = None,
        initial_weights: dict = None,
        device: str = "cuda",
        load_training_info: bool = True,
    ):  
        if metadata is not None:
            self.snr_threshold = metadata['train_settings']['data'].get('snr_threshold', None)

        # need to get the snr threshold from metadata for model type
        if model_filename is not None:
            self.snr_threshold = fix_me_get_snr_threshold_from_filename(model_filename)

        type_pm = 'pdet_model'

        super().__init__(model_filename, metadata, initial_weights, device, load_training_info, type_pm)
        self.initialize_transform_pre()

    def sample(self):
        raise NotImplementedError

    def __call__(self, params):
        x = self.model(params)
        x_norm = - torch.nn.functional.softplus(-x)

        return x_norm
    
    def initialize_transform_pre(self):
        params_for_embedding = copy.deepcopy(self.metadata['train_settings']['data']['parameters'])

        self.transform1 = SelectStandardizeRepackageParameters(
            {
                "inference_parameters": params_for_embedding 
            },
            self.metadata['settings_pm_single_event']["train_settings"]["data"]["standardization"],
            inverse=False,
            as_type="dict",
        )
        self.transform2 = UnpackDict(selected_keys=["inference_parameters"])

        self.transform_params_to_array = torchvision.transforms.Compose([self.transform1, self.transform2])

    def get_pdet_from_params(self, params):
        x = self.transform_params_to_array(params)
        log_pdet = self(x)

        return torch.exp(log_pdet)

class PopulationModel(GenericModel):
    def __init__(
        self,
        model_filename: str = None,
        metadata: dict = None,
        initial_weights: dict = None,
        device: str = "cuda",
        load_training_info: bool = True,
        embedding_emulator: EmbeddingEmulator = None,
        snr_estimator: SNREstimator = None
    ):
        type_pm = 'population_model'
        self.embedding_emulator = embedding_emulator
        self.snr_estimator = snr_estimator

        super().__init__(model_filename, metadata, initial_weights, device, load_training_info, type_pm)

    # def sample(self, params, batch_size, device=None):

    #     x = self.transform(dict(parameters=params))
    #     x = torch.tensor(np.array(x)).squeeze()

    #     if(batch_size != None):
    #         x = x.expand(batch_size, *x.shape)

    #     if(device != None):
    #         x = x.to(device)
        
    #     return super().sample(x, batch_size=batch_size)