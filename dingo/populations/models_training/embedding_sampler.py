import pickle
import copy

import numpy as np
import torch, torchvision
from typing import Dict, Union

import dingo
from dingo.pipe.default_settings import DENSITY_RECOVERY_SETTINGS
from dingo.core.models.posterior_model import PosteriorModel
from dingo.gw.inference.gw_samplers import GWSampler, GWSamplerGNPE
from dingo.gw.injection import Injection
from dingo.gw.noise.asd_dataset import ASDDataset
from dingo.gw.inference.inference_pipeline import prepare_log_prob
from dingo.gw.dataset import WaveformDataset
from dingo.gw.gwutils import (
    get_standardization_dict,
    get_extrinsic_prior_dict,
)
import dingo.core.utils as utils

from dingo.gw.training.train_builders import (
    build_dataset,
    set_train_transforms,
)

from dingo.gw.transforms import (
    AddWhiteNoiseComplex,
    UnpackDict,
    SampleExtrinsicParameters,
    SelectStandardizeRepackageParameters
)

import torch
import time
from threadpoolctl import threadpool_limits

class EmbeddingSampler(PosteriorModel):
    def __init__(
        self,
        model_filename: str = None,
        metadata: dict = None,
        initial_weights: dict = None,
        device: str = "cuda",
        load_training_info: bool = True,
        pm_single_event = None
    ):
        if(pm_single_event is not None):
            self.add_pm_single_event(pm_single_event)

        super().__init__(model_filename, metadata, initial_weights, device, load_training_info)

    def get_pm_single_event(self):
        main_model_path = self.metadata['train_settings']['data']['posterior_model']
        pm_single_event = PosteriorModel(main_model_path, device=str(self.device))

        return pm_single_event

    def add_pm_single_event(self):

        pm_single_event = self.get_pm_single_event()

        pm_single_event.set_embedding_only()
        pm_single_event.model.eval()

        self.pm_single_event = pm_single_event
        self.initialize_transform_pre()

    def initialize_transform_pre(self):
        self.transform1 = SelectStandardizeRepackageParameters(
            {"inference_parameters": self.pm_single_event.metadata['train_settings']['data']['inference_parameters']},
            self.pm_single_event.metadata["train_settings"]["data"]["standardization"],
            inverse=False,
            as_type="dict",
        )
        self.transform2 = UnpackDict(selected_keys=["inference_parameters"])

        self.transform = torchvision.transforms.Compose([self.transform1, self.transform2])


    def sample(self, params, batch_size, device=None):

        x = self.transform(dict(parameters=params))
        x = torch.tensor(np.array(x)).squeeze()

        if(batch_size != None):
            x = x.expand(batch_size, *x.shape)

        if(device != None):
            x = x.to(device)
        
        return super().sample(x, batch_size=batch_size)

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
            test_loss = test_epoch_embedding(self, test_loader, self.pm_single_event)
            print(f"test loss: {test_loss:.3f}")
        else:
            while not runtime_limits.limits_exceeded(self.epoch):
                self.epoch += 1

                # Training
                lr = utils.get_lr(self.optimizer)
                with threadpool_limits(limits=1, user_api="blas"):
                    print(f"\nStart training epoch {self.epoch} with lr {lr}")
                    time_start = time.time()
                    train_loss = train_epoch_embedding(self, train_loader, self.pm_single_event)
                    train_time = time.time() - time_start

                    print(
                        "Done. This took {:2.0f}:{:2.0f} min.".format(
                            *divmod(train_time, 60)
                        )
                    )

                    # Testing
                    print(f"Start testing epoch {self.epoch}")
                    time_start = time.time()
                    test_loss = test_epoch_embedding(self, test_loader, self.pm_single_event)
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

def train_epoch_embedding(pm, dataloader, pm_single_event):
    pm.model.train()
    pm_single_event.model.eval()
    loss_info = dingo.core.utils.trainutils.LossInfo(
        pm.epoch,
        len(dataloader.dataset),
        dataloader.batch_size,
        mode="Train",
        print_freq=1,
    )

    for batch_idx, data in enumerate(dataloader):
        loss_info.update_timer()
        pm.optimizer.zero_grad()
        # data to device
        data = [d.to(pm.device, non_blocking=True) for d in data]
        
        context = pm_single_event.model(*data[1:])
        # compute loss, remember, we want to determine the embedding, so context and 
        # params switch place
        loss = -pm.model(context, data[0]).mean()
        # backward pass and optimizer step
        loss.backward()
        pm.optimizer.step()
        # update loss for history and logging
        loss_info.update(loss.detach().item(), len(data[0]))
        loss_info.print_info(batch_idx)

    return loss_info.get_avg()


def test_epoch_embedding(pm, dataloader, pm_single_event):
    with torch.no_grad():
        pm.model.eval()
        pm_single_event.model.eval()
        loss_info = dingo.core.utils.trainutils.LossInfo(
            pm.epoch,
            len(dataloader.dataset),
            dataloader.batch_size,
            mode="Test",
            print_freq=1,
        )

        for batch_idx, data in enumerate(dataloader):
            loss_info.update_timer()
            # data to device
            data = [d.to(pm.device, non_blocking=True) for d in data]

            context = pm_single_event.model(*data[1:])
            # compute loss
            loss = -pm.model(context, data[0]).mean()
            # update loss for history and logging
            loss_info.update(loss.item(), len(data[0]))
            loss_info.print_info(batch_idx)

        return loss_info.get_avg()

class WaveformEmbeddingDataset(WaveformDataset):
    def __init__(self, pm_single_event, params_for_embedding, transform_embedding=None):
        """
        Initialize the WaveformEmbeddingDataset with an event model, parameters for embedding,
        and optional transformations.
        
        Parameters
        ----------
        pm_single_event : Model
            The model used for generating embeddings (e.g., samplers['NPE']).
        params_for_embedding : list
            List of parameter names for embedding purposes.
        transform : callable, optional
            Transformations to apply to the data (default is None).
        """
        pm_single_event.set_embedding_only()
        pm_single_event.model.eval()
        
        self.pm_single_event = pm_single_event
        self.params_for_embedding = params_for_embedding
        self.transform_embedding = transform_embedding

        # not sure that is the best solution? 
        self.device = pm_single_event.device

        # Load metadata from the event model
        train_settings = pm_single_event.metadata['train_settings']

        # for debugging
        train_settings['data']['waveform_dataset_path'] = '/mnt/lustre2/gravitational_waves/kleyde/dingo_population/waveform_datasets/waveform_test.hdf5'

        # Get the dataset path and initialize the parent class (WaveformDataset)
        domain_update = train_settings['data'].get("domain_update", None)
        
        # Initialize the parent class with the dataset settings
        super().__init__(
            file_name=train_settings['data']["waveform_dataset_path"],
            precision="single",
            domain_update=domain_update,
            svd_size_update=train_settings['data'].get("svd_size_update"),
        )

        # Apply any training-specific transforms
        set_train_transforms(
            self,
            train_settings["data"],
            train_settings["training"]["stage_0"]["asd_dataset_path"],
        )
        
        # Iterate over the transforms and adjust them as needed
        for t in self.transform.transforms:
            if isinstance(t, AddWhiteNoiseComplex):
                t.store_snr = True  # Store the signal-to-noise ratio for selection effects
            if isinstance(t, UnpackDict):
                t.selected_keys[0] = "parameters"  # Save all the parameters

    def __getitem__(self, idx) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """
        Return a dictionary containing parameters, waveforms, and embeddings
        for the sample with index `idx`.
        
        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        data : dict
            Dictionary containing 'embeddings' and 'params'.
        """
        # Retrieve the sample from the waveform dataset
        parameters, waveform = super().__getitem__(idx)

        # Filter parameters for embedding
        parameters = {k: v for k, v in parameters.items() if k in self.params_for_embedding}

        # Convert waveform to tensor and add a batch dimension
        waveform = torch.tensor(waveform).to(self.device)

        # TODO have to add an addditional dimension here for this to work
        waveform = waveform.unsqueeze(0)

        # Compute the embedding using the event model
        with torch.no_grad():
            embeddings = self.pm_single_event.model(waveform).squeeze().numpy()

        # Create the data dictionary
        data = dict(
            embeddings=embeddings,
            parameters=parameters,
        )

        # Apply the optional transformation
        if self.transform_embedding is not None:
            data = self.transform_embedding(data)

        return data

def set_embedding_transforms(wfed, data_settings):

    transforms = []

    try:
        standardization_dict = data_settings["standardization"]
        print("Parameters in embeddings: using previously-calculated parameter standardizations.")
    except KeyError:
        extrinsic_prior_dict = get_extrinsic_prior_dict(data_settings["extrinsic_prior"])
        
        print("Parameters in embeddings: calculating new parameter standardizations.")
        standardization_dict = get_standardization_dict(
            extrinsic_prior_dict,
            wfed,
            data_settings["inference_parameters"] + data_settings["context_parameters"],
            torchvision.transforms.Compose(transforms),
        )
        data_settings["standardization"] = standardization_dict

    dict_params = {
            k: data_settings[k]
            for k in ["inference_parameters", "context_parameters"]
        }
    transforms.append(
        SelectStandardizeRepackageParameters(
            dict_params,
            standardization_dict,
        )
    )

    if data_settings["context_parameters"]:
        selected_keys = ["embeddings", "inference_parameters", "context_parameters"]
    else:
        selected_keys = ["embeddings", "inference_parameters"]

    transforms.append(UnpackDict(selected_keys=selected_keys))
    
    wfed.transform_embedding = torchvision.transforms.Compose(transforms)
