import pickle
import copy

import numpy as np
import torch, torchvision
from typing import Dict, Union

import dingo
from dingo.pipe.default_settings import DENSITY_RECOVERY_SETTINGS
from dingo.core.models.posterior_model import PosteriorModel
from dingo.core.nn.enets import DenseResidualNet
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

def mean_square_error(x, y):
    return (x - y).abs()**2

# TODO: probably needs its own model class
# use PosteriorModel to use utilities
class SNREstimator(PosteriorModel):
    def __init__(
        self,
        model_filename: str = None,
        metadata: dict = None,
        initial_weights: dict = None,
        device: str = "cuda",
        load_training_info: bool = True,
        event_model: PosteriorModel = None,
    ):

        if(event_model is not None):
            self.add_event_model(event_model)

        super().__init__(model_filename, metadata, initial_weights, device, load_training_info)

    def initialize_model(self):
        """
        Initialize a model for the posterior by calling the
        self.model_builder with self.model_kwargs.

        """
        model_builder = get_model_callable(self.model_kwargs["type"])
        model_kwargs = {k: v for k, v in self.model_kwargs.items() if k != "type"}
        if self.initial_weights is not None:
            model_kwargs["initial_weights"] = self.initial_weights
        self.model = model_builder(**model_kwargs)

    def add_event_model(self, event_model):
        event_model.set_embedding_only()
        event_model.model.eval()
        self.event_model = event_model

    def sample(self):
        raise NotImplementedError

    def __call__(self, emb):
        mean_snr, std_snr = self.get_standardization_snr()

        snr_stand = self.model(emb)
        return snr_stand * std_snr + mean_snr


    def get_standardization_snr(self):
        mean_snr = self.metadata['train_settings']['data']['standardization']['mean']['snr']
        std_snr = self.metadata['train_settings']['data']['standardization']['std']['snr']

        return mean_snr, std_snr

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
            test_loss = test_epoch_snr_estimator(self, test_loader, self.event_model)
            print(f"test loss: {test_loss:.3f}")
        else:
            while not runtime_limits.limits_exceeded(self.epoch):
                self.epoch += 1

                # Training
                lr = utils.get_lr(self.optimizer)
                with threadpool_limits(limits=1, user_api="blas"):
                    print(f"\nStart training epoch {self.epoch} with lr {lr}")
                    time_start = time.time()
                    train_loss = train_epoch_snr_estimator(self, train_loader, self.event_model)
                    train_time = time.time() - time_start

                    print(
                        "Done. This took {:2.0f}:{:2.0f} min.".format(
                            *divmod(train_time, 60)
                        )
                    )

                    # Testing
                    print(f"Start testing epoch {self.epoch}")
                    time_start = time.time()
                    test_loss = test_epoch_snr_estimator(self, test_loader, self.event_model)
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

def train_epoch_snr_estimator(pm, dataloader, event_model, error_func=None):

    if error_func == None:
        error_func = mean_square_error

    pm.model.train()
    event_model.model.eval()

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
        
        snr = data[0]
        context = event_model.model(*data[1:])
        # compute loss
        loss = error_func(snr, pm.model(context)).mean()
        # backward pass and optimizer step
        loss.backward()
        pm.optimizer.step()
        # update loss for history and logging
        loss_info.update(loss.detach().item(), len(data[0]))
        loss_info.print_info(batch_idx)

    return loss_info.get_avg()


def test_epoch_snr_estimator(pm, dataloader, event_model, error_func=None):

    if error_func == None:
        error_func = mean_square_error

    with torch.no_grad():
        pm.model.eval()
        event_model.model.eval()

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

            snr = data[0]
            context = event_model.model(*data[1:])
            # compute loss
            loss = error_func(snr, pm.model(context)).mean()
            # update loss for history and logging
            loss_info.update(loss.item(), len(data[0]))
            loss_info.print_info(batch_idx)

        return loss_info.get_avg()

def get_model_callable(type):
    if type == 'nn-dense-res':
        return build_dense_res_net

def build_dense_res_net(**kwargs):
    return DenseResidualNet(**kwargs)