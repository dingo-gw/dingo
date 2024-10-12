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
    SelectStandardizeRepackageParameters
)

import time
from threadpoolctl import threadpool_limits

def get_detected_embeddings(x, embedding_emulator, selection_model, number_events, maximum_population_size):

    """ 
    Produce detected embeddings from a tensor x of the shape 
    (number of events, number of parameters).

    Parameters
    ----------

    x : torch.Tensor
        Tensor of shape (number of events, number of parameters).
    embedding_emulator : EmbeddingEmulator
        Embedding emulator. 
    selection_model : SNREstimator
        Selection model.
    number_events : int
        Number of events to detect.

    Returns
    -------
    emb_det : torch.Tensor
        Detected embeddings.
    
    """
    
    # sample from embedding emulator
    emb = embedding_emulator.sample(x=x)

    # check whether embeddings are detected
    idx = selection_model.apply_selection_to_embeddings(emb)

    # make sure there are enough detected embeddings
    not_enough_events = check_below_required_event_numbers(idx, number_events)

    if(torch.any(not_enough_events)):
        print("We have generated event numbers: ", idx.sum(axis=1))
        print("We need the following event numbers: ", number_events)
        # raise ValueError("Not enough detected embeddings.")

    # only return detected embeddings
    emb_det = torch.zeros((emb.shape[0], maximum_population_size, emb.shape[2]))

    for i,v in enumerate(emb):
        # collate zeros for missing events, but should never happen, 
        # because we check above if we have enough events
        vv = v[idx[i]]
        emb_det[i,:number_events[i]] = vv[:number_events[i]]

    return emb_det

def train_epoch_population_model(pm, dataloader):

    pm_embedding_emulator = pm.embedding_emulator
    selection_model = pm.snr_estimator

    pm_embedding_emulator.model.eval()
    selection_model.model.eval()

    maximum_population_size = pm.metadata["train_settings"]["data"]["maximum_population_size"]

    pm.model.train()
    
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

        hp = data[0]
        emb_det = get_detected_embeddings(data[1], pm_embedding_emulator, selection_model, data[2], maximum_population_size)
        emb_det = emb_det.to(pm.device, non_blocking=True)

        loss = -pm.model(hp, emb_det).mean()
        # backward pass and optimizer step
        loss.backward()
        pm.optimizer.step()
        # update loss for history and logging
        loss_info.update(loss.detach().item(), len(data[0]))
        loss_info.print_info(batch_idx)

    return loss_info.get_avg()


def test_epoch_population_model(pm, dataloader):

    pm_embedding_emulator = pm.pm_embedding_emulator
    selection_model = pm.selection_model

    pm_embedding_emulator.model.eval()
    selection_model.model.eval()    

    with torch.no_grad():
        pm.model.eval()

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

            hp = data[0]
            emb_det = get_detected_embeddings(data[1], pm_embedding_emulator, selection_model, data[2], maximum_population_size)
            emb_det = emb_det.to(pm.device, non_blocking=True)

            # compute loss
            loss = -pm.model(hp, emb_det).mean()
            # update loss for history and logging
            loss_info.update(loss.item(), len(data[0]))
            loss_info.print_info(batch_idx)

        return loss_info.get_avg()

def check_below_required_event_numbers(idx, number_min):

    """
    Check whether there are enough detected events. Takes as input a 
    two-dimension tensor idx of shape (batch size, number of events)
    
    Parameters
    ----------
    idx : torch.Tensor
        Two-dimension tensor of shape (batch size, number of events).
    number_min : int
        Minimum number of detected events.
    
    Returns
    -------
    bool
        True if there are enough detected events, False otherwise.
        shape: (batch size)

    """

    num_events_det = idx.sum(axis=1)
    
    return num_events_det < number_min