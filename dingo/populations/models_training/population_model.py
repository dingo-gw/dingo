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

def check_below_required_event_numbers(n_lim_eff, n_lim):

    """
    Check whether there are enough detected events. Takes as input a 
    one-dimension tensor n_lim_eff of shape (batch size)
    
    Parameters
    ----------
    n_lim_eff : torch.Tensor
        Two-dimension tensor of shape (batch size, number of events).
    n_lim : int
        Required number of events.
    
    Returns
    -------
    bool
        True if there are enough detected events, False otherwise.

    """

    return torch.any(n_lim_eff != n_lim)

def limit_ones_vectorized(tensor, n_lim, maximum_population_size):

    """
    This function takes as input a tensor of shape (batch size, number of events) and a tensor
    of shape (batch size) that specifies the number of events that were drawn.
    Finally, maximum_population_size defines the maximum number of events that can be drawn.

    The function returns two tensors: 

    (1) A tensor of shape (batch size, maximum_population_size) where per batch we have exactly
    n_lim ones. I.e. we cut all ones per batch if they exceed n_lim.

    (2) A tensor of shape (batch size, maximum_population_size) where per batch we have exactly
    maximum_population_size ones. I.e. we add ones so we have exactly maximum_population_size ones

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor of shape (batch size, number of events).
    n_lim : torch.Tensor
        Tensor of shape (batch size).
    maximum_population_size : int

    Returns
    -------
    limited_tensor : torch.Tensor
        Tensor of shape (batch size, maximum_population_size).
    limited_tensor2 : torch.Tensor
        Tensor of shape (batch size, maximum_population_size).

    
    """

    if torch.any(n_lim > maximum_population_size):
        raise 'Cannot produce more event than the maximum number of events. '

    device = tensor.device

    start_time = time.time()
    times = {}
    
    # Get the shape of the input tensor
    num_batch, num_events = tensor.size()

    tensor_int = (~tensor).int()
    
    # Get the indices where the True values are located, sorted along the rows
    sorted_indices = torch.argsort(tensor_int, dim=1, descending=False)

    n_detected_per_pop = tensor.sum(axis=1)

    print('Minimum events: ', torch.min(n_detected_per_pop))

    n_lim.to(device)
    n_lim_eff = torch.minimum(n_lim, n_detected_per_pop)

    # make sure there are enough detected embeddings
    if check_below_required_event_numbers(n_lim_eff, n_lim):
        print("Not enough detected embeddings.")
    
    # Create an index mask that selects the first n_lim ones for each row
    mask = (torch.arange(num_events).expand(num_batch, num_events).to(device) < n_lim_eff[:,None])
    mask2 = (torch.arange(num_events).expand(num_batch, num_events).to(device) < maximum_population_size)

    # Use the sorted indices to mask out the extra ones
    limited_tensor = torch.zeros_like(mask)
    limited_tensor2 = torch.zeros_like(mask2)

    limited_tensor.scatter_(1, sorted_indices, mask)
    limited_tensor2.scatter_(1, sorted_indices, mask2)

    return limited_tensor, limited_tensor2

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

    time_start = time.time()
    train_time = {}

    # sample from embedding emulator
    emb = embedding_emulator.sample(x=x)

    num_batch, num_events, _ = emb.size()

    train_time['compute_embeddings'] = time.time() - time_start

    # check whether embeddings are detected
    idx = selection_model.apply_selection_to_embeddings(emb)

    train_time['compute_selection'] = time.time() - train_time['compute_embeddings'] - time_start

    # cut all detections above the required number of events
    idx, idx2 = limit_ones_vectorized(idx, number_events, maximum_population_size)

    train_time['compute_idx'] = time.time() - train_time['compute_selection'] - time_start

    # mask embeddings
    emb[~idx] = 0

    train_time['set_emb_to_zero'] = time.time() - train_time['compute_idx'] - time_start

    # for k in train_time.keys():
    #     print(f"{k}: {train_time[k]}")

    # only return detected embeddings
    emb_det = emb[idx2].view(num_batch, maximum_population_size, -1)

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

    pm_embedding_emulator = pm.embedding_emulator
    selection_model = pm.snr_estimator

    pm_embedding_emulator.model.eval()
    selection_model.model.eval()

    maximum_population_size = pm.metadata["train_settings"]["data"]["maximum_population_size"]

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

def check_below_required_event_numbers(n_lim_eff, n_lim):

    """
    Check whether there are enough detected events. Takes as input a 
    one-dimension tensor n_lim_eff of shape (batch size)
    
    Parameters
    ----------
    n_lim_eff : torch.Tensor
        Two-dimension tensor of shape (batch size, number of events).
    n_lim : int
        Required number of events.
    
    Returns
    -------
    bool
        True if there are enough detected events, False otherwise.

    """

    return torch.any(n_lim_eff != n_lim)