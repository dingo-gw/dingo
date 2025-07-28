import pickle
import copy

import numpy as np
import torch
from typing import Dict, Union

import dingo

def train_epoch_pdet_model(pm, dataloader, error_func=None):

    if error_func == None:
        error_func = binary_cross_entropy

    snr_threshold = pm.snr_threshold

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
        
        params = data[0][...,:-1]
        snr = data[0][...,-1]
        
        mask = snr_threshold < snr
        log_pdet_pred = pm(params).squeeze()

        # compute loss
        loss = error_func(mask.float(), log_pdet_pred).mean()
        # backward pass and optimizer step
        loss.backward()
        pm.optimizer.step()
        # update loss for history and logging
        loss_info.update(loss.detach().item(), len(data[0]))
        loss_info.print_info(batch_idx)

    return loss_info.get_avg()


def test_epoch_pdet_model(pm, dataloader, error_func=None):

    if error_func == None:
        error_func = binary_cross_entropy

    snr_threshold = pm.snr_threshold

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

            params = data[0][...,:-1]
            snr = data[0][...,-1]
            
            mask = snr_threshold < snr
            log_pdet_pred = pm(params).squeeze()

            # compute loss
            loss = error_func(mask.float(), log_pdet_pred).mean()
            # update loss for history and logging
            loss_info.update(loss.item(), len(data[0]))
            loss_info.print_info(batch_idx)

        return loss_info.get_avg()

def binary_cross_entropy(x, log_x_pred):
    neg_loss = x * log_x_pred + (1 - x) * torch.log(1 - torch.exp(log_x_pred))
    
    return - neg_loss