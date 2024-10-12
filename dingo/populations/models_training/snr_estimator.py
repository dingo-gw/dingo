import pickle
import copy

import numpy as np
import torch
from typing import Dict, Union

import dingo
from dingo.gw.injection import Injection
import dingo.core.utils as utils

import time

def train_epoch_snr_estimator(pm, dataloader, error_func=None):

    if error_func == None:
        error_func = mean_square_error

    pm_single_event = pm.pm_single_event

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
        
        snr = data[0]
        context = pm_single_event.model(*data[1:])
        # compute loss
        loss = error_func(snr, pm.model(context)).mean()
        # backward pass and optimizer step
        loss.backward()
        pm.optimizer.step()
        # update loss for history and logging
        loss_info.update(loss.detach().item(), len(data[0]))
        loss_info.print_info(batch_idx)

    return loss_info.get_avg()


def test_epoch_snr_estimator(pm, dataloader, error_func=None):

    if error_func == None:
        error_func = mean_square_error

    pm_single_event = pm.pm_single_event

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

            snr = data[0]
            context = pm_single_event.model(*data[1:])
            # compute loss
            loss = error_func(snr, pm.model(context)).mean()
            # update loss for history and logging
            loss_info.update(loss.item(), len(data[0]))
            loss_info.print_info(batch_idx)

        return loss_info.get_avg()

def mean_square_error(x, y):
    return (x - y).abs()**2