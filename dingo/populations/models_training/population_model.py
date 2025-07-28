import torch

import dingo

import time
from threadpoolctl import threadpool_limits

def limit_mask(mask: torch.Tensor, max_ones: int) -> torch.Tensor:
    """
    Vectorized version that limits the number of True values per batch in a boolean mask.

    Parameters:
        mask (torch.Tensor): Boolean tensor of shape (batch_size, num_events).
        max_ones (int): Maximum number of True values allowed per row.

    Returns:
        torch.Tensor: Boolean tensor with the same shape, each row has at most max_ones True values.
    """

    mask_per_batch = torch.sum(mask, axis=-1)

    # print(f"Mask per batch: {torch.min(mask_per_batch)}")

    if torch.any(mask_per_batch < max_ones):
        raise ValueError(f'Not enough True entries in mask, min: {torch.min(mask_per_batch)}. ')

    # Convert boolean mask to int for sorting
    mask_int = mask.int()  # (B, E)

    # Cumsum across events, but only for True entries
    cumsum = torch.cumsum(mask_int, dim=1)

    # Create a boolean mask: True if cumsum <= max_ones and original mask was True
    limited_mask = (cumsum <= max_ones) & mask.bool()

    return limited_mask

def get_params_with_pdet(x, pdet_model, number_events_maximum):

    # shape x: (batch size, large number of events, number of parameters)

    batch_size, num_events, num_params = x.size()

    # draw uniformly between 0 and 1
    random_0_1 = torch.rand(batch_size, num_events, device=x.device)
    log_pdet = pdet_model(x).squeeze(-1)  # shape (batch size, number of events)
    pdet = torch.exp(log_pdet)

    mask = (random_0_1 < pdet)
    # limit the mask to the maximum number of events
    mask = limit_mask(mask, number_events_maximum)

    output_tensor = x[mask.bool()].view(batch_size, number_events_maximum, num_params)

    # output_tensor is now of shape (batch size, number_events_maximum, number of parameters)
    return output_tensor

def get_mask_from_population_size(number_events_per_batch, maximum_population_size):
    """
    Create a mask from the number of events per batch.

    Parameters
    ----------
    number_events_per_batch : torch.Tensor
        Tensor of shape (batch size,) containing the number of events per batch.
    maximum_population_size : int
        Maximum population size.

    Returns
    -------
    torch.Tensor
        Mask of shape (batch size, maximum_population_size) with True for valid events.
    """
    
    mask = torch.arange(maximum_population_size, device=number_events_per_batch.device).expand(
        number_events_per_batch.size(0), -1
    ) < number_events_per_batch.unsqueeze(1)

    return mask


def get_detected_embeddings(x, embedding_emulator):

    """ 
    Produce detected embeddings from a tensor x of the shape 
    (number of events, number of parameters).

    Parameters
    ----------

    x : torch.Tensor
        Tensor of shape (number of events, number of parameters).
    embedding_emulator : EmbeddingEmulator
        Embedding emulator. 

    Returns
    -------
    emb_det : torch.Tensor
        Detected embeddings.
    
    """

    # sample from embedding emulator
    return embedding_emulator.sample(x=x, batch_size=None)

def train_epoch_population_model(pm, dataloader):

    pm_embedding_emulator = pm.embedding_emulator
    pdet_model = pm.pdet_model

    pm_embedding_emulator.model.eval()
    pdet_model.model.eval()

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
        params = data[1]
        number_events_per_batch = data[2]

        # params_pdet is of shape (batch size, max number of events, number of parameters)
        params_pdet = get_params_with_pdet(params, pdet_model, maximum_population_size)
        emb_det = get_detected_embeddings(params_pdet, pm_embedding_emulator)
    
        mask = get_mask_from_population_size(number_events_per_batch, maximum_population_size)
        loss = -pm.model(hp, emb_det, mask).mean()
        
        # backward pass and optimizer step
        loss.backward()
        pm.optimizer.step()
        # update loss for history and logging
        loss_info.update(loss.detach().item(), len(data[0]))
        loss_info.print_info(batch_idx)

    return loss_info.get_avg()


def test_epoch_population_model(pm, dataloader):

    pm_embedding_emulator = pm.embedding_emulator
    pdet_model = pm.pdet_model

    pm_embedding_emulator.model.eval()
    pdet_model.model.eval()

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
            params = data[1]
            number_events_per_batch = data[2]

            # params_pdet is of shape (batch size, max number of events, number of parameters)
            params_pdet = get_params_with_pdet(params, pdet_model, maximum_population_size)
            emb_det = get_detected_embeddings(params_pdet, pm_embedding_emulator)
        
            mask = get_mask_from_population_size(number_events_per_batch, maximum_population_size)
            loss = -pm.model(hp, emb_det, mask).mean()
            
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