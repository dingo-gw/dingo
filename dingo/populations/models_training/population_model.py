import torch

import dingo

import time
from threadpoolctl import threadpool_limits

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

def get_detected_embeddings(x, embedding_emulator, selection_model, number_events_per_batch, maximum_population_size):

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
    number_events_per_batch : int
        Number of events to detect.

    Returns
    -------
    emb_det : torch.Tensor
        Detected embeddings.
    
    """

    # sample from embedding emulator
    emb = embedding_emulator.sample(x=x, batch_size=None)

    num_batch, num_events, _ = emb.size()

    # check whether embeddings are detected
    mask = selection_model.apply_selection_to_embeddings(emb)

    # cut all detections above the required number of events in mask1
    # Fill up mask2 with ones so that each population contains exactly maximum_population_size events
    # The unused events are set to zero via mask1
    mask1, mask2 = limit_ones_vectorized(mask, number_events_per_batch, maximum_population_size)

    # mask all non-detected embeddings
    emb[~mask1] = 0

    # only return detected embeddings
    # shape is cut-down from (num_batch, num_events, emb_dim) to 
    # (num_batch, maximum_population_size, emb_dim)
    emb_det = emb[mask2].view(num_batch, maximum_population_size, -1)

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
        
        # assert emb_det.device == pm.device
        # emb_det.to(pm.device, non_blocking=True)

        mask = (emb_det == 0).all(dim=-1)
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
            emb_det.to(pm.device, non_blocking=True)
            mask = (emb_det == 0).all(dim=-1)

            # compute loss
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