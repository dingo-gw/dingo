# Multi-GPU Training

Dingo supports data-parallel training across multiple GPUs using
[PyTorch DDP](https://pytorch.org/docs/stable/notes/ddp.html)
(DistributedDataParallel). This tutorial explains how DDP works in Dingo, why
simply setting `num_gpus: 8` is not enough to achieve a speedup, how to read the
extended training log, and which hyperparameters require tuning.

## How DDP works in Dingo

The goal of DDP is to increase the data throughput during training by processing
subsets of the mini-batch in parallel on separate GPUs.
Each GPU holds a full copy of the DINGO model, but each model sees a different, random subset of the waveform data set. 
After every backward pass, DDP performs a gradient all-reduce: it
aggregates gradients across all GPUs resulting in a gradient update based on the full effective batch size, 
not the per-GPU batch size.
The resulting averaged gradient is used to update all models equivalently such that every model replica stays the same. 
The optimizer step is identical to single-GPU training with the effective batch size.

Through DDP, it is possible to increase the effective batch size beyond the single-GPU memory limit.
Now, the limiting factor for training throughput is the memory limit of a single GPU which processes the effective
batch size divided by the number of GPUs. Out-of-memory errors can be addressed by either increasing the number of GPUs
or by reducing the effective batch size.

There are certain caveats to DDP:
* Since the gradients have to be synced across GPUs, the interconnect between GPUs becomes highly relevant. 
  Slow interconnects can significantly slow down training. Therefore, it is recommended to only run DDP on GPUs 
  that are located on the same node where fast interconnect is independent of network traffic.
* In addition to the full model replica (with its gradients and optimizer state) and the batch subset, 
  additional GPU memory is required for syncing and aggregating gradients.
  Therefore, the per-GPU batch size can be lower than in single-GPU training.

## Compute requirements

- A single node with N GPUs connected by fast interconnect (tested on A100 and H100 nodes).
- An NCCL-capable PyTorch build (the default for any CUDA-enabled installation).
- When submitting via HTCondor, `request_gpus` is derived automatically from
  `num_gpus` — no changes to the `condor:` block are needed.

## Settings changes

Multi-GPU training is enabled within Dingo by changing `num_gpus: 1` to the number of available GPUs.
Additionally, it is recommended to scale `batch_size` and `lr` in each training stage (see explanations below):

```yaml
# single-GPU baseline
local:
  device: cuda
  num_workers: 16
  num_gpus: 1
training:
  stage_0:
    batch_size: 4096
    optimizer:
      lr: 5.0e-5
  stage_1:
    batch_size: 4096
    optimizer:
      lr: 1.0e-5
```

```yaml
# 8-GPU DDP — scale batch_size and lr by num_gpus
local:
  device: cuda
  num_workers: 32   # scale with num_gpus, e.g. 4–8 per GPU
  num_gpus: 8
training:
  stage_0:
    batch_size: 32768   # = 4096 × 8
    optimizer:
      lr: 4.0e-4        # = 5e-5 × 8
  stage_1:
    batch_size: 32768
    optimizer:
      lr: 8.0e-5        # = 1e-5 × 8
```

### Increasing the `batch_size`

Dingo interprets `batch_size` as the **total effective batch size across all GPUs** and divides it equally:
```
per-GPU batch size = batch_size / num_gpus
```

Pytorch's `DistributedSampler` also splits the waveform dataset equally, so each GPU sees `1/N` of
the data per epoch. Putting both together, the number of optimizer steps per epoch is:
```
steps / epoch  =  dataset_size × train_fraction / batch_size
```

This formula is **independent of `num_gpus`**. The table below shows the
consequences:

| `num_gpus` | `batch_size` | per-GPU batch size | steps/epoch | outcome                                                          |
|:---:|:---:|:------------------:|:---:|:-----------------------------------------------------------------|
| 1 | 4 096 |       4 096        | N | baseline                                                         |
| 8 | 4 096 |        512         | N | same steps + all-reduce overhead → **slower** than the baseline  |
| 8 | 32 768 |       4 096        | N/8 | fewer steps, good GPU utilisation → **faster** than the baseline |

When the `batch_size` is scaled by `num_gpus`, the actual wall-clock speedup is somewhat less than 8× because the 
gradient all-reduce adds overhead per step (visible as inflated *Time Network* in the log); a factor of 4–6× is typical.

In general, it is recommended to increase the batch size as much as possible to fully utilize the GPU memory.

### Learning rate scaling

When the effective batch size increases by a factor of N, the learning rate should be increased
by the same factor (linear scaling rule, {cite:p}`Goyal:2017`):

```
lr_multi  =  lr_single × num_gpus
```

**Intuition**: with N× more samples per gradient update the gradient estimate has
lower variance, so a proportionally larger step can be taken without destabilising
training.

Every training stage has to be adapted independently.  

### Freezing layers

It is currently not possible to set `freeze_rb_layer: True` in DDP. The reason is that when starting the separate 
DDP processes, it is fixed which network parameters have to be synced across GPUs and GPU memory is allocated 
accordingly. In the current implementation, the stages are initialized (and therefore `freeze_rb_layer: True` is set) 
afterward. If the layer is frozen within a DDP process, the all-reduce fails since the reduced tensors have a different
shape than expected.
Allowing `freeze_rb_layer: True` with DDP would require significant restructuring of the code. 

## Reading the training log

Within Dingo, the separate processes for DDP are spawned after loading the waveform dataset, building the Dingo model, 
and computing the SVD which initializes the first layer of the embedding network.
At this point, a statement like 
`Process group initialised: backend=nccl, rank=0, world_size=8` is printed for every GPU
into `info.out`.
Afterward, every `print` statement within the Dingo code is executed by every GPU process (if this is not specifically 
prevented). As a result, duplicate information can appear in `info.out`. Since each GPU process prints outputs to 
`info.out` in parallel, the order of these statements can be mixed up compared to single-GPU training. 
If information is not duplicated, it is only printed by `rank = 0`.

During training, each printed line looks like the following in single-GPU mode:
```
Train Epoch: 1 [4096/4750000 (0%)]   Loss: -2.771 (-2.799)   Time Dataloader: 0.043 (0.043)   Time Network: 0.415 (0.415)
```

In multi-GPU mode, rank 0 additionally prints a third timing column:
```
Train Epoch: 1 [4096/4750000 (0%)]   Loss: -2.771 (-2.799)   Time Dataloader: 0.018 (0.018)   Time Network: 0.700 (0.700)   Time Loss Aggregation: 0.090 (0.090)
```

Each value is the current step; the value in parentheses is the running average
over the epoch. **In DDP mode all three timings are reported as the maximum across
all GPUs** (`dist.ReduceOp.MAX`), i.e. they reflect the slowest rank.

### Time Dataloader
Wall-clock time spent loading and pre-processing one batch (waveform
decompression, projection to detectors, noise generation). Expect this to be
smaller than in single-GPU mode because each GPU loads a smaller per-GPU batch.

### Time Network
Forward pass + backward pass + **DDP gradient all-reduce** + optimizer step.
Unlike single-GPU training this includes gradient synchronisation, so *Time
Network* per step will typically be *larger* than in single-GPU mode even though
each GPU processes fewer samples. The speedup comes from fewer steps per epoch,
not from faster individual steps.

### Time Loss Aggregation *(DDP only)*
Time required to reduce per-GPU losses to rank 0 and compute the global average.
Includes a `dist.barrier()` that waits for the slowest GPU to finish its network
pass before proceeding. Occasional spikes can indicate increased wait time at the barrier.
Sustained high values could indicate a load imbalance between GPUs.

### Guidelines for tuning `num_workers`

- *Time Dataloader* should be clearly smaller than *Time Network*. If dataloader
  time dominates, increase `num_workers` (aim for 4–8 per GPU).
- *Time Network* being larger per step than single-GPU is expected; this is the
  all-reduce cost.
- *Time Loss Aggregation* should be small relative to *Time Network*.

## Monitoring with WandB

Only rank 0 writes to WandB, so the run appears in the dashboard exactly as a
single-GPU run. The loss logged is the globally reduced value (weighted average
over all GPU contributions), making curves directly comparable across single- and
multi-GPU runs.

Enable WandB in the `local` section:

```yaml
local:
  wandb:
    project: dingo
    group: O4   # optional: group related runs for comparison
```
