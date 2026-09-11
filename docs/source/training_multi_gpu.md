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
  `num_gpus` — no changes to the `condor:` block are needed. Cluster-specific
  submit-file directives (e.g., full-node templates) can be passed through
  verbatim via `condor: extra_submit_lines:`.

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
  num_workers: 32   # total across GPUs (= 4 per GPU); scale with num_gpus
  num_gpus: 8
  # ddp_port: 12355  # change when running several DDP jobs on one node
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

### Normalization layers

`BatchNorm` computes its statistics from the batch on each GPU, so under DDP every replica normalizes with
slightly different statistics, and synchronizing them (`SyncBatchNorm`) stalls the forward pass at every
normalization layer. Use `LayerNorm` instead, which normalizes each sample independently and needs no
communication between GPUs. The `norm` option of the flow and embedding network settings selects the
normalization layer (`LayerNorm`, `BatchNorm` or `null` for none). The single-GPU example settings use
`BatchNorm`; for DDP, switch both to `LayerNorm`:

```yaml
model:
  posterior_kwargs:
    base_transform_kwargs:
      norm: LayerNorm
  embedding_kwargs:
    norm: LayerNorm
```

Checkpoints trained with the former `batch_norm: True` setting keep their `BatchNorm` layers when loaded.
If a network with `BatchNorm` layers is trained on multiple GPUs, they are converted to `SyncBatchNorm`
(with a warning), so that the batch statistics are still computed over the full effective batch.

`LayerNorm` is only available for `base_transform_type: rq-coupling`. For `rq-autoregressive` transforms
it would break the causal structure of the MADE layers, so `BatchNorm` or `null` must be used there.

### Compiling the network (`torch_compile`)

The neural spline flow launches tens of thousands of small CUDA kernels per step, so on a
modern GPU the step is bound by kernel-launch overhead rather than by arithmetic (GPU
utilization plateaus well below 100% even at large batch sizes). Setting `torch_compile: true`
in the `local` section wraps the network with
[`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html), which fuses
these kernels. The gain depends on how launch-bound the network is: for the `npe_model` example
network (30 flow steps, `hidden_dim` 1024) trained on a 10M-waveform dataset at a per-GPU batch
size of 4096 (A100), the training step is 1.22× faster on one GPU and 1.14× faster on four GPUs;
with `hidden_dim` 512 it is 1.4× / 1.35×. Peak GPU memory drops by about 20%. The speedup is
independent of `num_gpus` and larger at smaller per-GPU batch sizes.

```yaml
local:
  torch_compile: true                    # default: false
  torch_compile_cache_dir: /scratch/tmp  # optional; see note below
```

Notes:

- Compilation is slow: 4–12 minutes for a production-size network, paid on the first training
  step of a run and again at every stage boundary that changes which parameters are trainable.
  The compiled steps that follow are the fast ones, so `torch_compile` pays off for trainings
  of tens of epochs or more, not for short runs. To avoid further compilations the *training*
  loader drops the last, smaller batch of each epoch (the compiled graph is specialized to the
  batch shape; the test loader keeps it) and the test epoch runs the network eagerly (an
  eval-mode graph would cost another compilation that a short test epoch never amortizes).
- A stage boundary that changes `freeze_rb_layer` discards the compiled graphs and the next
  step recompiles (a compiled graph does not track which parameters are trainable).
- `torch_compile_cache_dir` sets the base directory of the on-disk Inductor/Triton cache
  (default: the system temp directory); under DDP each rank gets its own subdirectory. That
  cache must live on **node-local** disk. If the system temp directory is a shared network
  filesystem, set it to a node-local path (e.g. the HTCondor scratch directory); otherwise a
  just-compiled kernel can be unloadable on another rank and the run hangs on an NCCL timeout.
- Once the network step is faster, the dataloader can become the bottleneck: watch
  `Time Dataloader` in the log and raise `num_workers` if needed.

### TensorFloat-32 matrix multiplications (`float32_matmul_precision`)

PyTorch runs float32 matrix multiplications at full precision by default, which leaves the
tensor cores of Ampere and newer GPUs unused. Setting

```yaml
local:
  float32_matmul_precision: high   # default: highest
```

lets them use TensorFloat-32: matmul inputs are rounded to 10 mantissa bits (the range of
float32 is kept, and accumulation, weights, gradients and optimizer state stay float32). For
the `npe_model` network (`hidden_dim` 1024) this roughly doubles the speed of the network step
and combines with `torch_compile`: with both (and the fused optimizer below) the network step
drops from 0.85 s to 0.28 s per 4096 samples on an A100, a 3× gain in GPU time. All non-matmul
operations (splines, normalization, the optimizer) are unaffected.

TF32 and `automatic_mixed_precision` are alternatives, not a stack: under AMP the matmuls
already run in float16, so `float32_matmul_precision: high` changes nothing there. On the
`npe_model` network TF32 alone and AMP alone give the same speedup (0.90 to 0.47 s per step);
TF32 is the option for trainings that stay in float32. The value in effect is stored in the
checkpoint metadata (`float32_matmul_precision`).

### Fused optimizer (`fused`)

Unlike the settings above, this one is not in the `local` section: it belongs to the
`optimizer` block of an individual training stage, and is passed straight through to the
PyTorch optimizer. Setting `fused: true` selects the fused CUDA kernel, which performs the
whole optimizer update in one kernel instead of one per step of the update math, and is about
10% faster per step for large networks:

```yaml
training:
  stage_0:
    optimizer:
      type: adam
      lr: 0.0001
      fused: true   # default: false
```

Each stage builds its own optimizer, so this has to be repeated in every stage that should use
it. It is supported by `adam`, `adamw`, `adagrad` and `sgd` (not `lbfgs`); the fused kernel
exists for CUDA and (since PyTorch 2.4) CPU tensors, the speedup matters on the GPU.

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
  time dominates, increase `num_workers`. Like `batch_size`, `num_workers` is the
  total across all GPUs and is divided equally between them (aim for 4–8 per GPU).
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
