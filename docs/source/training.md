# Training

Training a network can require a significant amount of time (for production models, typically a week with a fast GPU). We therefore expect that this will almost always be done non-interactively using a command-line script. Dingo offers two options, `dingo_train` and `dingo_train_condor`, depending on whether your GPU is local or cluster-based.

Both of these scripts take as main argument a settings file, which specifies options relating to [](training_transforms.ipynb), training strategy, [](network_architecture.ipynb), hardware, and checkpointing. They produce a trained model in PyTorch `.pt` format, and they save checkpoints and the training history. The settings file is furthermore saved within the model files for reproducibility and to be able to resume training from a checkpoint. Finally, all *precursor* settings files (for the waveform or noise datasets) are also saved with the model.

## Settings file

```{code-block} yaml
---
caption: Example `train_settings.yaml` file. This is also available in the examples/ folder. The specific settings listed will train a production-size network, taking about a week on an NVIDIA A100. Consider reducing some model hyperparameters for experimentation.
---
data:
  waveform_dataset_path: /path/to/waveform_dataset.hdf5  # Contains intrinsic waveforms
  train_fraction: 0.95
  window:
    type: tukey
    f_s: 4096
    T: 8.0
    roll_off: 0.4
  domain_update:
    f_min: 20.0
    f_max: 1024.0
  svd_size_update: 200 
  detectors:
    - H1
    - L1
  extrinsic_prior:
    dec: default
    ra: default
    geocent_time: bilby.core.prior.Uniform(minimum=-0.10, maximum=0.10)
    psi: default
    luminosity_distance: bilby.core.prior.Uniform(minimum=100.0, maximum=1000.0)
  ref_time: 1126259462.391
  gnpe_time_shifts:
    kernel: bilby.core.prior.Uniform(minimum=-0.001, maximum=0.001)
    exact_equiv: True
  inference_parameters: default

model:
  type: nsf+embedding
  nsf_kwargs:
    num_flow_steps: 30
    base_transform_kwargs:
      hidden_dim: 512
      num_transform_blocks: 5
      activation: elu
      dropout_probability: 0.0
      batch_norm: True
      num_bins: 8
      base_transform_type: rq-coupling
  embedding_net_kwargs:
    output_dim: 128
    hidden_dims: [1024, 1024, 1024, 1024, 1024, 1024,
                  512, 512, 512, 512, 512, 512,
                  256, 256, 256, 256, 256, 256,
                  128, 128, 128, 128, 128, 128]
    activation: elu
    dropout: 0.0
    batch_norm: True
    svd:
      num_training_samples: 20000
      num_validation_samples: 5000
      size: 200

# Training is divided in stages. They each require all settings as indicated below.
training:
  stage_0:
    epochs: 300
    asd_dataset_path: /path/to/asds_fiducial.hdf5
    freeze_rb_layer: True
    optimizer:
      type: adam
      lr: 0.0001
    scheduler:
      type: cosine
      T_max: 300
    batch_size: 64

  stage_1:
    epochs: 150
    asd_dataset_path: /path/to/asds.hdf5
    freeze_rb_layer: False
    optimizer:
      type: adam
      lr: 0.00001
    scheduler:
      type: cosine
      T_max: 150
    batch_size: 64

# Local settings that have no impact on the final trained network.
local:
  device: cpu  # Change this to 'cuda' for training on a GPU.
  num_workers: 6
#  wandb:
#    project: dingo
#    group: O4
  runtime_limits:
    max_time_per_run: 36000
    max_epochs_per_run: 500
  checkpoint_epochs: 10
#   condor:
#     bid: 100
#     num_cpus: 16
#     memory_cpus: 128000
#     num_gpus: 1
#     memory_gpus: 8000
```

The train settings file is grouped into **four sections:**

### `data_settings`

These settings point to a saved dataset of waveform polarizations and describe the transforms to obtain detector waveforms. A detailed description of these settings is available [here](training_transforms.ipynb#building-the-transforms).

### `model`

This describes the model architecture, including network type and hyperparameters. All of these settings are described in the section on [](network_architecture.ipynb).

### `training`

This describes the training strategy. Training is divided into **stages**, each of which can differ to some extent. Stages are numbered (`stage_0`, `stage_1`, ...) and executed in this order. Each stage is defined by the following settings:

epochs
: Total number of training epochs for the stage. The network sees the entire training set once per epoch.

asd_dataset_path
: Points to an `ASDDataset` file. Each stage can have its own ASD dataset, which is useful for implementing a pre-training stage with fixed ASD and a fine-tuning stage with variable ASD.

freeze_rb_layer
: Whether to freeze the first layer of the embedding network in `nsf+embedding` models. This layer is seeded with reduced (SVD) basis vectors, so freezing this layer during pre-training simply projects data onto the basis coefficients. In the fine-tuning stage, when other weights are more stable, unfreezing this can be useful.

optimizer
: Specify [optimizer](https://pytorch.org/docs/stable/optim.html) type and parameters such as initial learning rate.

scheduler
: Use a [learning rate scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) to reduce the learning rate over time. This can improve overall optimization.

batch_size
: Number of training samples per mini-batch. For a training dataset of size $N$, then each epoch will consist of $N / \text{batch_size}$ batches. Generally training will be faster for a larger batch size, but will require additional iterations.

```{important}
The stage-training framework allows for separate pre-training and fine-tuning stages. We found that having a pre-training stage where we freeze certain network weights and fix the noise ASD improves overall training results.
```

### `local`

The `local` settings are the only group that have no impact on the final trained network. Indeed, they are not even saved in the `.pt` files; rather they are split off and saved in a new file `local_settings.yaml`.

device
: `cpu` or `cuda`. Training on a GPU with CUDA is highly recommended.

num_workers
: Number of CPU worker processes to use for pre-processing training data before copying to the GPU. Data pre-processing (inluding decompression, projection to detectors, and noise generation) is quite expensive, so using 16 or 32 processes is recommended, otherwise this can become a bottleneck. We recommend monitoring the GPU utilization percentage as well as time spent on pre-processing (output during training) to fine-tune this number.

wandb
: Settings for [Weights & Biases](https://wandb.ai/site). If you have an account, you can use this to track your training progress and compare different runs.

runtime_limits
: Maximum time (in seconds) or maximum number of epochs per run. Using this could make sense in a cluster environment.

checkpoint_epochs
: Dingo saves a temporary checkpoint in `model_latest.py` after every epoch, but this is later overwritten by the next checkpoint. This setting saves a permanent checkpoint after the specified number of epochs. Having these checkpoints can help in recovering from training failures that do not result in program termination.

condor
: Settings for [HTCondor](https://htcondor.readthedocs.io/en/latest/index.html). The condor script will (re)submit itself according to these options.

## Command-line scripts

### `dingo_train`

On a local machine, simply pass the settings file (or checkpoint) and an output directory to `dingo_train`. It will train until complete, or until a runtime limit is reached.

```text
usage: dingo_train [-h] [--settings_file SETTINGS_FILE] --train_dir TRAIN_DIR [--checkpoint CHECKPOINT]

Train a neural network for gravitational-wave single-event inference.

This program can be called in one of two ways:
    a) with a settings file. This will create a new network based on the 
    contents of the settings file.
    b) with a checkpoint file. This will resume training from the checkpoint.

optional arguments:
  -h, --help            show this help message and exit
  --settings_file SETTINGS_FILE
                        YAML file containing training settings.
  --train_dir TRAIN_DIR
                        Directory for Dingo training output.
  --checkpoint CHECKPOINT
                        Checkpoint file from which to resume training.
```

### `dingo_train_condor`

On a cluster using HTCondor, use `dingo_train_condor`. This calls itself recursively as follows:
1. The first time you call it, use the flag `--start-submission`. This creates a condor submission file `submission_file.sub` that again calls the executable `dingo_train_condor` (now without the flag) and submits it. This will run `dingo_train_condor` directly on the cluster node that is assigned.
2. On the cluster node, `dingo_train_condor` first trains the network until done or a runtime limit is reached (be careful to set this shorter than the condor time limit). Then it creates a *new* submission file that once again calls `dingo_train_condor`, and submits it. This will resume the run on a new node, and repeat.

```text
usage: dingo_train_condor [-h] --train_dir TRAIN_DIR [--checkpoint CHECKPOINT] [--start_submission]

optional arguments:
  -h, --help            show this help message and exit
  --train_dir TRAIN_DIR
                        Directory for Dingo training output.
  --checkpoint CHECKPOINT
  --start_submission
```

## Output

Output from training is stored in the `TRAIN_DIR` folder passed to the training scripts. This consists of the following:
* `model_latest.pt` checkpoints every epoch (overwritten);
* `model_XXX.pt` checkpoints where `XXX` is the epoch number, every `checkpoint_epochs` epochs;
* `model_stage_X.pt` at the end of training stage `X`;
* `history.txt` with columns (epoch number, train loss, test loss, learning rate);
* `svd_L1.hdf5`, ..., storing SVD basis information used for seeding the embedding network;
* `local_settings.yaml` with local settings for the run (not stored with checkpoints).

The `.pt` and `.hdf5` files may all be inspected using `dingo_ls`. This prints all the settings, as well as diagnostic information for SVD bases. The saved settings include all the settings provided in the settings file, as well as several derived quantities, such as parameter standardizations, additional context parameters (for GNPE), etc.

### Modifying a checkpoint

Occasionally it may be necessary to change a setting of a partially trained model. For example, a model may have been successfully pre-trained, but the fine-tuning failed, and one may wish to change the fine-tuning settings without starting from scratch. Since the model setting are all stored with the checkpoint, they just need to be changed.

The script `dingo_append_training_stage` allows for appending a model stage or replacing an existing planned stage. It will fail if the stage has already begun training, so be sure to use it on a sufficiently early checkpoint.
```text
usage: dingo_append_training_stage [-h] --checkpoint CHECKPOINT --stage_settings_file STAGE_SETTINGS_FILE --out_file OUT_FILE [--replace REPLACE]

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
  --stage_settings_file STAGE_SETTINGS_FILE
  --out_file OUT_FILE
  --replace REPLACE
```

For more detailed adjustments to the training settings the script one can use the script `compatibility/update_model_metadata.py`.
```text
usage: update_model_metadata.py [-h] --checkpoint CHECKPOINT --key KEY [KEY ...] --value VALUE

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
  --key KEY [KEY ...]
  --value VALUE
  ```
  
```{warning}
Modifications to model metadata can easily break things. Do not use this unless completely sure what you are doing!
```
