# Quickstart tutorial

The `.yaml` files contained in this directory (and subdirectories) contain configuration settings for the various Dingo tasks. These files are typically provided as input to the command-line scripts, which then run Dingo and save output files. These output files contain as metadata the settings in the `.yaml` files, and they may usually be inspected by running `dingo_ls`.

After configuring the settings files, the scripts may be used as follows, assuming the Dingo `venv` is active.

## Generate training data

### Waveforms

To generate a waveform dataset for training, execute
```
dingo_generate_dataset --settings_file waveform_dataset_settings.yaml --num_processes N --out_file waveform_dataset.hdf5
```
where `N` is the number of processes you would like to use to generate the waveforms in parallel. This saves the dataset of waveform polarizations in the file `waveform_dataset.hdf5` (typically compressed using SVD, depending on configuration).

One can use `dingo_generate_dataset_dag` to set up a condor DAG for generating waveforms on a cluster. This is typically useful for slower waveform models.

### Noise ASDs

(Jonas)

## Training

With a waveform dataset and ASD dataset(s), one can train a neural network. Configure the `train_settings.yaml` file to point to these datasets, and run
```
dingo_train --settings_file train_settings.yaml --train_dir train_dir
```
This will configure the network, train it, and store checkpoints, a record of the history, and the final network in the directory `train_dir`. Alternatively, to resume training from a checkpoint file, run
```
dingo_train --checkpoint model.pt --train_dir train_dir
```
If using CUDA on a machine with several GPUs, be sure to first select the desired GPU number using the `CUDA_VISIBLE_DEVICES` environment variable. If using a cluster, Dingo can be trained using `dingo_train_condor`.

## Inference

Once a Dingo model is trained, inference for real events can be performed with 
```
dingo_analyze_event 
  --model model 
  --gps_time_event gps_time_event
  --num_samples num_samples
  --batch_size batch_size
```
where `model` is the path of the trained Dingo mode, `gps_time_event` is the GPS time of the event to be analyzed (e.g., 1126259462.4 for GW150914), `num_samples` is the number of desired samples and `batch_size` is the batch size (the larger the faster the computation, but limited by GPU memory). 

If Dingo was trained using GNPE, by setting the `data/gnpe_time_shifts` option in the settings file, one needs to provide an additional Dingo model for the initialization of inference. This model infers initial estimates for the coalescence times in the individual detectors and is trained just like any other dingo model. See `training/train_settings_init.yaml` for an example settings file. The command for GNPE inference reads
```
dingo_analyze_event 
  --model model 
  --model_init model_init
  --gps_time_event gps_time_event
  --num_samples num_samples
  --num_gnpe_iterations num_gnpe_iterations
  --batch_size batch_size
```
where `model_init` is the path of the aforementioned initialization model, and `num_gnpe_iterations` specifies the number of GNPE iterations (typically, `num_gnpe_iterations=30`).

Finally, the option `--event_dataset </path/to/event_dataset.hdf5>` can be set to cache downloaded event data for future use.

## Importance sampling

(Max)
