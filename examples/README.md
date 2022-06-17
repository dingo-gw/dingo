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

To complement the waveform dataset with an ASD dataset, run
```
dingo_generate_ASD_dataset --data_dir data_dir --settings_file settings_file
```
This will download data from the GWOSC website and create a /tmp directory, in which the estimated PSDs are stored. Subsequently, these are processed together for the final ```.hdf5``` ASD dataset. 
If no ```settings_file``` is passed, the script will attempt to use the default one ```data_dir/asd_dataset_settings.yaml```. 

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

(Max)

## Importance sampling

(Max)