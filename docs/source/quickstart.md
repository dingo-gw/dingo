# Quickstart tutorial

To learn to use Dingo, we recommend starting with the examples provided in the [examples/](https://github.com/dingo-gw/dingo/tree/main/examples)
folder. The YAML files contained in this directory (and subdirectories) contain
configuration settings for the various Dingo tasks (constructing training data, training networks, and performing inference). These files should be provided as input to the
command-line scripts, which then run Dingo and save output files. These output files
contain as metadata the settings in the YAML files, and they may usually be inspected
by running `dingo_ls`.

```{mermaid}
flowchart TB
    dataset_settings[dataset_settings.yaml]
    dataset_settings-->generate_dataset(["dingo_generate_dataset
    #nbsp; #nbsp; --settings_file dataset_settings.yaml
    #nbsp; #nbsp; --out_file waveform_dataset.hdf5"])
    style generate_dataset text-align:left
    asd_settings[asd_dataset_settings.yaml]
    asd_settings-->generate_asd(["generate_asd_dataset
    #nbsp; #nbsp; --settings_file dataset_settings.yaml
    #nbsp; #nbsp; --data_dir asd_dataset"])
    style generate_asd text-align:left
    train_init(["dingo_train 
    #nbsp; #nbsp; --settings_file train_settings_init.yaml
    #nbsp; #nbsp; --train_dir model_init"])
    style train_init text-align:left
    train_settings_init[train_settings_init.yaml]
    train_settings_init-->train_init
    generate_dataset--->train_init
    generate_asd--->train_init
    generate_dataset--->train_main(["dingo_train 
    #nbsp; #nbsp; --settings_file train_settings_main.yaml
    #nbsp; #nbsp; --train_dir model_main"])
    style train_main text-align:left
    train_settings_main[train_settings_main.yaml]
    generate_asd--->train_main
    train_settings_main-->train_main
    train_init-->inference(["dingo_pipe GW150914.ini"])
    style inference text-align:left
    train_main-->inference
    inference-->samples[GW150914_data0_1126259462-4_sampling.hdf5]
```




After configuring the settings files, the scripts may be used as follows, assuming the
Dingo `venv` is active.

## Generate training data

### Waveforms

To generate a waveform dataset for training, execute

```
dingo_generate_dataset --settings_file waveform_dataset_settings.yaml --num_processes N --out_file waveform_dataset.hdf5
```

where `N` is the number of processes you would like to use to generate the waveforms in
parallel. This saves the dataset of waveform polarizations in the
file `waveform_dataset.hdf5` (typically compressed using SVD, depending on configuration).

One can use `dingo_generate_dataset_dag` to set up a condor DAG for generating waveforms
on a cluster. This is typically useful for slower waveform models.

### Noise ASDs

Training also requires a dataset of noise ASDs, which are sampled randomly for each
training sample. To generate this dataset based on noise observed during a run, execute

```
dingo_generate_ASD_dataset --data_dir data_dir --settings_file asd_dataset_settings.yaml
```

This will download data from [GWOSC](https://www.gw-openscience.org) and create a `/tmp` directory, in which the
estimated PSDs are stored. Subsequently, these are collected together into a final `.hdf5`
ASD dataset.
If no `settings_file` is passed, the script will attempt to use the default
one `data_dir/asd_dataset_settings.yaml`.

## Training

With a waveform dataset and ASD dataset(s), one can train a neural network. Configure
the `train_settings.yaml` file to point to these datasets, and run

```
dingo_train --settings_file train_settings.yaml --train_dir train_dir
```

This will configure the network, train it, and store checkpoints, a record of the history,
and the final network in the directory `train_dir`. Alternatively, to resume training from
a checkpoint file, run

```
dingo_train --checkpoint model.pt --train_dir train_dir
```

If using CUDA on a machine with several GPUs, be sure to first select the desired GPU
number using the `CUDA_VISIBLE_DEVICES` environment variable. If using a cluster, Dingo
can be trained using `dingo_train_condor`.

Example training files can be found under `examples/training`. 
`train_settings_toy.yaml` and `train_settings_production.yaml` train a flow to
estimate the full posterior of the event conditioned on the time of coalescence
in the detectors. The "toy" label is to indicate this should NOT be used for production but 
rather to get a feel for the Dingo pipeline. The production settings contain tested 
settings. Note that depending on the waveform model and event, these may need to occasionally
be tuned. `train_settings_init_toy.yaml` and `train_settings_init_production.yaml` train
flows to estimate the time of coalescence in the individual detectors. These two
networks are needed to use [GNPE](gnpe.md). This is the preferred and
most tested way of using Dingo. 

Alternatively, the `train_settings_no_gnpe_toy.yaml` and
`train_settings_no_gnpe_production.yaml` contain settings to train a network
without the GNPE step. Note the lack of a `data/gnpe_time_shifts` option. While this is not
recommended for production, it is still pedagogically useful and is good for prototyping 
new ideas or doing a less expensive training.   

## Inference

Once a Dingo model is trained, inference for real events can be performed using
[dingo_pipe](dingo_pipe.md). There are 3 main inference steps, downloading the data, 
running Dingo on this data and finally running importance sampling. The basic
idea is to create a .ini file which contains the filepaths of the Dingo networks
trained above and the segment of data to analyze. An example .ini file can be
found under `examples/pipe/GW150914.ini`. 

To do inference, cd into the directory with the .ini file and run 

```
dingo_pipe GW150914.ini
```


<!-- One can also just run the network without doing importance sampling with the following
command line argument. 

```
dingo_analyze_event
  --model model
  --model_init model_init
  --gps_time_event gps_time_event
  --num_samples num_samples
  --num_gnpe_iterations num_gnpe_iterations
  --batch_size batch_size
```

where model.pt is the path of the trained Dingo mode, gps_time_event is the GPS
time of the event to be analyzed (e.g., 1126259462.4 for GW150914), num_samples
is the number of desired samples and batch_size is the batch size (the larger
the faster the computation, but limited by GPU memory). Dingo downloads the
event data from GWOSC. It also estimates the noise ASD from data prior to the
event. -->
