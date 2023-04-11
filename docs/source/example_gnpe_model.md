# GNPE model (production)

This tutorial has the highest profile settings and is the one typically used for production use.
The main difference from the [NPE](example_npe_model.md) tutorial is that here we are now using [GNPE](gnpe.md)
(group neural posterior estimation). The data generation is exactly the same as the [previous](example_npe_model.md)
tutorial, but we repeat it here, for completeness.


Step 1 Generating a Waveform Dataset
------------------------------------ 

We generate the waveform dataset locally:

```
cd dingo
mkdir $(pwd)/gnpe_model_train_dir
export TRAIN_DIR=$(pwd)/gnpe_model_train_dir
dingo_generate_dataset --settings examples/gnpe_model/waveform_dataset_settings.yaml --out_file $TRAIN_DIR/waveform_dataset.hdf5
```

or using condor:

```
dingo_generate_dataset_dag --settings_file
$(pwd)/examples/gnpe_model/waveform_dataset_settings.yaml --out_file
$TRAIN_DIR/IMRPhenomXPHM.hdf5 --env_path $DINGO_VENV_PATH --num_jobs 4
--request_cpus 16 --request_memory 1280000 --request_memory_high 256000
```


Step 2 Generating an ASD dataset
--------------------------------

As before we generate a fiducial ASD dataset containing a single ASD:

```
dingo_generate_asd_dataset --settings_file examples/gnpe_model/asd_dataset_settings_fiducial.yaml --data_dir
$TRAIN_DIR/asd_dataset_folder -out_name $TRAIN_DIR/asds_O1_fiducial.hdf5

and a large ASD dataset:

dingo_generate_asd_dataset --settings_file $(pwd)/examples/gnpe_model/asd_dataset_settings.yaml --data_dir
$TRAIN_DIR/asd_dataset_folder -out_name $TRAIN_DIR/asds_O1.hdf5
```


Step 3 Training the network
---------------------------

Now we are ready for training using GNPE. Here we need to train two networks, one which estimates the time of arrival 
in the detectors and one which does the full inference task. First we train the initialization network for the detector times with:

```
sed -i 's+/path/to/waveform_dataset.hdf5+'"$TRAIN_DIR"'/waveform_dataset.hdf5+g' examples/gnpe_model/train_settings_init.yaml
sed -i 's+/path/to/asds_fiducial.hdf5+'"$TRAIN_DIR"'/asd_dataset_folder/asds_O1_fiducial.hdf5+g' examples/gnpe_model/train_settings_init.yaml
sed -i 's+/path/to/asds.hdf5+'"$TRAIN_DIR"'/asd_dataset_folder/asds_O1.hdf5+g' examples/gnpe_model/train_settings_init.yaml
dingo_train --settings_file examples/gnpe_model/train_settings.yaml --train_dir $TRAIN_DIR/init_network
```

Notice that the inference parameters are only the `H1_time` and `L1_time`. We train the main network 
for the `default_inference_parameters` (defined [here](https://github.com/dingo-gw/dingo/blob/main/dingo/gw/prior.py)) 
using:

```
sed -i 's+/path/to/waveform_dataset.hdf5+'"$TRAIN_DIR"'/waveform_dataset.hdf5+g' examples/gnpe_model/train_settings.yaml
sed -i 's+/path/to/asds_fiducial.hdf5+'"$TRAIN_DIR"'/asd_dataset_folder/asds_O1_fiducial.hdf5+g' examples/gnpe_model/train_settings.yaml
sed -i 's+/path/to/asds.hdf5+'"$TRAIN_DIR"'/asd_dataset_folder/asds_O1.hdf5+g' examples/gnpe_model/train_settings.yaml
dingo_train --settings_file examples/gnpe_model/train_settings.yaml --train_dir $TRAIN_DIR/main_network
```

Notice the `data.gnpe_time_shifts` section. The `kernel` describes how much to blur the GNPE proxies and is specified in 
seconds. To read more about this see [GNPE](gnpe.md).


Step 4 Doing Inference
----------------------

Performing inference requires a few changes to the previous NPE setup. Most notably, since we are now using GNPE, we 
have to specify the file path to both the initialization network and the main network. Another 
difference is the new attribute under sampler arguments `num-gnpe-iterations` which indicates the 
number of GNPE steps to take. If the initialization network is not fully converged or if 
the length of the segment being analyzed is very long, it is recommended to increase this number.

```
sed -i "s|TRAIN_DIR/|$TRAIN_DIR/|g" examples/gnpe_model/GW150914.ini
sed -i "s|/path/to/init_model.pt|$TRAIN_DIR/init_network/model_latest.pt|g" examples/gnpe_model/GW150914.ini
sed -i "s|/path/to/model.pt|$TRAIN_DIR/main_network/model_latest.pt|g" examples/gnpe_model/GW150914.ini
dingo_pipe examples/gnpe_model/GW150914_toy.ini
```

