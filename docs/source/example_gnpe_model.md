# GNPE model (production)

This tutorial has the highest profile settings and is the one typically used for production use.
The main difference from the [NPE](example_npe_model.md) tutorial is that here we are now using [GNPE](gnpe.md)
(group neural posterior estimation). The data generation is exactly the same as the [previous](example_npe_model.md)
tutorial, but we repeat it here, for completeness.

The file structure is similar to the NPE example, except now there are two
training sub-directories and two `train_settings.yaml` files. 

```
gnpe_model/

    #  config files
    waveform_dataset_settings.yaml
    asd_dataset_settings_fiducial.yaml
    asd_dataset_settings.yaml
    train_settings_main.yaml
    train_settings_init.yaml
    GW150914.ini

    training_data/
        waveform_dataset.hdf5
        asd_dataset.hdf5
        asd_dataset_fiducial.hdf5
        asd_dataset_fiducial/ # Contains the asd_dataset.hdf5 and also temp files for asd generation
        asd_dataset/ # Contains the asd_dataset.hdf5 and also temp files for asd generation

    training/
        main_train_dir/
            model_050.pt
            model_stage_0.pt
            model_latest.pt
            history.txt
            #  etc...
        init_train_dir/
            model_050.pt
            model_stage_0.pt
            model_latest.pt
            history.txt
            #  etc...

    outdir_GW150914/
        #  dingo_pipe output
```

Step 1 Generating a Waveform Dataset
------------------------------------ 


First generate the directory structure:

```
cd gnpe_model
mkdir training_data
mkdir training
mkdir training/main_train_dir
mkdir training/init_train_dir
```

Generate the waveform dataset:

```
dingo_generate_dataset --settings waveform_dataset_settings.yaml --out_file training_data/waveform_dataset.hdf5
```

or using condor:

```
dingo_generate_dataset_dag --settings_file
waveform_dataset_settings.yaml --out_file
training_data/waveform_dataset.hdf5 --env_path $DINGO_VENV_PATH --num_jobs 4
--request_cpus 16 --request_memory 1280000 --request_memory_high 256000
```


Step 2 Generating an ASD dataset
--------------------------------

As before we generate a fiducial ASD dataset containing a single ASD:

```
dingo_generate_asd_dataset --settings_file asd_dataset_settings_fiducial.yaml --data_dir
training_data/asd_dataset_fiducial -out_name training_data/asd_dataset_fiducial/asds_O1_fiducial.hdf5
```

and a large ASD dataset:

```
dingo_generate_asd_dataset --settings_file asd_dataset_settings.yaml --data_dir
training_data/asd_dataset -out_name training_data/asd_dataset/asds_O1.hdf5
```


Step 3 Training the network
---------------------------

Now we are ready for training using GNPE. Here we need to train two networks, one which estimates the time of arrival 
in the detectors and one which does the full inference task. A natural question
is why train two networks. The main idea is if one is able to align (and thus
standardize) the times of arrival in the detectors, the inference task will
become significantly easier. To do this we first need to train an initialization
network which estimates the time of arrival in the detectors:

```
dingo_train --settings_file train_settings_init.yaml --train_dir training/init_network
```

Notice that the inference parameters are only the `H1_time` and `L1_time`. Also notice that the embedding_net 
is significantly smaller and the number of flow steps, `num_flow_steps` is reduced.

```
dingo_train --settings_file train_settings_main.yaml --train_dir training/main_network
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
dingo_pipe GW150914.ini
```

