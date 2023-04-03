# No GNPE Network

We will now do a tutorial with higher profile settings. Note these are not the
full production settings used for runs since we are not using [GNPE](gnpe.md), but
they should decent results. Go to [this](example_gnpe_network.md) tutorial for the full production network. The
steps are the essentially same as [the toy example](example_toy_network.md) but with higher level settings. It is
recommended to run this on a cluster or GPU machine. 

We can repeat the same first few steps from the previous tutorial with a couple differences. 


Step 1 Generating a Waveform Dataset
------------------------------------ 

```
cd dingo
mkdir $(pwd)/no_gnpe_network_train_dir
export TRAIN_DIR=$(pwd)/no_gnpe_network_train_dir
dingo_generate_dataset --settings examples/waveform_dataset_settings.yaml --out_file $TRAIN_DIR/waveform_dataset.hdf5
```

Note there is a new attribute `compression`. This creates a singular value decomposition
(SVD) of the waveform polarizations which is stored in disk. The `size` attribute 
describes the number of basis vectors to represent the waveform. This can later be 
changed during training. When the compression phase is finished, there will be output
displaying the mismatch between the decompressed waveform and generated waveform. You can
also access these mismatch settings by running `dingo_ls` on a generated `waveform_dataset.hdf5`
file, there will be multiple mismatches shown corresponding to the number of basis vectors used
to decompress the waveform. It is up to the user as to what type of mismatch is acceptable, typically a max 
mismatch of $10^{-3}-10^{-4}$ is recommended. 

We could also generate the waveform dataset using a dag on a cluster with condor. To do this run

```
dingo_generate_dataset_dag --settings_file $(pwd)/examples/no_gnpe_network/waveform_dataset_settings.yaml --out_file $TRAIN_DIR/IMRPhenomXPHM.hdf5 --env_path $DINGO_VENV_PATH --num_jobs 4 --request_cpus 64 --request_memory 128000 --request_memory_high 256000
```

Then run 

```
condor_submit_dag condor/submit/dingo_generate_dataset_dagman_DATE.submit
```

Where DATE is specified in the filename of the .submit file that was generated. 

Step 2 Generating an ASD dataset
--------------------------------

To generate an ASD dataset we can run the same command as last time. 

```
dingo_generate_asd_dataset --settings_file examples/no_gnpe_network/asd_dataset_settings_fiducial.yaml --data_dir $TRAIN_DIR/asd_dataset_folder -out_name $TRAIN_DIR/asds_O1_fiducial.hdf5
```

However, this time, during training we will need two sets of ASDs one which will
fixed during the initial training, this is the fiducial dataset generated above.
This dataset will contain only 1 ASD. Then one ASDDataset which contains many
ASDs which is used during the fine tuning stage. To generate this second dataset
run

```
dingo_generate_asd_dataset --settings_file $(pwd)/examples/no_gnpe_network/asd_dataset_settings.yaml --data_dir $TRAIN_DIR/asd_dataset_folder -out_name $TRAIN_DIR/asds_O1.hdf5
```

We can see that in `examples/no_gnpe_network/asd_dataset_settings.yaml` the `num_psds_max`
attribute is set to 0 indicating all possible ASDs will be downloaded. If you want to 
decrease this, make sure that there are enough ASDs in the training set to represent 
any possible data the dingo network will see. Typically this should be at least 1000,
but of course more is better. 

Step 3 Training the network
---------------------------

Now we are ready for training. The command is analogous to the previous tutorial 
but the settings are increased to production values. To run the training do

```
sed -i 's+/path/to/waveform_dataset.hdf5+'"$TRAIN_DIR"'/waveform_dataset.hdf5+g' examples/no_gnpe_network/train_settings.yaml
sed -i 's+/path/to/asds_fiducial.hdf5+'"$TRAIN_DIR"'/asd_dataset_folder/asds_O1_fiducial.hdf5+g' examples/no_gnpe_network/train_settings.yaml
sed -i 's+/path/to/asds.hdf5+'"$TRAIN_DIR"'/asd_dataset_folder/asds_O1.hdf5+g' examples/no_gnpe_network/train_settings.yaml
dingo_train --settings_file examples/no_gnpe_network/train_settings.yaml --train_dir $TRAIN_DIR
```

```{tip}
If running on a machine with multiple GPUs make sure to specify the GPU by running 
export `CUDA_VISIBILE_DEVICES=GPU_NUM` before running `dingo_train`
```

The main difference from the toy example in the network architecture is the size of the embedding
network which is described in `model.embedding_net_kwargs.hidden_dims` and the
number of neural spline flow transforms described in
`model.nsf_kwargs.num_flow_steps`. These increase the depth of the network and the 
number/size of the layers in the embedding network. 

Also notice there are now two training stages `stage_0` and `stage_1`. In `stage_0` a fixed ASD is used and the reduced basis layer
is frozen. Then in `stage_1` all ASDs are used and the reduced basis layer is unfrozen. 

The main difference in the local settings is that the `device` is set to `CUDA`.
Note if you have multiple GPUs on the machine, you can select which GPU to use
by running 

```{important}
It is recommended to have at least 40 GB of GPU memory on the device. 
```

Step 4 Doing Inference
----------------------

We can run inference with the same command as before

```
sed -i "s|TRAIN_DIR/|$TRAIN_DIR/|g" examples/no_gnpe_network/GW150914.ini
sed -i "s|/path/to/model.pt|$TRAIN_DIR/model_latest.pt|g" examples/no_gnpe_network/GW150914.ini
dingo_pipe examples/no_gnpe_network/GW150914_toy.ini
```

There is just one difference from the previous example. It is possible to reweight the posterior to a new prior 
if the user wants. Note though, that the new prior must be a subset of the previous prior. Otherwise, the proposal distribution
generated by dingo will never have seen samples from the new prior (a low effective sample size) leading to poor results. As an example see the `prior-dict` attribute in 
`examples/no_gnpe_network/GW150914.ini`.