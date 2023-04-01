# Example Normalizing Flow

We will now do a tutorial with higher profile settings. Note these are not the
full production settings used for runs since we are not using [gnpe]GNPE. The
steps are the essentially same as [example_toy]toy but with higher level settings. It is
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

Note there is a new attribute `compression`. This creates a singular value decomposion 
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
dingo_generate_dataset_dag

```

Step 2 Generating an ASD dataset
--------------------------------

To generate an ASD dataset we can run the same command as last time. 

```
dingo_generate_asd_dataset --settings_file examples/toy_example/asd_dataset_settings.yaml --data_dir $TRAIN_DIR/asd_dataset_folder
```

However, this time, during training we will need two sets of ASDs one which will fixed during the initial training. This dataset will 
contain only 1 ASD. Then one ASDDataset which contains many ASDs which is used during the fine tuning stage. 


Step 3 Training the network
---------------------------

Now we are ready for training. The command is analogous to the previous tutorial 
but the settings are increased to production values. To run the training do

```
sed -i 's+/path/to/waveform_dataset.hdf5+'"$TRAIN_DIR"'/waveform_dataset.hdf5+g' examples/toy_example/train_settings.yaml
sed -i 's+/path/to/asds_fiducial.hdf5+'"$TRAIN_DIR"'/asd_dataset_folder/asds_O1_fiducial.hdf5+g' examples/toy_example/train_settings.yaml
sed -i 's+/path/to/asds.hdf5+'"$TRAIN_DIR"'/asd_dataset_folder/asds_O1.hdf5+g' examples/toy_example/train_settings.yaml
dingo_train --settings_file examples/no_gnpe/train_settings.yaml --train_dir $TRAIN_DIR
```

Step 4 Doing Inference
----------------------

We can run inference with the same command as before

```
sed -i "s|TRAIN_DIR/|$TRAIN_DIR/|g" examples/toy_example/GW150914_toy.ini
sed -i "s|/path/to/model.pt|$TRAIN_DIR/model_latest.pt|g" examples/toy_example/GW150914_toy.ini
dingo_pipe examples/toy_example/GW150914_toy.ini
```

