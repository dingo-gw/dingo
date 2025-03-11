# NPE Model (production)

We will now do a tutorial with higher profile settings. Note these are not the
full production settings used for runs since we are not using [GNPE](gnpe.md), but
they should lead to decent results. Go to [this](example_gnpe_model.md) tutorial for the full production network. The
steps are the essentially same as [the toy example](example_toy_model.md) but with higher level settings. It is
recommended to run this on a cluster or GPU machine. 

We can repeat the same first few steps from the previous tutorial with a couple
differences. The file structure is mostly the same but now there is an additional
`asd_dataset_fiducial` which will be explained below.

```
npe_model/

    #  config files
    waveform_dataset_settings.yaml
    asd_dataset_settings.yaml
    asd_dataset_settings_fiducial.yaml
    train_settings.yaml
    GW150914.ini

    training_data/
        waveform_dataset.hdf5
        asd_dataset_fiducial/ # Contains the asd_dataset.hdf5 and also temp files for asd generation
        asd_dataset/ # Contains the asd_dataset.hdf5 and also temp files for asd generation

    training/
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

Again the first step is to generate the necessary folders

```
cd npe_model
mkdir training_data
mkdir training
```

As before we run `dingo_generate_dataset`:

```
dingo_generate_dataset --settings waveform_dataset_settings.yaml --out_file training_data/waveform_dataset.hdf5
```


The `waveform_dataset_settings.yaml` settings file now includes a new attribute `compression`.
This creates a truncated singular value decomposition (SVD) of the waveform polarizations which 
is stored on disk as a compressed representation of the dataset. The `size` attribute 
refers to the number of basis vectors included in the expansion of the waveform. This can later be
changed during training. When the compression phase is finished, the log will
display the mismatch between the decompressed waveform and generated waveform. You can
also access these mismatch settings by running `dingo_ls` on a generated `waveform_dataset.hdf5`
file. It will show multiple mismatches corresponding to the number of basis vectors used
to decompress the waveform. It is up to the user as to what type of mismatch is acceptable,
typically a maximum mismatch of $10^{-3}-10^{-4}$ is recommended. 

We could also generate the waveform dataset using a
[condor DAG](https://htcondor.readthedocs.io/en/latest/users-manual/dagman-workflows.html)
on a cluster. To do this run

```
dingo_generate_dataset_dag --settings_file waveform_dataset_settings.yaml --out_file training_data/waveform_dataset.hdf5 --env_path $DINGO_VENV_PATH --num_jobs 4 --request_cpus 64 --request_memory 128000 --request_memory_high 256000
```

and then submit the generated DAG

```
condor_submit_dag condor/submit/dingo_generate_dataset_dagman_DATE.submit
```

where `DATE` is specified in the filename of the `.submit` file that was generated.


Step 2 Generating an ASD dataset
--------------------------------

To generate an ASD dataset we can run the same command as in the previous tutorial.

```
dingo_generate_asd_dataset --settings_file asd_dataset_settings_fiducial.yaml --data_dir training_data/asd_dataset_fiducial --out_name training_data/asd_dataset_fiducial/asds_O1_fiducial.hdf5
```

However, this time, during training we will need two sets of ASDs. The first one will be
fixed during the initial training -- this is the fiducial dataset generated above.
This dataset will contain only a single ASD. The second ASDDataset will contain many
ASDs and is used during the fine tuning stage. The reason to use just one ASD during the first 
stage is to allow the network to train in an easier inference setting. It should learn how
to infer parameters in the presence of that one ASD. However, during inference
the ASD will be variable. Thus, in the second stage many ASDs are used so that dingo learns the distribution
of ASDs from the observing run. We find this split leads to an improvement in
overall performance. To generate this second dataset run

```
dingo_generate_asd_dataset --settings_file asd_dataset_settings.yaml --data_dir training_data/asd_dataset --out_name training_data/asd_dataset/asds_O1.hdf5
```

We can see that in `asd_dataset_settings.yaml` the `num_psds_max`
attribute is set to `0` indicating that all possible ASDs will be downloaded. If you want to 
decrease this, make sure that there are enough ASDs in the training set to represent 
any possible data the dingo network will see. Typically this should be at least 1000,
but of course more is better. 


Step 3 Training the network
---------------------------

Now we are ready for training. The command is analogous to the previous tutorial 
but the settings are increased to production values. To run the training do

```
dingo_train --settings_file train_settings.yaml --train_dir training
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

Notice, we are not inferring the phase parameter here as it is not listed below `inference_parameters`. However,
we do recover the phase in post processing. To see why and how this is done see [synthetic phase](result.md)

Also notice there are now two training stages `stage_0` and `stage_1`. In `stage_0`, a fixed ASD is used and the reduced basis layer
is frozen. Then in `stage_1`, all ASDs are used and the reduced basis layer is unfrozen. 

The main difference in the local settings is that the `device` is set to `CUDA`.

```{important}
It is recommended to have at least 40 GB of GPU memory on the device. If there is not enough memory on the machine,
first try halving the `batch_size`. In this case one should also multiply the learning rate, `lr`, by $\frac{1}{\sqrt{2}}$. If 
there is still not enough memory, consider reducing the number of hidden dimensions.
```

Step 4 Doing Inference
----------------------

We can run inference with the same command as before

```
dingo_pipe GW150914.ini
```

There is just one difference from the previous example. It is possible to reweight the posterior to a new prior.
Note though, that the new prior must be a subset of the previous prior. Otherwise, the proposal distribution
generated by dingo will include regions from the new prior where the network has not been trained which will
result in a low effective sample size and lead to poor results. As an example see the `prior-dict` attribute in 
`GW150914.ini`.