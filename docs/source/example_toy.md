# Toy Example

The goal of the following tutorial is take a user from start to finish analyzing GW150914 using dingo.

```{caution}
This is only a toy example which is useful for testing on a local machine. This
is NOT meant be used for production gravitational wave analyses.  
```

There are 4 main steps. 

1). Generate the waveform dataset
2). Generate the ASD dataset
3). Train the network
4). Do inference 

Step 1 Generating a waveform dataset
------------------------------------

First make a directory for this example where we will store all of our files from this tutorial

```
cd dingo
mkdir $(pwd)/toy_example_train_dir
export TRAIN_DIR=$(pwd)/toy_example_train_dir
```

To generate a waveform dataset run 

```
dingo_generate_dataset --settings examples/toy_example/waveform_dataset_settings.yaml --out_file $TRAIN_DIR/waveform_dataset.hdf5
```

{py:class}`dingo.gw.waveform_generator.waveform_generator.WaveformGenerator` object.

The file examples/toy_example/waveform_dataset_settings.yaml contains four
sections: domain, waveform_generator, intrinsic_prior, and compression. The
domain section defines the settings for storing the waveform. Note the type
attribute; this does not refer to the native domain of the waveform model, but
rather to the internal {py:class}`dingo.gw.domains.Domain` class. This allows the use
of time domain waveform models, which are transformed into Fourier domain before
being passed to the network. Currently, only
the {py:class}`dingo.gw.domains.FrequencyDomain` class is supported for training the
network. The `f_max`, `f_min`, and `delta_f` attributes are also included. It is
common to generate waveforms with `f_max=2048`. Hz and then later truncate them at
`f_max=1024` Hz during training. This is recommended, as many prior combinations
and waveforms may generate errors due to the ringdown frequency being too large
and unsupported by LALSimulation when `f_max=1024`. If this occurs, it is
advisable to increase `f_max` to 2048 Hz.

The waveform_generator section specifies the approximant attribute. If you would
like to implement your own waveform model, it should work in dingo if it works
in LALSimulation. However, it is best to first generate the waveforms using the
dingo.gw.waveform_generator.waveform_generator module (see
[generating_waveforms]generating_waveforms.

The `intrinsic_prior` section is based on Bilby's prior module, with many values
set to their default values. These values can be found in :mod:dingo.gw.prior.
Two priors to note are the `chirp_mass` and `mass_ratio`, whose minimum values are set
to 15.0 and 0.125, respectively. Lower values tend to lead to poor performance of
the embedding network during training. Therefore, it is best to stick to this
range. Note that the `luminosity_distance` and `geocent_time` are defined as single
numbers to generate the waveform at a fixed reference point.

The compression section can be set to None for testing purposes. For a practical
example of how it is used, see the next tutorial.


Step 2 Generating the Amplitude Spectral Density (ASD) dataset
--------------------------------------------------------------

To generate an ASDdataset run 

```
dingo_generate_asd_dataset --settings_file examples/toy_example/asd_dataset_settings.yaml --data_dir $TRAIN_DIR/asd_dataset_folder
```

Running this command will generate an {py:class}`dingo.gw.noise.asd_dataset.ASDDataset` object in the form of an .hdf5 file, which will be used later for training. The reason for specifying a folder instead of a file, as in the waveform dataset example, is because some temporary data is downloaded to create Welch estimates of the ASD. This data can be removed later, but it is sometimes useful for understanding how the ASDs were estimated.

The `examples/toy_example/asd_dataset_settings.yaml` file includes several attributes. `f_s` is the sampling frequency in Hz, `time_psd` is the length of time used for an ASD estimate, and `T` is the duration of each ASD segment. The value of `time_psd/T` gives the number of segments analyzed to estimate one ASD. To avoid spectral leakage, a window is applied to each segment. We use the standard window used in LVK analyses, tukey with a roll off of $\alpha=0.4$. The next attribute, num_psds_max=1, defines the number of ASDs stored in the ASD dataset. For now, we will use only one. See the next tutorial for a more advanced setup.


Step 3 Training the network
---------------------------

To train the network run

```
sed -i 's+/path/to/waveform_dataset.hdf5+'"$TRAIN_DIR"'/waveform_dataset.hdf5+g' examples/toy_example/train_settings.yaml
sed -i 's+/path/to/asd_dataset.hdf5+'"$TRAIN_DIR"'/asd_dataset_folder/asds_O1.hdf5+g' examples/toy_example/train_settings.yaml
dingo_train --settings_file examples/toy_example/train_settings.yaml --train_dir $TRAIN_DIR
```

The first two commands just replace the parts in the `train_settings.yaml` file with the datasets generated in the previous steps. While this file contains numerous settings that can be found in [training](training), we will cover the most significant ones here. During training, several extrinsic_priors` are set, which project the waveforms generated in step 1 according to the specified priors. This effectively increases the size of the training set without the need to generate additional waveforms.

Another crucial setting is `inference_parameters`, which is set to default and infers all the parameters described in :mod:`dingo.gw.prior`. However, if a parameter needs to be marginalized over, inference_parameters can be easily specified to include all parameters except for the ones to be marginalized.

Under the model attribute, several other essential settings are visible, including `nsf_kwargs.num_flow_steps`, `embedding_net_kwargs.hidden_dim`, and `embedding_net_kwargs.svd`. `nsf_kwargs.num_flow_steps` describes the number of flow transforms from the base distribution to the final distribution, while `embedding_net_kwargs.hidden_dim` defines the dimensions of the neural network's hidden layer, which selects the most important data features. Finally, `embedding_net_kwargs.svd` describes the settings of the SVD used as a pre-processing step before passing data vectors to the embedding network. In general, these values should be much higher for a real network.

Next, focus on the training section. The `stage_0` attribute describes the first stage of the training process, which uses the training dataset generated in step 1 for 30 epochs. We also specify the asd_dataset_path here, which was created in step 2.

Finally, the local settings section affects only the speed of training and the devices used. An important setting here is `num_workers`, which determines how many PyTorch dataloader processes are spawned during training. If training is too slow, a potential cause is a lack of workers to load data into the network. This can be identified if the dataloader times in the dingo_train output exceed 100ms. The solution is generally to increase the number of workers.



Step 4 Doing Inference
----------------------

The final step is to do inference, for example on GW150914. To do this we will use 
[dingo_pipe]dingo_pipe. To run dingo pipe locally run:

```
sed -i "s|TRAIN_DIR/|$TRAIN_DIR/|g" examples/toy_example/GW150914_toy.ini
sed -i "s|/path/to/model.pt|$TRAIN_DIR/model_latest.pt|g" examples/toy_example/GW150914_toy.ini
dingo_pipe examples/toy_example/GW150914_toy.ini
```

This will generate files which are described in [dingo_pipe]dingo_pipe. To see the results, take a look in $TRAIN_DIR/outdir_GW150914.

We can load and manipulate the data with the following code. For example, here we create a cornerplot 

```
from dingo.gw.result import Result
result = Result(file_name="$TRAIN_DIR/outdir_GW150914/result/GW150914_data0_1126259462-4_importance_sampling.hdf5")
result.plot_corner()
```
Notice the results don't look very promising, but this is expected as the settings used in this 
example are not enough to warrant convergence. 