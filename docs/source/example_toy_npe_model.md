# Toy Example

The goal of the following tutorial is to take a user from start to finish analyzing GW150914 using dingo.

```{caution}
This is only a toy example which is useful for testing on a local machine. This
is NOT meant be used for production gravitational wave analyses.
```

There are 4 main steps:

1. Generate the waveform dataset
2. Generate the ASD dataset
3. Train the network
4. Do inference

In this tutorial as well as the [npe model](example_npe_model) and [gnpe model](example_gnpe_model) the following file structure will
be employed

```
toy_npe_model/

    #  config files
    waveform_dataset_settings.yaml
    asd_dataset_settings.yaml
    train_settings.yaml
    GW150914.ini

    training_data/
        waveform_dataset.hdf5
        asd_dataset.hdf5
        tmp/    #  Contains temporary files from ASD dataset generation

    training/
        model_050.pt
        model_stage_0.pt
        model_latest.pt
        history.txt
        #  etc...

    outdir_GW150914/
        #  dingo_pipe output
```

The config files which are the only ones which need to be edited are contained in the top level directory. In the next
few sections these config files will be explained. To download sample config files, please visit 
https://github.com/dingo-gw/dingo/tree/main/examples. In this tutorial the toy_npe_model folder will be used.


Step 1 Generating a waveform dataset
------------------------------------

After downloading the files for the tutorial run 

```
cd toy_npe_model/
export BASE_DIR=$(pwd)
dingo_generate_dataset --settings waveform_dataset_settings.yaml --out_file training_data/waveform_dataset.hdf5
```

This will first change directories into the tutorial directory and then create a
{py:class}`dingo.gw.waveform_generator.waveform_generator.WaveformGenerator`
object and store it at the location provided with `--out_file`.

The file `examples/toy_npe_model/waveform_dataset_settings.yaml` contains four
sections: `domain`, `waveform_generator`, `intrinsic_prior`, and `compression`. The
domain section defines the settings for storing the waveform. Note the `type`
attribute; this does not refer to the native domain of the waveform model, but
rather to the internal {py:class}`dingo.gw.domains.Domain` class. This allows the use
of time domain waveform models, which are transformed into Fourier domain before
being passed to the network. Currently, only
the {py:class}`dingo.gw.domains.FrequencyDomain` class is supported for training the
network. It is common to generate waveforms with `f_max`=2048 Hz and then later truncate them at
`f_max`=1024 Hz during training. This is recommended, as some combinations of priors
and waveform models may generate errors due to the ringdown frequency being too large
and unsupported by LALSimulation when `f_max=1024`. If this occurs, it is
advisable to increase `f_max` to 2048 Hz.

The `waveform_generator` section specifies the `approximant` attribute.
At present any waveform model, aka `approximant`, that is callable through LALSimulation's
`SimInspiralFD` API can be used to generate waveforms for dingo via the
{py:class}`dingo.gw.waveform_generator.waveform_generator.WaveformGenerator` module (see
[generating_waveforms](generating_waveforms.md)).

The `intrinsic_prior` section is based on Bilby's prior module.
Default values can be found in `dingo.gw.prior`.
Two priors to note are the `chirp_mass` and `mass_ratio`, whose minimum values are set
to 15.0 and 0.125, respectively. Extending these priors towards lower chirp masses
or more extreme mass-ratios may lead to poor performance of the embedding network
during training and would require changes to the network setup.
Note that the `luminosity_distance` and `geocent_time` are defined as constants
to generate the waveform at a fixed reference point.

The compression section can be set to None for testing purposes. For a practical
example of how it is used, see the next tutorial.

Step 2 Generating the Amplitude Spectral Density (ASD) dataset
--------------------------------------------------------------

To generate an ASD dataset run

```
dingo_generate_asd_dataset --settings_file examples/toy_npe_model/asd_dataset_settings.yaml --data_dir $TRAIN_DIR/asd_dataset_folder
```

This command will generate an {py:class}`dingo.gw.noise.asd_dataset.ASDDataset` object in the form of an .hdf5 file, which will be used later for training. The reason for specifying a folder instead of a file, as in the waveform dataset example, is because some temporary data is downloaded to create Welch estimates of the ASD. This data can be removed later, but it is sometimes useful for understanding how the ASDs were estimated.

The `examples/toy_npe_model/asd_dataset_settings.yaml` file includes several attributes. `f_s` is the sampling frequency in Hz, `time_psd` is the length of time used for an ASD estimate, and `T` is the duration of each ASD segment. Thus, the value of `time_psd`/`T` gives the number of segments analyzed to estimate one ASD. To avoid spectral leakage, a window is applied to each segment. We use the standard window used in LVK analyses, a Tukey window with a roll off of $\alpha=0.4$. The next attribute, `num_psds_max=1`, defines the number of ASDs stored in the ASD dataset. For now, we will use only one. See the next [tutorial](example_npe_model.md) for a more advanced setup.

Step 3 Training the network
---------------------------

To train the network run

```
sed -i 's+/path/to/waveform_dataset.hdf5+'"$TRAIN_DIR"'/waveform_dataset.hdf5+g' examples/toy_npe_model/train_settings.yaml
sed -i 's+/path/to/asd_dataset.hdf5+'"$TRAIN_DIR"'/asd_dataset_folder/asds_O1.hdf5+g' examples/toy_npe_model/train_settings.yaml
dingo_train --settings_file examples/toy_npe_model/train_settings.yaml --train_dir $TRAIN_DIR
```

The two `sed` commands just replace the parts in the `train_settings.yaml` file with the datasets generated in the previous steps. While this file contains numerous settings that are discussed in [training](training.md), we will cover the most significant ones here. For training, several `extrinsic_priors` are set, which project the waveforms generated in step 1 onto the detector network according to the specified priors. This is considerably cheaper than generating waveforms sampled from the full intrinsic plus extrinsic prior in step 1.

Another crucial setting is `inference_parameters`. By default all the parameters described in `dingo.gw.prior` are inferred. If a parameter needs to be marginalized over this parameter can be omitted from `inference_parameters`.

Essential settings for the model architecture (the neural spline flow and the embedding network) are as follows: `nsf_kwargs.num_flow_steps` describes the number of flow transforms from the base distribution to the final distribution, while `embedding_net_kwargs.hidden_dim` defines the dimensions of the neural network's hidden layer, which selects the most important data features. Finally, `embedding_net_kwargs.svd` describes the settings of the SVD used as a pre-processing step before passing data vectors to the embedding network. For a production network, these values should be much higher than those used in this tutorial.

Next, we turn to the training section. Here we only employ a single stage of training with settings provided under the `stage_0` attribute. This stage uses the training dataset generated in step 1 for 30 epochs. We also specify the `asd_dataset_path` here, which was created in step 2.

Finally, the local settings section affects only parallelization during training and the device used. An important setting here is `num_workers`, which determines how many PyTorch dataloader processes are spawned during training. If training is too slow, a potential cause is a lack of workers to load data into the network. This can be identified if the dataloader times in the `dingo_train` output exceed 100ms. The solution is generally to increase the number of workers.

Step 4 Doing Inference
----------------------

The final step is to do inference, for example on GW150914. To do this we will use
[dingo_pipe](dingo_pipe.md). For a local run execute:

```
sed -i "s|TRAIN_DIR/|$TRAIN_DIR/|g" examples/toy_npe_model/GW150914_toy.ini
sed -i "s|/path/to/model.pt|$TRAIN_DIR/model_latest.pt|g" examples/toy_npe_model/GW150914_toy.ini
dingo_pipe examples/toy_npe_model/GW150914.ini
```

This will generate files which are described in [dingo_pipe](dingo_pipe.md). To see the results, take a look in `outdir_GW150914`.

We can load and manipulate the data with the following code. For example, here we create a cornerplot

```
from dingo.gw.result import Result
result = Result(file_name="$TRAIN_DIR/outdir_GW150914/result/GW150914_data0_1126259462-4_importance_sampling.hdf5")
result.plot_corner()
```

Notice the results don't look very promising, but this is expected as the settings used in this
example are not enough to warrant convergence.
