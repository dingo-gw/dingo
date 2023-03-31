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

To generate a waveform dataset run 

```
dingo_generate_dataset --settings examples/toy_example/waveform_dataset_settings_toy.yaml --out_file path/to/waveform_dataset.hdf5
```
This will create a .hdf5 file which contains a `:class:`dingo.gw.waveform_generator.waveform_generator.WaveformGenerator object.

There are 4 sections in `examples/toy_example/waveform_dataset_settings.yaml`:
`domain`, `waveform_generator`, `intrinsic_prior` and `compression`. The domain section
defines the settings which to store the waveform. Notice the `type` attribute.
This is not the native domain of the waveform model, but rather refers to the
internal `:class:`dingo.gw.domains.Domain class. This means one can use a time domain
waveform model, it will just be fourier transformed before being passed to the
network. At the moment, we only have support for training a network using the
`:class:`dingo.gw.domains.FrequencyDomain class. There are also attributes for `f_max`,
`f_min` and `delta_f`. Often it is the case that one wants to generate waveforms 
with `f_max=2048.` Hz and then later during training truncate the waveforms at `f_max=1024`. 
This is supported and in fact recommended. In general, many prior combinations and waveforms will
throw errors due to the ringdown frequency being too large and not supported by LALSimulation if
`f_max=1024`. A good rule of thumb to do if this happens is to just increase `f_max=2048`Hz. 


The next section is the waveform generator section. Notice the `approximant` attribute. 
If you would like to implement your own waveform model, generally, if it works in 
LALSimulation it should work in dingo. However, it is best to first do a few sanity checks
by generating the waveforms using the `:class:`dingo.gw.waveform_generator.waveform_generator
module (see `:doc:`generating_waveforms`). 

The next section is the priors. We base our priors on Bilby's prior module. Many values are set to defaults 
which can explicitly be found written in `:mod:`dingo.gw.prior. Two priors to be careful of are the  
`chirp_mass` and `mass_ratio`. Their minimum are set to `15.0` and `0.125` respectively. This
is because lower than this the embedding network during training doesn't work so well. Thus
it is best to stick to this range. It may seem peculiar that the `luminosity_distance` and `geocent_time`
are defined to be single numbers here, but this is just so we can generate the waveform at a 
fixed reference point. 

The final section is `compression`, for testing we can set this to `None`. To see how it is
used in practice see the next tutorial.

Step 2 Generating the Amplitude Spectral Density (ASD) dataset
--------------------------------------------------------------

To generate an ASDdataset run 

```
dingo_generate_asd_dataset --settings_file examples/toy_example/asd_dataset_settings.yaml --data_dir /path/to/asd_dataset_folder
```


This will generate an `:class:`dingo.gw.noise.asd_dataset.ASDDataset object in the form of an .hdf5 file which will later be used for training. The reason 
you have to specify a folder instead of a file like in the waveform dataset example is because there will be some
temporary data downloaded which is used to create Welch estimates of the ASD. One can remove this later, but it is 
sometimes useful for knowing exactly how the ASDs were estimated. There are several attributes in the `examples/toy_example/asd_dataset_settings.yaml`
file. `f_s` is the sampling frequency in Hz, `time_psd` is length of time to use for a ASD estimate, and `T` is the duration of
each ASD segment. Thus `time_psd/T` would give use the number of segments analyzed to estimate one ASD. To avoid spectral leakage
there is also a `window` applied to each segement, we use the standard window used in LVK analyses, tukey with $$\alpha=0.4$$ roll off. 
The next attribute is `num_psds_max=1`. This just defines the number of ASDs stored in the ASD dataset, for now we will just use one, 
see the next tutorial for a more advanced setupl.

Step 2 Training the network
---------------------------

To train the network run

```
dingo_train --settings_file examples/toy_example/train_settings.yaml --train_dir /path/to/train/dir
```

From this default `examples/toy_example/train_settings.yaml` make sure to change the `waveform_dataset_path` and `asd_dataset_path` to 
the files created in step 1 and step 2. We will go step by step and explain the outputs of this command. 

The first step is titled "building SVD for initialization of embedding network"


