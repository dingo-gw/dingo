Quickstart Tutorial
===================

The quickest way to get started with Dingo is to follow the examples in the repository.

Running Your First Injection
----------------------------

A general pipeline to using dingo for inference on injections is to 

1. Generate a :class:`~dingo.gw.dataset.waveform_dataset.WaveformDataset` 
2. Generate a :class:`~dingo.gw.ASD_dataset.noise_dataset.ASDDataset`
3. Generate and train a :class:`~dingo.core.models.posterior_model.PosteriorModel` using the :class:`~dingo.gw.dataset.waveform_dataset.WaveformDataset`  and :class:`~dingo.gw.ASD_dataset.noise_dataset.ASDDataset` 
4. Generate a :class:`~dingo.gw.inference.gw_samplers.GWSampler` using the trained :class:`~dingo.core.models.posterior_model.PosteriorModel` to do inference on a :class:`~dingo.gw.inference.injection.Injection`


This tutorial will take you through how to start with various settings files and go through steps 1-4. At the end you will be able to generate a corner plot of an injection using dingo!


Step 1, Generating a :class:`~dingo.gw.dataset.waveform_dataset.WaveformDataset` 
------------------------------------

Generating a :class:`~dingo.gw.dataset.waveform_dataset.WaveformDataset` is largely done with the use of a a `settings.yaml` file. You can edit this file to change the 
priors, waveform approximant, f_max etc. Here is a sample settings.yaml file. 


.. literalinclude:: ../../tutorials/02_gwpe/datasets/waveforms/settings.yaml
   :language: yaml



Dingo's functionality is largely wrapped around the :class:`~dingo.core.models.posterior_model.PosteriorModel` class. This is the class which 