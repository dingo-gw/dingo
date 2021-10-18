#!/bin/bash

D=/home/mpuer/projects/dingo-devel

$D/env/bin/create_waveform_generation_bash_script \
--waveforms_directory $D/tutorials/02_gwpe/datasets/waveforms/ \
--parameters_file_basis parameters_basis.npy \
--parameters_file_dataset parameters_dataset.npy \
--basis_file polarization_basis.npy \
--settings_file settings.yaml \
--dataset_file waveform_dataset.hdf5 \
--env_path $D/env \
--num_threads 16

