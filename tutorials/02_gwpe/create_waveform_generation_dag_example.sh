#!/bin/bash

D=/home/mpuer/projects/dingo-devel/env/bin

$D/create_waveform_generation_dag \
  --waveforms_directory ./datasets/waveforms/ \
  --parameters_file_basis parameters_basis.npy \
  --parameters_file_dataset parameters_dataset.npy \
  --basis_file polarization_basis.npy \
  --settings_file settings.yaml \
  --dataset_file waveform_dataset.hdf5 \
  --num_wfs_basis 5000 \
  --num_wfs_dataset 10000 \
  --num_wf_per_process 500 \
  --rb_max 500 \
  --env_path /home/mpuer/projects/dingo-devel/env

