#!/bin/bash

cd /home/mpuer/projects/dingo-devel/
#source dingo-devenv/bin/activate
cd tutorials/02_gwpe

PYTHON=/.auto/home/mpuer/projects/dingo-devel/dingo-devenv/bin/python

$PYTHON create_waveform_generation_dag.py \
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
  --python_binary $PYTHON

