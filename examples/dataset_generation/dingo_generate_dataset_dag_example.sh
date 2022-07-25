#!/bin/bash

D=/work/sgreen/dingo-devel/venv/bin

$D/dingo_generate_dataset_dag \
  --num_jobs 5 \
  --request_cpus 4 \
  --settings_file /work/sgreen/dingo-devel/tutorials/02_gwpe/datasets/waveforms/settings.yaml \
  --out_file waveform_dataset.hdf5 \
  --env_path /work/sgreen/dingo-devel/venv

