#!/bin/bash

# Path to a virtual environment in which you have installed dingo
ENV_DIR=./dingo-devel/venv
# Path to the directory which contains settings.yaml
# This is where we will save the waveform data to.
OUTPUT_DIR=./datasets/waveforms/

$ENV_DIR/bin/create_waveform_generation_bash_script \
--waveforms_directory $OUTPUT_DIR \
--env_path $ENV_DIR \
--num_threads 16

