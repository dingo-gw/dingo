#!/bin/bash


# Check if RUN_DIR is provided as a positional argument
if [ $# -ne 1 ]; then
    die "Usage: $0 <RUN_DIR>" 1
fi

source ./setup_variables.bash

# Activate virtual environment for remaining steps
if ! source "$VENV_DIR/bin/activate"; then
    die "Failed to activate virtual environment at $VENV_DIR" 2
fi

# Change directory to the output folder
if ! cd "$OUTPUT_DIR"; then
    die "Failed to change directory to $OUTPUT_DIR" 3
fi

log_message "================= Generating Waveform Dataset ================="
if ! dingo_generate_dataset --settings waveform_dataset_settings.yaml --out_file training_data/waveform_dataset.hdf5; then
    die "Failed to generate waveform dataset" 4
fi
log_message "================================================================\n"

log_message "================= Generating ASD Dataset ================="
if ! dingo_generate_asd_dataset --settings_file asd_dataset_settings.yaml --data_dir training_data/asd_dataset; then
    die "Failed to generate ASD dataset" 5
fi
log_message "============================================================\n"

log_message "================= Training Model ================="
if ! dingo_train --settings_file train_settings.yaml --train_dir training; then
    die "Failed to train model" 6
fi
log_message "==============================================\n"

log_message "================= Running Inference ================="
if ! dingo_pipe GW150914.ini; then
    die "Failed to run inference" 7
fi
log_message "===================================================\n"
