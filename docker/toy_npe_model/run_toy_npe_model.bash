#!/bin/bash

# This executable runs all the steps of the dingo toy npe model as described here:
# https://dingo-gw.readthedocs.io/en/latest/example_toy_npe_model.html

# all functions used in this script are imported from dingo.bash
source /opt/dingo/dingo.bash

# from which git repo dingo will be cloned
DINGO_REPO="https://github.com/dingo-gw/dingo.git"

# where all the files will be located (cloned source code, python venv,
# output files of the script, etc)
# Note: if this is running in docker, it is assumed /data is shared with
# the host machine.
BASE_DIR="/data/dingo"

# we reuse the same venv over all the run
VENV_DIR="${BASE_DIR}/venv"

# for this specific run, we will use a folder based on the current date and time
RUN_DIR=$(create_date_folder ${BASE_DIR})

# where the code will be cloned and installed
INSTALL_DIR=${RUN_DIR}/install

# where the output files will be created
OUTPUT_DIR=${RUN_DIR}/output

# we run toy_npe_model, i.e. dingo/examples/toy_npe_model
DINGO_EXAMPLE_FOLDER="toy_npe_model"


# summary
echo "base directory: ${BASE_DIR}"
echo "virtual environment: ${VENV_DIR}"
echo "run directory: ${RUN_DIR}"
echo "dingo example: ${DINGO_EXAMPLE_FOLDER}"

# checking the GPUs are detected
# by pytorch.
# (this just print related info, it does not exit with
# error if no GPU is detected)
check_gpus_detection

# clone and pip install dingo 
install_dingo $DINGO_REPO $INSTALL_DIR $VENV_DIR

# copy the example files tp $OUTPUT_DIR
# and create the training_data and training directories
setup_directory ${INSTALL_DIR}/examples/${DINGO_EXAMPLE_FOLDER} ${OUTPUT_DIR}

# running steps.
# If the user passed the '-y' argument, all steps are run in a row.
# Otherwise, before each step, the script pause and prompt the user

# Note: if any step fail, the script exits with error code; and prints
# an error message to stderr.

if confirm_step "generating waveform dataset"; then
    echo "-- Generating waveform dataset --"
    print_output dingo_generate_dataset dingo_generate_dataset --settings waveform_dataset_settings.yaml --out_file training_data/waveform_dataset.hdf5 || error_handler "dingo_generate_dataset"
fi

if confirm_step "generating ASD dataset"; then
    echo "-- Generating ASD dataset --"
    print_output dingo_generate_asd_dataset dingo_generate_asd_dataset --settings_file asd_dataset_settings.yaml --data_dir training_data/asd_dataset || error_handler "dingo_generate_asd_dataset"
fi
    
if confirm_step "training"; then
    echo "-- Training --"
    print_output dingo_train dingo_train --settings_file train_settings.yaml --train_dir training || error_handler "dingo_train"
fi
    
if confirm_step "performing inference"; then
    echo "-- Performing inference --"
    print_output dingo_pipe dingo_pipe GW150914.ini || error_handler "dingo_pipe"
    echo "Results can be found in ${TOY_FOLDER}/outdir_GW150914"
fi
    
# Exit with success
exit 0
