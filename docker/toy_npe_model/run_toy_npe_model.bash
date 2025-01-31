#!/bin/bash

# This executable runs all the steps of the dingo toy npe model as described here:
# https://dingo-gw.readthedocs.io/en/latest/example_toy_npe_model.html

source /opt/dingo/docker/dingo.bash


# we run toy_npe_model, i.e. dingo/examples/toy_npe_model
dingo_example_folder="toy_npe_model"


# copying the example files to /tmp 
setup_directory ${dingo_example_folder}


# running steps

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
