#!/bin/bash

# This executable runs all the steps of the dingo toy npe model as described here:
# https://dingo-gw.readthedocs.io/en/latest/example_toy_npe_model.html


# (github) clone dingo, create a python venv
# and (pip) install dingo in the folder
# $DINGO_INSTALL (/tmp/dingo/install, will be first deleted if already exists !),
install_dingo() {
    local dingo_repo=$1
    local dingo_install=$2
    local dingo_venv=$3
    if [ -d "$dingo_venv" ]; then
	echo "using virtual environment: ${dingo_venv}. (python: $(python --version))"
    else
	echo "creating virtual environment: ${dingo_venv}"
	python3 -m venv "$dingo_venv"
    fi
    mkdir -p ${dingo_install}
    echo "cloning ${dingo_repo} to ${dingo_install}"
    git clone ${dingo_repo} ${dingo_install}
    echo "python: $(python3 --version)"
    echo "installing dingo in ${dingo_install}/venv"
    python3 -m venv ${dingo_venv} && \
	. ${dingo_venv}/bin/activate && \
	cd ${dingo_install} && \
	pip install .
}

# check that pytorch (as install in $DINGO_VENV)
# detects the GPU(s)
check_gpus_detection(){
    local venv_dir=$1
    source ${venv_dir}/bin/activate && \
    python - <<END
import torch

def check_gpus():
    if torch.cuda.is_available():
        print("CUDA is available.")
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs detected: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  CUDA Capability: {props.major}.{props.minor}")
    else:
        print("No GPU detected. CUDA is not available.")

check_gpus()
END
}


# Function to set up the directory:
# moving data from dingo repo (examples folder) to /tmp
setup_directory() {
    local src_dir=$1
    local target_dir=$2
    echo "Copying the content of ${src_dir} to ${target_dir}"
    mkdir -p "$target_dir"
    cp -r "$src_dir"/* "$target_dir" 
    cd "$target_dir" && \
    mkdir -p training_data && \
    mkdir -p training 
}


job_summary() {
    example=$1
    install_dir=$2
    run_dir=$3
    venv_dir=$4

    # git related
    branch=$(cd ${install_dir} && git branch --show-current)
    commit=$(cd ${install_dir} && git rev-parse HEAD)

    # python related
    python_version=$(source ${venv_dir}/bin/activate && python --version)

    # date and time
    datetime=$(date +"%Y-%m-%d_%H-%M-%S")

    echo "${example} ran with success"
    echo "branch: ${branch}"
    echo "commit: ${commit}"
    echo "python: ${python_version}"
    echo "job finished: ${datetime}"
}




function run_job() {

    local dingo_repo=$1
    local venv_dir=$1
    local run_dir=$3
    
    set -e
    error_handler() {
	local cmd_name=$1
	local error_message=$2
	local error="Error: ${cmd_name} failed with message: ${error_message}"
	echo ${error}>&2
	echo ${error} > ${run_dir}/error.txt
	exit 1
    }
    
    # where the code will be cloned and installed
    local install_dir=${run_dir}/install
    
    # where the output files will be created
    local output_dir=${run_dir}/output
    
    # where the job summary will be printed
    local job_summary=${run_dir}/summary.txt
    
    # we run toy_npe_model, i.e. dingo/examples/toy_npe_model
    local dingo_example_folder="toy_npe_model"
    
    # summary
    echo ""
    echo "----------------- DIRECTORIES -----------------"
    echo "virtual environment: ${VENV_DIR}"
    echo "run directory: ${run_dir}"
    echo "dingo example: ${dingo_example_folder}"
    echo "-----------------------------------------------"
    echo ""

    # clone and pip install dingo
    echo "----------------- installing dingo ----------------- "
    install_dingo $DINGO_REPO $install_dir $VENV_DIR || error_handler "cloning and installing dingo"
    echo "-----------------------------------------------------"
    echo ""

    # checking the GPUs are detected
    # by pytorch.
    # (this just print related info, it does not exit with
    # error if no GPU is detected)
    echo "----------------- GPU/CUDA ----------------- "
    check_gpus_detection $VENV_DIR || error_handler "checking pytorch GPU access"
    echo "---------------------------------------------"
    echo ""

    # copy the example files tp $output_dir
    # and create the training_data and training directories
    echo "----------------- setting up output directory -----------------"
    setup_directory ${INSTALL_DIR}/examples/${dingo_example_folder} ${output_dir} || error_handler "setup directory"
    echo "---------------------------------------------------------------"
    echo ""

    # running steps.
    # If the user passed the '-y' argument, all steps are run in a row.
    # Otherwise, before each step, the script pause and prompt the user

    # Note: if any step fail, the script exits with error code; and prints
    # an error message to stderr.

    echo "-----------------  Generating waveform dataset ----------------- "
    dingo_generate_dataset dingo_generate_dataset \
			   --settings waveform_dataset_settings.yaml \
			   --out_file training_data/waveform_dataset.hdf5 || error_handler "dingo_generate_dataset"
    echo "-----------------------------------------------------------------"
    echo ""
    
    echo "-----------------  Generating ASD dataset ----------------- "
    dingo_generate_asd_dataset dingo_generate_asd_dataset \
			       --settings_file asd_dataset_settings.yaml \
			       --data_dir training_data/asd_dataset || error_handler "dingo_generate_asd_dataset"
    echo "------------------------------------------------------------"
    echo ""
    
    echo "-----------------  Training ----------------- "
    dingo_train dingo_train \
		--settings_file train_settings.yaml \
		--train_dir training || error_handler "dingo_train"
    echo "----------------------------------------------"
    echo ""
    
    echo "-----------------  Performing inference ----------------- "
    dingo_pipe dingo_pipe GW150914.ini || error_handler "dingo_pipe"
    echo "----------------------------------------------------------"
    echo ""

    echo "----------------- job summary -----------------"
}


# from which git repo dingo will be cloned
DINGO_REPO="https://github.com/dingo-gw/dingo.git"

# where all the files will be located (cloned source code, python venv,
# output files of the script, etc)
# Note: if this is running in docker, it is assumed /data is shared with
# the host machine.
if [ -n "$1" ]; then
  BASE_DIR="$1"
else
  BASE_DIR="/tmp/dingo"
fi

# we reuse the same venv over all the run
VENV_DIR="${BASE_DIR}/venv"

current_datetime=$(date +"%Y-%m-%d_%H-%M-%S")
RUN_DIR="${base_dir}/$current_datetime"
mkdir -p "${RUN_DIR}"

res=$(run_job $DINGO_REPO $VENV_DIR $RUN_DIR > $RUN_DIR/output.txt)


# Exit with success
exit 0
