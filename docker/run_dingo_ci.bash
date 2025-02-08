#!/bin/bash

# Configuration
DEFAULT_BASE_DIR="/tmp/dingo"

# Setting base_dir: either default, or value of -base-dir optional argument
BASE_DIR="$DEFAULT_BASE_DIR"
while getopts ":b:" opt; do
  case $opt in
    b)
      BASE_DIR="$OPTARG"
      ;;
    \?)
      die "Invalid option: -$OPTARG" 1
      ;;
    :)
      die "Option -$OPTARG requires an argument" 2
      ;;
  esac
done

RUN_DIR=$(setup_run_directory ${BASE_DIR})

send_email() {
    # Save the exit status immediately
    local status=$?
    
    # Check if EMAIL_CONFIG_PATH is provided
    if [ -n "${EMAIL_CONFIG_PATH}" ]; then
        # Attempt to send the report email
        if ! report_email "${RUN_DIR}" "${status}" "${EMAIL_CONFIG_PATH}"; then
            echo "Error: Failed to send report email" >&2
        fi
    fi
    
    # Exit with the original status
    exit ${status}
}

# Enable error handling
set -e

# Set up the trap
trap send_email EXIT

setup_dingo "${RUN_DIR}"
check_gpu "${RUN_DIR}"
run_toy_npe_model "${RUN_DIR}"

#!/bin/bash

RUN_DIR=$(setup_run_directory)

send_email() {
    local status=$?
    if [ -n "${EMAIL_CONFIG_PATH}" ]; then
        if ! report_email "${RUN_DIR}" "${status}" "${EMAIL_CONFIG_PATH}"; then
            echo "Error: Failed to send report email" >&2
        fi
    fi
    exit ${status}
}

set -e
trap send_email EXIT

setup_dingo ${RUN_DIR}
check_gpu ${RUN_DIR}
run_toy_npe_model ${RUN_DIR}

