#!/bin/bash

# Configuration
DEFAULT_BASE_DIR="/tmp/dingo"

# Function to print error message and exit
die() {
    local msg="$1"
    local code="${2:-1}"
    echo "ERROR: $msg" >&2
    exit "$code"
}

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

# Verify base directory exists or can be created
if ! mkdir -p "$BASE_DIR"; then
    die "Failed to create base directory '$BASE_DIR'" 3
fi

# Verify base directory is writable
if [ ! -w "$BASE_DIR" ]; then
    die "Base directory '$BASE_DIR' is not writable" 4
fi

# Get current timestamp for unique run directory
CURRENT_DATETIME="$(date +"%Y-%m-%d_%H-%M-%S")"
RUN_DIR="${BASE_DIR}/${CURRENT_DATETIME}"

# Create and verify run directory
if ! mkdir -p "$RUN_DIR"; then
    die "Failed to create run directory '$RUN_DIR'" 5
fi

# Output the run directory path
echo "$RUN_DIR"

