#!/bin/bash

# Function to print error message and exit
die() {
    local msg="$1"
    local code="${2:-1}"
    echo "ERROR: $msg" >&2
    exit "$code"
}

# Check if RUN_DIR is provided as a positional argument
if [ $# -ne 1 ]; then
    die "Usage: $0 <RUN_DIR>" 1
fi

# Set RUN_DIR from the first positional argument
RUN_DIR="$1"
INSTALL_DIR="${RUN_DIR}/install"
VENV_DIR="${RUN_DIR}/../venv"
LOG_FILE="${RUN_DIR}/log.txt"
ERROR_FILE="${RUN_DIR}/error.txt"
SUMMARY_FILE="${RUN_DIR}/summary.txt"


log_message "Generating job summary..."

# Collect information
local branch=$(cd "$install_dir" && git branch --show-current)

local commit=$(cd "$install_dir" && git rev-parse HEAD)

local python_version=$(source "$venv_dir/bin/activate" && python --version)

local end_time=$(date +"%Y-%m-%d_%H-%M-%S")

# Write summary
cat > "$SUMMARY_FILE" <<EOF
Job Summary
===========
Branch: $branch
Commit: $commit
Python Version: $python_version
End Time: $end_time
EOF

