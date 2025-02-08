#!/bin/bash

source ./setup_variables.bash

log_message "Checking GPU availability..."

# Activate the virtual environment
if ! source "$VENV_DIR/bin/activate"; then
    die "Failed to activate virtual environment at $VENV_DIR" 2
fi

# Check GPU availability using Python
python - <<EOF
import torch

def check_gpus():
    if torch.cuda.is_available():
        print("CUDA is available")
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPU(s)")
        for i in range(num_gpus):
            print(f"\nGPU {i}:")
            print(f"  Device Name: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
    else:
        print("No CUDA-capable devices found")

check_gpus()
EOF
