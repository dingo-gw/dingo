#!/bin/bash

# This executable runs all the steps of the dingo toy npe model as described here:
# https://dingo-gw.readthedocs.io/en/latest/example_toy_npe_model.html

# Configuration
DINGO_REPO="https://github.com/dingo-gw/dingo.git"
DEFAULT_BASE_DIR="/tmp/dingo"

parse_arguments() {
    local base_dir="$DEFAULT_BASE_DIR"
    local email_config=""
    local test_email="false"

    while [[ $# -gt 0 ]]; do
        case $1 in
            --base-dir)
                if [[ $# -lt 2 ]]; then
                    echo "Error: --base-dir requires a directory path" >&2
                    return 1
                fi
                base_dir="$2"
                shift 2
                ;;
            --email-config)
                if [[ $# -lt 2 ]]; then
                    echo "Error: --email-config requires a JSON file name" >&2
                    return 1
                fi
                if [[ ! -f "$2" ]]; then
                    echo "Error: Email configuration file '$2' not found" >&2
                    return 1
                fi
                email_config=$(jq '.' "$2")
                if [[ $? -ne 0 ]]; then
                    echo "Error: Invalid JSON format in configuration file '$2'" >&2
                    return 1
                fi
                shift 2
                ;;
            --test-email)
                test_email="true"
                shift
                ;;
            *)
                echo "Error: Unknown option: $1" >&2
                return 1
                ;;
        esac
    done

    # Validate base directory
    if [[ ! "$base_dir" =~ ^/ ]]; then
        echo "Error: Base directory must be an absolute path" >&2
        return 1
    fi

    # Output variables in a safe format
    echo "BASE_DIR=\"$base_dir\""
    echo "EMAIL_CONFIG='$(echo "$email_config" | jq -c .)'"
    echo "TEST_EMAIL=\"$test_email\""
}


# Parse arguments
parsed_args=$(parse_arguments "$@")
if [[ $? -ne 0 ]]; then
    exit 1
fi

# Set variables from parsed arguments
# $BASE_DIR and
# $EMAIL_CONFIG
# $TEST_EMAIL
while IFS= read -r line; do
    eval "$line"
done <<< "$parsed_args"

# The virtual env. Will be reused accross jobs
VENV_DIR="${BASE_DIR}/venv"

# Create base directory if it doesn't exist
mkdir -p "$BASE_DIR"

# Get current timestamp for unique run directory
CURRENT_DATETIME="$(date +"%Y-%m-%d_%H-%M-%S")"
RUN_DIR="${BASE_DIR}/${CURRENT_DATETIME}"
mkdir -p "$RUN_DIR"

# Directory structure
INSTALL_DIR="${RUN_DIR}/install"
OUTPUT_DIR="${RUN_DIR}/output"

# Log file paths
LOG_FILE="${RUN_DIR}/log.txt"
ERROR_FILE="${RUN_DIR}/error.txt"
SUMMARY_FILE="${RUN_DIR}/summary.txt"

# Initialize logging with separate streams
exec > >(tee -a "$LOG_FILE") 2> >(tee -a "$ERROR_FILE")

log_message() {
    local message="$1"
    echo -e "[$(date +"%Y-%m-%d %H:%M:%S")] $message"
}

error_handler() {
    local cmd_name="$1"
    local error_message="$2"
    local error="Error: ${cmd_name} failed with message: ${error_message}"
    log_message "$error"
    echo -e "$error" >&2  # Send to stderr
    exit 1
}

install_dingo() {
    local dingo_repo="$1"
    local dingo_install="$2"
    local dingo_venv="$3"

    log_message "Starting DINGO installation..."

    # Handle virtual environment
    if [ -d "$dingo_venv" ]; then
        log_message "Using existing virtual environment: $dingo_venv"
        log_message "Python version: $(source "$dingo_venv/bin/activate" && python --version)"
    else
        log_message "Creating new virtual environment: $dingo_venv"
        python3 -m venv "$dingo_venv"
        local exit_code=$?
        if [ $exit_code -ne 0 ]; then
            error_handler "create_venv" "Failed to create virtual environment"
        fi
    fi

    # Clone repository
    rm -rf "$dingo_install"
    mkdir -p "$dingo_install"
    log_message "Cloning repository from $dingo_repo to $dingo_install"
    git clone "$dingo_repo" "$dingo_install"
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        error_handler "git_clone" "Failed to clone repository"
    fi

    # Install DINGO
    (
        source "$dingo_venv/bin/activate"
        cd "$dingo_install"
        log_message "Installing DINGO dependencies..."
        pip install .
        local exit_code=$?
        if [ $exit_code -ne 0 ]; then
            error_handler "pip_install" "Failed to install DINGO"
        fi
    )
}

check_gpu_detection() {
    local venv_dir="$1"
    log_message "Checking GPU availability..."
    
    source "$venv_dir/bin/activate"
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
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        error_handler "check_gpus" "Failed to check GPU availability"
    fi
}

setup_directory() {
    local src_dir="$1"
    local target_dir="$2"
    
    log_message "Setting up directory structure..."
    mkdir -p "$target_dir"
    
    if [ -d "$src_dir" ]; then
        log_message "Copying files from $src_dir to $target_dir"
        cp -r "$src_dir"/* "$target_dir/"
        local exit_code=$?
        if [ $exit_code -ne 0 ]; then
            error_handler "copy_files" "Failed to copy files"
        fi
        
        log_message "Creating required subdirectories..."
        mkdir -p "$target_dir/training_data"
        mkdir -p "$target_dir/training"
        local exit_code=$?
        if [ $exit_code -ne 0 ]; then
            error_handler "create_dirs" "Failed to create directories"
        fi
    else
        error_handler "setup_directory" "Source directory not found: $src_dir"
    fi
}

generate_job_summary() {
    local example="$1"
    local install_dir="$2"
    local run_dir="$3"
    local venv_dir="$4"

    log_message "Generating job summary..."
    
    # Collect information
    local branch=$(cd "$install_dir" && git branch --show-current)
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        error_handler "get_branch" "Failed to get git branch"
    fi

    local commit=$(cd "$install_dir" && git rev-parse HEAD)
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        error_handler "get_commit" "Failed to get git commit"
    fi

    local python_version=$(source "$venv_dir/bin/activate" && python --version)
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        error_handler "get_python_version" "Failed to get Python version"
    fi

    local end_time=$(date +"%Y-%m-%d_%H-%M-%S")

    # Write summary
    cat > "$SUMMARY_FILE" <<EOF
Job Summary
===========
Example: $example
Branch: $branch
Commit: $commit
Python Version: $python_version
End Time: $end_time
EOF
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        error_handler "write_summary" "Failed to write summary file"
    fi
}


run_job() {
    local dingo_repo="$1"
    local venv_dir="$2"
    local install_dir="$3"
    local output_dir="$4"

    echo -e "\n\nRAVIOLI BANZAI !\n\n"
    ravioli !
    
    log_message "Starting job execution..."

    # Step 1: Install DINGO
    log_message "\n================= Installing DINGO ================="
    install_dingo "$dingo_repo" "$install_dir" "$venv_dir"
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        error_handler "install_dingo" "DINGO installation failed"
    fi
    log_message "==================================================\n"

    # Step 2: Check GPU Detection
    log_message "================= Checking GPU Availability ================="
    check_gpu_detection "$venv_dir"
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        error_handler "check_gpu" "GPU detection failed"
    fi
    log_message "==========================================================\n"

    # Step 3: Setup Directory Structure
    log_message "================= Setting Up Directories ================="
    setup_directory "$install_dir/examples/toy_npe_model" "$output_dir"
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        error_handler "setup_directories" "Directory setup failed"
    fi
    log_message "=========================================================\n"

    # Activate virtual environment for remaining steps
    source "$venv_dir/bin/activate"

    # cd to the output folder
    cd $output_dir
    
    # Step 4: Generate Waveform Dataset
    log_message "================= Generating Waveform Dataset ================="
    dingo_generate_dataset --settings waveform_dataset_settings.yaml --out_file training_data/waveform_dataset.hdf5
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        error_handler "generate_dataset" "Failed to generate waveform dataset"
    fi
    log_message "================================================================\n"

    # Step 5: Generate ASD Dataset
    log_message "================= Generating ASD Dataset ================="
    dingo_generate_asd_dataset --settings_file asd_dataset_settings.yaml --data_dir training_data/asd_dataset
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        error_handler "generate_asd" "Failed to generate ASD dataset"
    fi
    log_message "============================================================\n"

    # Step 6: Train Model
    log_message "================= Training Model ================="
    dingo_train --settings_file train_settings.yaml --train_dir training
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        error_handler "train_model" "Failed to train model"
    fi
    log_message "==============================================\n"

    # Step 7: Run Inference
    log_message "================= Running Inference ================="
    dingo_pipe GW150914.ini
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        error_handler "run_inference" "Failed to run inference"
    fi
    log_message "===================================================\n"

    # Final Step: Generate Summary
    generate_job_summary "toy_npe_model" "$INSTALL_DIR" "$run_dir" "$venv_dir"
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        error_handler "generate_summary" "Failed to generate summary"
    fi
    log_message "Job completed successfully!"
}


send_email_report() {
    local success="$1"
    local email_config="$2"
    local run_dir="$3"
    local install_dir="$4"
    local log_file="$5"
    local error_file="$6"

    # Extract email configuration
    local smtp_server=$(jq -r '.mailhub' <<< "$email_config")
    local port=$(jq -r '.port' <<< "$email_config")
    local from_addr=$(jq -r '.root' <<< "$email_config")
    local auth_user=$(jq -r '.authUser' <<< "$email_config")
    local auth_pass=$(jq -r '.authPass' <<< "$email_config")
    local recipients=$(jq -r '.recipients[]' <<< "$email_config")

    # Set email subject based on success/failure
    local subject="dingo toy-npe-model example: $(if [ "$success" = "true" ]; then echo "SUCCEEDED"; else echo "FAILED"; fi)"

    # Create email body
    local body=$(cat <<EOF
DINGO Workflow Report
====================
Start Time: $(if [[ -f "$log_file" ]]; then head -n 1 "$log_file" | cut -d']' -f1 | cut -c2-; else echo "N/A"; fi)
End Time: $(date +"%Y-%m-%d %H:%M:%S")
Status: $(if [ "$success" = "true" ]; then echo "SUCCESS"; else echo "FAILURE"; fi)
Branch: $(if [[ -d "$install_dir" ]]; then cd "$install_dir" && git branch --show-current; else echo "N/A"; fi)
Commit: $(if [[ -d "$install_dir" ]]; then cd "$install_dir" && git rev-parse HEAD; else echo "N/A"; fi)
EOF
)

    # Prepare attachments
    local attachments=()
    if [[ -f "$log_file" ]]; then
        attachments+=("-a" "$log_file")
    fi
    if [[ -f "$error_file" ]]; then
        attachments+=("-a" "$error_file")
    fi

    # URI the mail will be set to
    smpt_url="smtps://${auth_user}@${smtp_server}:${port}"

    # sending the email
    echo "$body" | mutt -e "set smtp_url=${smpt_url}" -e "set smtp_pass=${auth_pass}" -e "set smtp_authenticators='login'" -s "${subject}" ${attachments[@]} -- ${recipients}
 
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_message "Warning: Failed to send email to $recipients"
    else
        log_message "Email sent successfully to $recipients"
    fi
}



main() {
    log_message "Starting DINGO workflow..."
    log_message "Base directory: $BASE_DIR"
    log_message "Run directory: $RUN_DIR"
    log_message "Virtual environment: $VENV_DIR\n"

    if [[ "$TEST_EMAIL" = "true" ]]; then
        # Test email functionality
        log_message "Running in test email mode..."
        
        # Create test files if they don't exist
        echo "Test error message" > "$ERROR_FILE"
        echo "Test output message" > "$LOG_FILE"
        echo "Test summary information" > "$SUMMARY_FILE"
        
        # Send test email
        send_email_report "true" "$EMAIL_CONFIG" "$RUN_DIR" "$INSTALL_DIR" "$LOG_FILE" "$ERROR_FILE"
        exit 0
    fi
    
    # Run the job and capture success/failure
    if run_job "$DINGO_REPO" "$VENV_DIR" "$INSTALL_DIR" "$OUTPUT_DIR"; then
	echo -e "\n\nSUCCESS\n\n"
        success="true"
    else
	echo -e "\n\nFAILED\n\n"
        success="false"
    fi

    echo ""
    echo "$EMAIL_CONFIG"
    echo ""
    
    # Send email report if configured
    if [[ -n "$EMAIL_CONFIG" ]]; then
	echo -e "\n\nsending email\n\n"
        send_email_report "$success" "$EMAIL_CONFIG" "$RUN_DIR" "$INSTALL_DIR" "$LOG_FILE" "$ERROR_FILE"
    fi
}

main "$@"

