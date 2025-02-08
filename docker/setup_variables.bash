# Function to print error message and exit
die() {
    local msg="$1"
    local code="${2:-1}"
    echo "ERROR: $msg" >&2
    exit "$code"
}

log_message() {
    local message="$1"
    echo -e "[$(date +"%Y-%m-%d %H:%M:%S")] $message"
}


# Check if RUN_DIR is set
if [[ -z "${RUN_DIR}" ]]; then
    die "Error: RUN_DIR is not set"
fi

# Define directories based on RUN_DIR
VENV_DIR="${RUN_DIR}/../venv"
INSTALL_DIR="${RUN_DIR}/install"
OUTPUT_DIR="${RUN_DIR}/output"

# Log file paths
LOG_FILE="${RUN_DIR}/log.txt"
ERROR_FILE="${RUN_DIR}/error.txt"
SUMMARY_FILE="${RUN_DIR}/summary.txt"

# Initialize logging with separate streams
exec > >(tee -a "$LOG_FILE") 2> >(tee -a "$ERROR_FILE")
