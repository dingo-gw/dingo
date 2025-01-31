#!/bin/bash

# - setup the '-y' option (running all steps without confirmation)
# - setup exit at first error
# - setting the functions:
#   - setup_directory: copying files from dingo examples to /tmp
#   - confirm_step: asking user if the step should be executed or skipped (for resuming)
#   - error_handler: exiting with error after printing to stderr
#   - print_output: run a command and print its output to the terminal

echo "setting up '-y' global option (running all steps without confirmation)"
FORCE_YES=false
while getopts "y" opt; do
  case $opt in
    y)
      FORCE_YES=true
      ;;
    *)
      echo "Usage: $0 [-y]" >&2
      exit 1
      ;;
  esac
done


echo "bash: setting exit at first error ('set -e')"
set -e

# printing to stderr and exiting upon error
error_handler() {
  local cmd_name=$1
  local error_message=$2
  echo "Error: ${cmd_name} failed with message: ${error_message}" >&2
  exit 1
}


# Function to set up the directory:
# moving data from dingo repo (examples folder) to /tmp
setup_directory() {
  local example_folder="/opt/dingo/examples/${1}"
  local target_folder="/tmp/dingo/${1}"
  
  # Starting from scratch
  if [ -d "$target_folder" ]; then
      echo "Directory ${target_folder} already exists, using existing files and folders"
      return
  fi

  # Create a subfolder in /tmp and copying content to it
  echo "Copying the content of ${example_folder} to ${target_folder}"
  mkdir -p "$target_folder" || error_handler "mkdir"
  cp -r "$example_folder"/* "$target_folder" || error_handler "copying files and folder from ${} to ${target_folder}"
  cd "$target_folder" || error_handler "moving to ${target_folder}"
  echo "Working from $(pwd)"
  mkdir -p training_data || error_handler "mkdir"
  mkdir -p training || error_handler "mkdir"
}


# Function to confirm step execution (or for skipping the step)
confirm_step() {
  local step_name=$1
  if [ "$FORCE_YES" = true ]; then
    return 0
  fi
  read -p "Do you want to proceed with ${step_name}? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Skipping ${step_name}."
    return 1
  fi
  return 0
}


# Function to print output with command name, tab and grey color
print_output() {
  local cmd_name=$1
  shift
  local output
  output=$("$@" 2>&1) # Capture both stdout and stderr
  local exit_status=$?
  echo "$output" | sed "s/^/[${cmd_name}]\t/" | while IFS= read -r line; do echo -e "\033[90m$line\033[0m"; done
  if [ $exit_status -ne 0 ]; then
    error_handler "$cmd_name" "$output"
  fi
}
