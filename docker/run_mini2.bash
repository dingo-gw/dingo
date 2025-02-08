#!/bin/bash

set -e

error_handler() {
    echo "Error caught in script at line $1"
    exit 1
}

trap 'error_handler $LINENO' ERR

function1() {
    echo "f1 in"
    false
    echo "f1 out"
}

function2() {
    echo "f2 in"
    invalid_command
    echo "f2 out"
}

main() {
    function1
    function2
}

main
