#!/bin/bash

out_function(){
    echo "out function"
}


trap 'out_function' ERR
set -e

function1(){
    a=$(pwd)
    echo $a
}

function2(){
    echo "f1 in"
    ravioli
    echo "f1 out"
}

function3(){
    v=$(df -h)
    echo $v
}

run_job(){
    function1
    function2
    function3
}

main() {
    run_job
    out_function
}

main "$@"
