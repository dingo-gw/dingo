#!/bin/bash

D=/home/mpuer/projects/dingo-devel

$D/env/bin/create_waveform_generation_bash_script \
--waveforms_directory $D/tutorials/02_gwpe/datasets/waveforms/ \
--env_path $D/env \
--num_threads 16

