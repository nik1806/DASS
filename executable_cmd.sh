#!/bin/bash

source /home/paliwal/miniconda3/bin/activate 
source activate dass

export MPLCONFIGDIR=/home/paliwal/DASS/cache
export WANDB_CONFIG_DIR=/home/paliwal/DASS/cache
export WANDB_RUN_DIR=/home/paliwal/DASS/cache
export WANDB_CACHE_DIR=/home/paliwal/DASS/cache
export XDG_CACHE_HOME=/home/paliwal/DASS/cache
export CONDARC=/home/paliwal/DASS/cache
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

python run.py
