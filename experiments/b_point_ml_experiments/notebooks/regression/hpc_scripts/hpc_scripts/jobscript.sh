#!/bin/bash -l
# first non-empty non-comment line ends PBS options

# load bash config
#source etc/.bashrc
source ~/.bashrc

module load python/3.12-conda

# jobs always start in submit directory - change to work directory
cd "$WORK"/pepbench || exit
# activate the venv environment stored in the ".venv" directory
source .venv/bin/activate
# change into script directory
cd experiments/pep_algorithm_benchmarking/notebooks/hpc_scripts || exit
# set environment variable to disable outdated package
export OUTDATED_IGNORE=1

uv sync --dev

# run (all parameters are passed via environment variables)
python "$FILE_NAME.py"
