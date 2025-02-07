#!/bin/bash -l
# first non-empty non-comment line ends PBS options

echo "Reached shell script"
# load bash config
source etc/.bashrc

module load python/3.9-anaconda
echo "Successfully loaded python/3.9-anaconda"

# jobs always start in submit directory - change to work directory
cd "$WORK"/pepbench || exit
echo "Reached working directory"
# activate the venv environment stored in the ".venv" directory
source .venv/bin/activate
echo "Activated virtual environment"
# change into script directory
cd experiments/pep_algorithm_benchmarking/notebooks/hpc_scripts || exit
echo "changed into script directory"
# set environment variable to disable outdated package
export OUTDATED_IGNORE=1

# run (all parameters are passed via environment variables)
python "$FILE_NAME.py"
