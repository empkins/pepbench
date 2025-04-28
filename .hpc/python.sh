#!/bin/bash
# load the HPC bash config
source /etc/bash.bashrc
source /etc/bash.bashrc.local

# load python 3.8
module load python/3.8-anaconda

WORK_DIR=`pwd`

# get the directory of the helper scripts
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# on Windows the working directory seems to always be $HOME
# cd into the project directory (parent of .hpc/)
cd $SCRIPT_DIR/..

PROJ_ROOT=`pwd`

# check if venv or .venv directory exists, otherwise initialize .venv
if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
  echo "venv and .venv do not exist. Creating a Python venv in .venv..."
  python3 -m venv .venv
fi

if [ -d "venv" ]; then
  source venv/bin/activate
elif [ -d ".venv" ]; then
  source .venv/bin/activate
else
  echo "No venv found; Skipping venv activation..."
fi

cd $WORK_DIR

# python [-bBdEhiIOqsSuvVWx?] [-c command | -m module-name | script | - ] [args]
PARAMS=""
while [[ $# -gt 0 ]]
do
  key="$1"
  case $key in
    --submit)
      SUBMIT=1
      shift
      ;;
    --run-torque)
      START_TORQUE_JOB=1
      shift
      ;;
    --run)
      START_JOB=1
      shift
      ;;
    --qsub)
      QSUB_OPTIONS="$2"
      shift 2
      ;;
    --sbatch)
      SBATCH_OPTIONS="$2"
      shift 2
      ;;
    --cluster)
      CLUSTER="$2"
      shift 2
      ;;
    -[bBdEhiIOqsSuvVWx?])  # keep python options to pass to the interpreter
      PARAMS="$PARAMS $1"
      shift
      ;;
    -c)  # run python with code directly
      CODE="$2"
      shift 2
      ;;
    --port)
      DEBUG="True"
      DEBUG_PORT="$2"
      PARAMS="$PARAMS $1 $2"
      shift 2
      ;;
    *) # preserve positional arguments
      PARAMS="$PARAMS $1"
      shift
      ;;
  esac
done

# restore $@ to contain the remaining options
eval set -- "$PARAMS"

variables="params=$*"
if [[ -n $START_TORQUE_JOB ]]; then
  if [[ -n $PYCHARM_DEBUG ]]; then
    variables="PYCHARM_DEBUG=True,$variables"
  fi
  echo "Starting interactively with $variables"
  variables="PROJ_ROOT=$PROJ_ROOT,WORK_DIR=$WORK_DIR,$variables"
  qsub.tinygpu -l nodes=1:ppn=4 -l walltime=01:00:00 -v "$variables" -I -x "$SCRIPT_DIR/init_and_run.sh" -w $WORK_DIR
elif [[ -n $START_JOB ]]; then
  # set variables as environment variables
  export PROJ_ROOT="$PROJ_ROOT"
  export WORK_DIR="$WORK_DIR"
  export params="$PARAMS"

  if [[ -n $DEBUG ]]; then
    if [[ -n $PYCHARM_DEBUG ]]; then
      export PYCHARM_DEBUG="True"
      variables="PYCHARM_DEBUG=True,$variables"
    fi

    export PORT="$DEBUG_PORT"

    echo "Starting debugging with $variables"
    salloc.tinygpu --gres=gpu:1 --time=01:00:00 --chdir="$SCRIPT_DIR" "debugger.sh"
  else
    echo "Starting interactively with $variables"
    srun.tinygpu --gres=gpu:1 --time=01:00:00 --chdir="$WORK_DIR" --pty "$SCRIPT_DIR/init_and_run.sh"
  fi
elif [[ -n $SUBMIT ]]; then
  export QSUB_OPTIONS="$QSUB_OPTIONS"
  export SBATCH_OPTIONS="$SBATCH_OPTIONS"
  export PROJ_ROOT="$PROJ_ROOT"
  export WORK_DIR="$WORK_DIR"
  export CLUSTER="$CLUSTER"
  python3 "$SCRIPT_DIR/submit_wrapper.py" "$@"
elif [[ -n $CODE ]]; then
  python3 -c "$CODE"
else
  python3 "$@"
fi



