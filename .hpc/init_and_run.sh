#!/bin/bash -l
# Test whether we are on Slurm or Torque
if [ -z ${SLURM_JOBID+x} ]; then
  TORQUE=true
else
  TORQUE=false
fi

# load the HPC bash config
source /etc/bash.bashrc
source /etc/bash.bashrc.local

# load python 3.8
module load python/3.8-anaconda

# Activate virtual environment
cd ${PROJ_ROOT}
if [ -d "venv" ]; then
  source venv/bin/activate
elif [ -d ".venv" ]; then
  source .venv/bin/activate
else
  echo "No venv found; Skipping venv activation..."
fi

# cd into the work directory
if [[ "$TORQUE" = true && -n ${PBS_O_WORKDIR} ]]; then
  cd "${PBS_O_WORKDIR}"
else
  cd "${WORK_DIR}"
fi

if [ "$TORQUE" = true ]; then
  arr=($params)
  set -- "${arr[@]}"
  while [[ $# -gt 0 ]]
  do

  key="$1"
  case $key in
      --port)
      PORT="$2"
      shift
      ;;
      *)
      # ignore everything else
      shift
      ;;
  esac
  done

  if [ -n "$PORT" ]; then
    echo "Detected debugging on port $PORT. Setting up port forwarding helper proxy..."
    ssh -N -L 4242:127.0.0.1:$PORT woody &
    python3 ${PROJ_ROOT}/.hpc/pydev_proxy.py $PORT &
    sleep 3
  fi
fi

echo "Running python3 $params in $(pwd)"
python3 $params
