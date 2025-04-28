#!/bin/bash -l
# Helper script that gets run by salloc to set up the python debug proxy and port forwardings

if [ "$SLURM_JOB_NUM_NODES" -gt 1 ]; then
  echo "Debugging only works when you request exactly one compute node!" >&2
  exit
fi

echo "Starting port forwarding and debugging proxy for port $PORT on woody..."
python3 "${PROJ_ROOT}/.hpc/pydev_proxy_slurm.py" "$PORT" "$SLURM_JOB_NODELIST" &

sleep 3

srun --pty init_and_run.sh
