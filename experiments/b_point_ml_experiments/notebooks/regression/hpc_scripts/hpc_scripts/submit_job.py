#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

from hpc_helper import build_job_submit_slurm, check_interpreter

# print path of python interpreter
print(sys.executable)

# deploy_type = sys.argv[1]

# check_interpreter(deploy_type)

job_list = [
    {
         "name": "regression",
         "type": "general",
         "walltime": "24:00:00",
         "random_state": 0,
    }
]

print("CURRENT DIRECTORY")
print(Path().absolute())
print(f"Length of job_list: {len(job_list)}")

for job_dict in job_list:
    walltime = job_dict.pop("walltime")
    file_name = job_dict.pop("name")
    kwargs = {f"PARAM__{k.upper()}": v for k, v in job_dict.items()}

    job_name = file_name
    for key, value in job_dict.items():
        if key in ("previous_job_id", "feature_extraction_job_id"):
            continue
        job_name += f"_{key}_{value}"

    qsub_command = build_job_submit_slurm(
        job_name=job_name,
        target_system="woody",
        script_name="jobscript.sh",
        nodes=1,
        tasks_per_node=32,
        walltime=walltime,
        FILE_NAME=file_name,
        **kwargs,
    )

    print(qsub_command)  # Uncomment this line when testing to view the qsub command
    # Comment the following 3 lines when testing to prevent jobs from being submitted
    exit_status = subprocess.call(qsub_command, shell=True)
    if exit_status == 1:  # Check to make sure the job submitted
        print(f"Job '{qsub_command}' failed to submit")

print("Done submitting all jobs!")
