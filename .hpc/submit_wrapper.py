import argparse
import os
import shlex
import subprocess
import sys
import hashlib


def submit_job_qsub(params, qsub_options):
    args = shlex.split(qsub_options)

    work_dir = os.environ.get("WORK_DIR")
    proj_root = os.environ.get("PROJ_ROOT")
    cluster = os.environ.get("CLUSTER")

    parser = argparse.ArgumentParser()
    parser.add_argument("-N")
    parser.add_argument("-l", action="append")
    parser.add_argument("-m")
    parser.add_argument("-o")
    parser.add_argument("-j")

    options = parser.parse_args(args)

    if cluster == "" or cluster == "tinygpu":
        command = ["qsub.tinygpu"]
    elif cluster == "woody":
        command = ["qsub"]
    else:
        print(f"Cluster {cluster} unknown. Aborting...")
        exit()

    # give the job a name
    if options.N is not None:
        command.extend(["-N", options.N])
    else:
        job_name = hashlib.md5(qsub_options.encode("UTF-8")).hexdigest()
        command.extend(["-N", job_name])

    # request one GPU
    if options.l is not None:
        for arg in options.l:
            command.extend(["-l", arg])
    else:
        command.extend(["-l", "nodes=1:ppn=4", "-l", "walltime=06:00:00"])

    if options.m is not None:
        command.extend(["-m", options.m])

    if options.o is not None:
        command.extend(["-o", options.o])

    if options.j is not None:
        command.extend(["-j", options.j])

    # make sure we are in the correct directory
    command.extend(["-w", work_dir])

    # pass options to the script
    command.extend(["-v", "params=" + params + ",PROJ_ROOT=" + proj_root])

    # TODO: support passing custom variables with -v

    # add the path to the script that the HPC should execute
    command.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "init_and_run.sh"))

    print("Submitting via " + " ".join(command))

    process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=work_dir)
    out, err = process.communicate()
    return out.strip()


def submit_job_sbatch(params, sbatch_options):
    args = shlex.split(sbatch_options)

    work_dir = os.environ.get("WORK_DIR")
    proj_root = os.environ.get("PROJ_ROOT")
    cluster = os.environ.get("CLUSTER")

    parser = argparse.ArgumentParser()
    parser.add_argument("--gres")
    parser.add_argument("-p")
    parser.add_argument("--time")
    parser.add_argument("--job-name")
    parser.add_argument("--export")
    parser.add_argument("--output")
    parser.add_argument("--mail-user")
    parser.add_argument("--mail-type")

    options = parser.parse_args(args)

    command = ["sbatch.tinygpu"]
    if cluster != "" and cluster != "tinygpu":
        print("Submitting via SLURM is currently only implemented for TinyGPU. Aborting...")
        exit()

    # give the job a name
    if options.job_name is not None:
        command.append(f"--job-name={options.job_name}")
    else:
        job_name = hashlib.md5(sbatch_options.encode("UTF-8")).hexdigest()
        command.append(f"--job-name={job_name}")

    # request one GPU
    if options.gres is not None:
        command.append(f"--gres={options.gres}")
    else:
        command.append("--gres=gpu:1")

    # choose work queue by default (can be changed by user to work on v100/a100)
    if options.p is not None:
        command.extend(["-p", options.p])
    else:
        command.extend(["-p", "work"])

    # request 6 hours by default
    if options.time is not None:
        command.append(f"--time={options.time}")
    else:
        command.append("--time=06:00:00")

    if options.mail_user is not None:
        command.append(f"--mail-user={options.mail_user}")

    if options.mail_type is not None:
        command.append(f"--mail-type={options.mail_type}")
    else:
        command.append("--mail-type=ALL")

    if options.output is not None:
        command.append(f"--output={options.output}")

    if options.export is None or options.export == "NONE":
        options.export = ""
    else:
        options.export += ","

    command.append(f"--export={options.export}params,PROJ_ROOT,WORK_DIR")

    # make sure we are in the correct directory
    command.extend(["-D", work_dir])

    # add the path to the script that the HPC should execute
    command.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "init_and_run.sh"))

    print("Submitting via " + " ".join(command))

    process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=work_dir,
                               env=dict(os.environ, params=params, PROJ_ROOT=proj_root, WORK_DIR=work_dir))
    out, err = process.communicate()
    if err:
        print(err)
    return out.strip()


def main():
    params = " ".join(sys.argv[1:])
    qsub_options = os.environ.get("QSUB_OPTIONS")
    sbatch_options = os.environ.get("SBATCH_OPTIONS")

    # strings are falsy so we can just check which of the strings is not empty like this
    if qsub_options:
        job_id = submit_job_qsub(params, qsub_options)
        print("Submitted job with ID {}".format(job_id))
    elif sbatch_options:
        out = submit_job_sbatch(params, sbatch_options)
        print(out)
    else:
        print("Neither --qsub nor --sbatch was specified! Defaulting to using SBATCH")
        out = submit_job_sbatch(params, "")
        print(out)


if __name__ == '__main__':
    main()
