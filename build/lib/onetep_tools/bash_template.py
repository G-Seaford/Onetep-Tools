#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from string import Template

from .data_structures import SlurmParams


# GPU SBATCH logic
def gpu_sbatch_lines(slurm: SlurmParams) -> list[str]:
    """
    Return GPU-related SBATCH directives appropriate for the system.
    Empty list means CPU-only job.

    Policy:
      - Sulis: use --gres=gpu:lovelace_l40:N
      - Blythe: use --gres=gpu:lovelace_l40:N
      - Isambard-AI: use --gpus=N
      - Archer2: use --gpus=N
    """
    if slurm.gpus is None: return []
    if slurm.system == "Sulis": return [f"#SBATCH --gres=gpu:lovelace_l40:{slurm.gpus}"]
    if slurm.system == "Isambard-AI": return [f"#SBATCH --gpus={slurm.gpus}"]
    if slurm.system == "Blythe": return [f"#SBATCH --gres=gpu:lovelace_l40:{slurm.gpus}"]
    if slurm.system == "Archer2": return [f"#SBATCH --gpus={slurm.gpus}"]
    return []


# Header + machine-specific comment blocks
HEADER = """#!/bin/bash
# --------------------------------------------------------------------------------
# A SLURM submission script generator for ONETEP.
# Supports hybrid (MPI/OMP) parallelism and (where appropriate) GPUs.
#
# Maintained in onetep_tools by:
#   Gianluca Seaford, Luca.Seaford@warwick.ac.uk
#   University of Warwick
#
# This script is based off of the ONETEP submission script for ARCHER2 created by:
#
#         Jacek Dziedzic, J.Dziedzic@soton.ac.uk
#                         University of Southampton
#         Lennart Gundelach, L.Gundelach@soton.ac.uk
#                            University of Southampton
#         Tom Demeyere, T.Demeyere@soton.ac.uk
#                       University of Southampton
#
# Full credit to them for the original script.
# --------------------------------------------------------------------------------
"""

COMMENTS_BLYTHE = """ --------------------------------------------------------------------------------
# Target machine: SCRTP Blythe (Warwick)
# --------------------------------------------------------------------------------
"""

COMMENTS_SULIS = """# --------------------------------------------------------------------------------
# Target machine: Sulis
# --------------------------------------------------------------------------------
"""

COMMENTS_ARCHER2 = """# --------------------------------------------------------------------------------
# Target machine: ARCHER2
# --------------------------------------------------------------------------------
"""

COMMENTS_ISAMBARDAI = """# --------------------------------------------------------------------------------
# Target machine: Isambard-AI
# --------------------------------------------------------------------------------
# Supports hybrid (MPI/OMP) parallelism, and optionally GPUs.
#
# IMPORTANT:
# It is crucial to submit jobs not from the login node, but from an interactive node
# (so run 'interactive' first). Otherwise modules may not work correctly and mpirun
# may not be found.
# --------------------------------------------------------------------------------
"""

# SBATCH blocks
SBATCH_BLYTHE = Template(r"""# =================================================================================

#SBATCH --partition=$partition
$gpu_block
#SBATCH --nodes=$nodes
#SBATCH --ntasks-per-node=$tasks_per_node
#SBATCH --cpus-per-task=$cpus_per_task
#SBATCH --time=$walltime
#SBATCH --mem-per-cpu=$mem_per_cpu_mb

#SBATCH --job-name=$job_name
#SBATCH --output=sbatch_files/$job_name-%j.out
#SBATCH --error=sbatch_files/$job_name-%j.err

# =================================================================================
""")

SBATCH_SULIS = Template(r"""# =================================================================================

#SBATCH --partition=$partition
$gpu_block
#SBATCH --nodes=$nodes
#SBATCH --ntasks-per-node=$tasks_per_node
#SBATCH --cpus-per-task=$cpus_per_task
#SBATCH --time=$walltime
#SBATCH --mem-per-cpu=${mem_per_cpu_mb}mb
$account_line
#SBATCH --job-name=$job_name
#SBATCH --output=sbatch_files/$job_name-%j.out
#SBATCH --error=sbatch_files/$job_name-%j.err

# =================================================================================
""")

SBATCH_ARCHER2 = Template(r"""# =================================================================================

#SBATCH --partition=$partition
$gpu_block
#SBATCH --nodes=$nodes
#SBATCH --ntasks-per-node=$tasks_per_node
#SBATCH --cpus-per-task=$cpus_per_task
#SBATCH --time=$walltime
$account_line
$qos_line
#SBATCH --job-name=$job_name
#SBATCH --output=sbatch_files/$job_name-%j.out
#SBATCH --error=sbatch_files/$job_name-%j.err

# =================================================================================
""")

SBATCH_ISAMBARDAI = Template(r"""# =================================================================================

$gpu_block
#SBATCH --nodes=$nodes
#SBATCH --ntasks-per-node=$tasks_per_node
#SBATCH --cpus-per-task=$cpus_per_task
#SBATCH --time=$walltime
#SBATCH --exclusive

#SBATCH --job-name=$job_name
#SBATCH --output=sbatch_files/$job_name-%j.out
#SBATCH --error=sbatch_files/$job_name-%j.err

# =================================================================================
"""
)

# Module / environment blocks

MODULES_BLYTHE = Template(r"""# Ensure the cpus-per-task option is propagated to srun commands
export OMP_NUM_THREADS=$$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$$SLURM_CPUS_PER_TASK

# Set up the job environment
module purge; module load GCC/13.3.0 OpenMPI/4.1.6-GCC-13.2.0 OpenBLAS/0.3.24-GCC-13.2.0 FFTW/3.3.10-GCC-13.3.0 ScaLAPACK/2.2.0-gompi-2024a-fb

# Set ONETEP executable and launcher
ONETEP_EXEC="$onetep_binary_path"
ONETEP_LAUNCHER="$onetep_launcher_path"
""")

MODULES_SULIS = Template(r"""# Ensure the cpus-per-task option is propagated to srun commands
export OMP_NUM_THREADS=$$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$$SLURM_CPUS_PER_TASK

# Set up the job environment
module purge; module load GCC/10.2.0 OpenMPI/4.0.5 OpenBLAS FFTW ScaLAPACK

# Set ONETEP executable and launcher
ONETEP_EXEC="$onetep_binary_path"
ONETEP_LAUNCHER="$onetep_launcher_path"
""")

MODULES_ARCHER2 = Template(r"""# Ensure the cpus-per-task option is propagated to srun commands
export OMP_NUM_THREADS=$$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$$SLURM_CPUS_PER_TASK

# Set up the job environment (edit to match ARCHER2 environment/module policy)
module purge
# module load PrgEnv-gnu ...
# module load cray-fftw ...

# Set ONETEP executable and launcher
ONETEP_EXEC="$onetep_binary_path"
ONETEP_LAUNCHER="$onetep_launcher_path"
""")

MODULES_ISAMBARDAI = Template(r"""# Point this to your ONETEP executable. Do not rename your executable, only adjust the directory.
                              
# Set ONETEP executable and launcher
ONETEP_EXEC="$onetep_binary_path"
ONETEP_LAUNCHER="$onetep_launcher_path"

omp_threads_per_mpi_rank=$$SLURM_CPUS_PER_TASK
gpus_per_node=4
""")

# Core blocks
CORE_UPPER = Template(r"""# =================================================================================
# !!! You should not need to modify anything below this line.
# =================================================================================

WORK_DIR=`pwd`
echo "--- Job submitted at: `date`."
                      
# Log files
modules_list="sbatch_files/modules-$$SLURM_JOB_ID.txt"
ldd_file="sbatch_files/ldd-$$SLURM_JOB_ID.txt"

$machine_echo_exec

# Ensure exactly 1 .dat file in there.
NDATS=$$(ls -1 *.dat 2>/dev/null | wc -l)

if [ "$$NDATS" == "0" ]; then
    echo "!!! There is no .dat file in the current directory. Aborting." >&2
    touch "%NO_DAT_FILE"
    exit 2
fi

if [ "$$NDATS" == "1" ]; then
    true
else
    echo "!!! More than one .dat file in the current directory, that's too many. Aborting!" >&2
    touch "%MORE_THAN_ONE_DAT_FILE"
    exit 3
fi

SEED=`echo *.dat | sed -r "s/\.dat$$//"`
SEED_DAT=$$SEED".dat"
SEED_OUT=$$SEED".out"
SEED_ERR=$$SEED".err"

echo "--- ONETEP Input: $$SEED_DAT."
echo "--- ONETEP Output: $$SEED_OUT."
echo "--- ONETEP Error File: $$SEED_ERR."

$machine_check_exec

# Dump the module list to a file.
module list >"$$modules_list" 2>&1

$machine_ldd

echo "--- Number of nodes (as reported by SLURM): $$SLURM_JOB_NUM_NODES."
echo "--- Number of tasks (as reported by SLURM): $$SLURM_NTASKS."
echo "--- Using this launcher: $machine_launcher_which"
echo "--- Executing ONETEP."
""")

CORE_LOWER = Template(r"""
echo "--- Run finished at `date`."

RESULT=$$?
if [ $$RESULT -ne 0 ]; then
    echo "!!! Launcher reported a non-zero exit code: $$RESULT. Aborting!" >&2
    touch "%RUN_ERROR"
    exit 6
fi

if [ -r $$SEED.error_message ]; then
    echo "!!! ONETEP error message present. Aborting!" >&2
    touch "%ONETEP_ERROR_DETECTED"
    exit 7
fi

tail $$SEED.out | grep completed >/dev/null 2>/dev/null
RESULT=$$?
if [ $$RESULT -ne 0 ]; then
    echo "!!! ONETEP calculation did not complete. Aborting!" >&2
    touch "%ONETEP_DID_NOT_COMPLETE"
    exit 8
fi

touch "%DONE"
echo "--- ONETEP finished successfully at `date`."
""")


# Run blocks
RUN_SRUN_BLYTHE = Template(r"""##########################################################################################################################################################################
srun --cpu-bind=verbose,cores --distribution=block:block -c$$OMP_NUM_THREADS -N $$SLURM_JOB_NUM_NODES -n $$SLURM_NTASKS $$ONETEP_LAUNCHER -e $$ONETEP_EXEC $$SEED_DAT > $$SEED_OUT 2> $$SEED_ERR
##########################################################################################################################################################################
""")

RUN_SRUN_GENERIC = Template(r"""##########################################################################################################################################################################
srun -c$$OMP_NUM_THREADS -N $$SLURM_JOB_NUM_NODES -n $$SLURM_NTASKS $$ONETEP_LAUNCHER -e $$ONETEP_EXEC $$SEED_DAT > $$SEED_OUT 2> $$SEED_ERR
##########################################################################################################################################################################
""")

RUN_MPIRUN_ISAMBARDAI_CPU = Template(r"""########################################################################################################################################################
tasks_per_node=`echo $$SLURM_NTASKS $$SLURM_JOB_NUM_NODES | awk '{print $$1/$$2}'`

# Establish environment to use.
envir=`basename "$$ONETEP_EXEC" | sed "s/onetep.iridisx.//"`
if echo $$envir | grep nvfortran; then
  envir_class="nvidia"
elif echo $$envir | grep ifx; then
  envir_class="intel"
  echo "!!! Intel environment is not supported on this machine. Aborting!" >&2
  exit 6
else
  echo "!!! Unknown environment -- check the name of your ONETEP executable. Aborting!" >&2
  exit 6
fi

if [ "$$envir_class" == "nvidia" ]; then
  module load nvidia nvhpc
fi

########################################################################################################################################################
mpirun --map-by ppr:$$tasks_per_node:node:PE=$$omp_threads_per_mpi_rank $$ONETEP_LAUNCHER -e $$ONETEP_EXEC -t $$omp_threads_per_mpi_rank $$SEED_DAT >$$SEED_OUT 2>$$SEED_ERR
########################################################################################################################################################
""")

RUN_MPIRUN_ISAMBARDAI_GPU = Template(r"""########################################################################################################################################################
tasks_per_node=`echo $$SLURM_NTASKS $$SLURM_JOB_NUM_NODES | awk '{print $$1/$$2}'`

envir=`basename "$$ONETEP_EXEC" | sed "s/onetep.iridisx.//"`
if echo $$envir | grep nvfortran; then
  envir_class="nvidia"
elif echo $$envir | grep ifx; then
  envir_class="intel"
  echo "!!! Intel environment is not supported on this machine. Aborting!" >&2
  exit 6
else
  echo "!!! Unknown environment -- check the name of your ONETEP executable. Aborting!" >&2
  exit 6
fi

if [ "$$envir_class" == "nvidia" ]; then
  module load nvidia nvhpc
fi

########################################################################################################################################################
mpirun --map-by ppr:$$tasks_per_node:node:PE=$$omp_threads_per_mpi_rank $$ONETEP_LAUNCHER -g $$gpus_per_node -G -e $$ONETEP_EXEC -t $$omp_threads_per_mpi_rank $$SEED_DAT >$$SEED_OUT 2>$$SEED_ERR
########################################################################################################################################################
""")


# Machine-specific tiny fragments=
MACHINE_FRAGMENTS = {
    "Blythe": {
        "echo_exec": 'echo "--- ONETEP executable is $ONETEP_EXEC."\n'
                     'echo "--- ONETEP launcher is $ONETEP_LAUNCHER."\n'
                     'echo "--- workdir is \'$WORK_DIR\'."',
        "check_exec": r"""# Ensure ONETEP executable exists and is indeed executable.
if [ ! -x "$ONETEP_EXEC" ]; then
    echo "!!! $ONETEP_EXEC does not exist or is not executable. Aborting!" >&2
    touch "%ONETEP_EXE_MISSING"
    exit 4
fi

# Ensure onetep_launcher exists and is executable.
if [ ! -x "$ONETEP_LAUNCHER" ]; then
    echo "!!! $ONETEP_LAUNCHER does not exist or is not executable. Aborting!" >&2
    touch "%ONETEP_LAUNCHER_MISSING"
    exit 5
fi
""",
        "ldd": 'ldd "$ONETEP_EXEC" >"$ldd_file"',
        "launcher_which": "`which srun`",
    },
    "Sulis": {
        "echo_exec": 'echo "--- ONETEP executable is $ONETEP_EXEC."\n'
                     'echo "--- ONETEP launcher is $ONETEP_LAUNCHER."\n'
                     'echo "--- workdir is \'$WORK_DIR\'."',
        "check_exec": r"""if [ ! -x "$ONETEP_EXEC" ]; then
    echo "!!! $ONETEP_EXEC does not exist or is not executable. Aborting!" >&2
    touch "%ONETEP_EXE_MISSING"
    exit 4
fi
if [ ! -x "$ONETEP_LAUNCHER" ]; then
    echo "!!! $ONETEP_LAUNCHER does not exist or is not executable. Aborting!" >&2
    touch "%ONETEP_LAUNCHER_MISSING"
    exit 5
fi
""",
        "ldd": 'ldd "$ONETEP_EXEC" >"$ldd_file"',
        "launcher_which": "`which srun`",
    },
    "Archer2": {
        "echo_exec": 'echo "--- ONETEP executable is $ONETEP_EXEC."\n'
                     'echo "--- ONETEP launcher is $ONETEP_LAUNCHER."\n'
                     'echo "--- workdir is \'$WORK_DIR\'."',
        "check_exec": r"""if [ ! -x "$ONETEP_EXEC" ]; then
    echo "!!! $ONETEP_EXEC does not exist or is not executable. Aborting!" >&2
    touch "%ONETEP_EXE_MISSING"
    exit 4
fi
if [ ! -x "$ONETEP_LAUNCHER" ]; then
    echo "!!! $ONETEP_LAUNCHER does not exist or is not executable. Aborting!" >&2
    touch "%ONETEP_LAUNCHER_MISSING"
    exit 5
fi
""",
        "ldd": 'ldd "$ONETEP_EXEC" >"$ldd_file"',
        "launcher_which": "`which srun`",
    },
    "Isambard-AI": {
        "echo_exec": 'echo "--- ONETEP executable is \'$ONETEP_EXEC\'."\n'
                     'echo "--- ONETEP launcher is \'$ONETEP_LAUNCHER\'."\n'
                     'echo "--- workdir is \'$WORK_DIR\'."',
        "check_exec": r"""if [ ! -x "$ONETEP_EXEC" ]; then
    echo "!!! $ONETEP_EXEC does not exist or is not executable. Aborting!" >&2
    touch "%ONETEP_EXE_MISSING"
    exit 4
fi
if [ ! -x "$ONETEP_LAUNCHER" ]; then
    echo "!!! $ONETEP_LAUNCHER does not exist or is not executable. Aborting!" >&2
    touch "%ONETEP_LAUNCHER_MISSING"
    exit 5
fi
""",
        "ldd": 'ldd "$ONETEP_EXEC" >"$ldd_file"',
        "launcher_which": "`which mpirun`",
    },
}


# Main writer function
def write_sbatch(workdir: Path, seed: str, slurm: SlurmParams) -> None:
    """
    Write the correct SLURM submission script into workdir/run_sub based on slurm.system.

    Note: #SBATCH directives must be resolved before submission, so job_name is set here
    from (slurm.job_name or seed), not inferred later in bash.
    """
    sbatch_dir = workdir / "sbatch_files"
    sbatch_dir.mkdir(parents=True, exist_ok=True)

    job_name = slurm.job_name or seed

    gpu_lines = gpu_sbatch_lines(slurm)
    gpu_block = ("\n".join(gpu_lines) + "\n") if gpu_lines else ""


    ctx = {
        "partition": slurm.partition,
        "nodes": str(slurm.nodes),
        "tasks_per_node": str(slurm.tasks_per_node),
        "cpus_per_task": str(slurm.cpus_per_task),
        "walltime": slurm.walltime,
        "mem_per_cpu_mb": str(slurm.mem_per_cpu_mb),
        "job_name": job_name,
        "onetep_binary_path": slurm.onetep_binary,
        "onetep_launcher_path": slurm.onetep_launcher,
        "gpu_block": gpu_block,
        "account_line": "" if not slurm.account else f"#SBATCH --account={slurm.account}",
        "qos_line": "" if not slurm.qos else f"#SBATCH --qos={slurm.qos}",
    }

    system = slurm.system
    if system not in MACHINE_FRAGMENTS:
        raise ValueError(f"Unknown slurm.system {system!r}")

    fr = MACHINE_FRAGMENTS[system]

    # Choose comment block and the per-system SBATCH/modules/run blocks
    if system == "Blythe":
        comment_block = COMMENTS_BLYTHE
        sbatch_block = SBATCH_BLYTHE.substitute(ctx)
        modules_block = MODULES_BLYTHE.substitute(ctx)
        run_block = RUN_SRUN_BLYTHE.substitute(ctx)

    elif system == "Sulis":
        comment_block = COMMENTS_SULIS
        sbatch_block = SBATCH_SULIS.substitute(ctx)
        modules_block = MODULES_SULIS.substitute(ctx)
        run_block = RUN_SRUN_GENERIC.substitute(ctx)

    elif system == "Archer2":
        comment_block = COMMENTS_ARCHER2
        sbatch_block = SBATCH_ARCHER2.substitute(ctx)
        modules_block = MODULES_ARCHER2.substitute(ctx)
        run_block = RUN_SRUN_GENERIC.substitute(ctx)

    elif system == "Isambard-AI":
        comment_block = COMMENTS_ISAMBARDAI
        sbatch_block = SBATCH_ISAMBARDAI.substitute(ctx)
        modules_block = MODULES_ISAMBARDAI.substitute(ctx)

        if slurm.gpus is None:
            run_block = RUN_MPIRUN_ISAMBARDAI_CPU.substitute(ctx)
        else:
            ctx["gpus"] = str(slurm.gpus)
            run_block = RUN_MPIRUN_ISAMBARDAI_GPU.substitute(ctx)

    else:
        # Should be unreachable due to check above
        raise ValueError(f"Unhandled system {system!r}")

    core_upper = CORE_UPPER.substitute(
        machine_echo_exec=fr["echo_exec"],
        machine_check_exec=fr["check_exec"],
        machine_ldd=fr["ldd"],
        machine_launcher_which=fr["launcher_which"],
    )
    core_lower = CORE_LOWER.substitute()

    script = "\n".join(
        [
            HEADER,
            comment_block,
            sbatch_block,
            modules_block,
            core_upper,
            run_block,
            core_lower,
        ]
    )

    (workdir / "run_sub").write_text(script)
