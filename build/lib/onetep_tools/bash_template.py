#!/usr/bin/env python3
from __future__ import annotations

from .data_structures import SlurmParams
from pathlib import Path

def write_sbatch(workdir: Path, seed: str, slurm: SlurmParams) -> None:
    """
    Write the preserved SLURM submission script into workdir/run_sub.
    """
    sbatch_dir = workdir / "sbatch_files"
    sbatch_dir.mkdir(parents=True, exist_ok=True)

    script = f"""#!/bin/bash
# --------------------------------------------------------------------------------
# A SLURM submission script for ONETEP on the SCRTP Blythe HPC machine at Warwick.
# Supports hybrid (MPI/OMP) parallelism.
#
# 2025.07 Gianluca Seaford, Luca.Seaford@warwick.ac.uk
#                         University of Warwick
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
# ---------------------------------------------------------------------------------
#
#
# =================================================================================
#
#SBATCH --partition={slurm.partition}
#SBATCH --nodes={slurm.nodes}
#SBATCH --ntasks-per-node={slurm.tasks_per_node}
#SBATCH --cpus-per-task={slurm.cpus_per_task}
#SBATCH --time={slurm.walltime}
#SBATCH --mem-per-cpu={slurm.mem_per_cpu_mb}
#SBATCH --job-name={slurm.job_name or seed}
#SBATCH --output=sbatch_files/Au-Graphene-%j.out
#SBATCH --error=sbatch_files/Au-Graphene-%j.err

# =================================================================================


# Ensure the cpus-per-task option is propagated to srun commands
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# Set up the job environment
module purge; module load GCC/13.3.0 OpenMPI/4.1.6-GCC-13.2.0 OpenBLAS/0.3.24-GCC-13.2.0 FFTW/3.3.10-GCC-13.3.0 ScaLAPACK/2.2.0-gompi-2024a-fb

# Set ONETEP executable and launcher
LAUNCHER="{slurm.onetep_launcher}"
EXEC="{slurm.onetep_binary}"


# =================================================================================
# !!! You should not need to modify anything below this line.
# =================================================================================

WORK_DIR=`pwd`
echo "--- Job submitted to $SLURM_PARTITION at: `date`."

echo "--- ONETEP executable is $EXEC."
echo "--- ONETEP launcher is $LAUNCHER."
echo "--- workdir is '$WORK_DIR'."

# Ensure exactly 1 .dat file in there.
NDATS=`ls -l *dat | wc -l`

if [ "$NDATS" == "0" ]; then
    echo "!!! There is no .dat file in the current directory. Aborting." >&2
    touch "%NO_DAT_FILE"
    exit 2
fi

if [ "$NDATS" == "1" ]; then
    true
else
    echo "!!! More than one .dat file in the current directory, that's too many. Aborting!" >&2
    touch "%MORE_THAN_ONE_DAT_FILE"
    exit 3
fi

SEED=`echo *.dat | sed -r "s/\\.dat$//"`
SEED_DAT=$SEED".dat"
SEED_OUT=$SEED".out"
SEED_ERR=$SEED".err"

echo "--- ONETEP Input: $SEED_DAT."
echo "--- ONETEP Output: $SEED_OUT." 
echo "--- ONETEP Error File: $SEED_ERR."

# Ensure ONETEP executable exists and is indeed executable.
if [ ! -x "$EXEC" ]; then
    echo "!!! $EXEC does not exist or is not executable. Aborting!" >&2
    touch "%ONETEP_EXE_MISSING"
    exit 4
fi

# Ensure onetep_launcher exiusts and is executable.
if [ ! -x "$LAUNCHER" ]; then
    echo "!!! $LAUNCHER does not exist or is not executable. Aborting!" >&2
    touch "%ONETEP_LAUNCHER_MISSING"
    exit 5
fi

# Dump the module list to a file.
module list >$modules_list 2>&1

ldd $EXEC >$ldd

echo "--- Number of nodes (as reported by SLURM): $SLURM_JOB_NUM_NODES."
echo "--- Number of tasks (as reported by SLURM): $SLURM_NTASKS."
echo "--- Using this srun executable: "`which srun`
echo "--- Executing ONETEP via $LAUNCHER."

# Run ONETEP
##########################################################################################################################################################################
srun --cpu-bind=verbose,cores --distribution=block:block -c$OMP_NUM_THREADS -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS $LAUNCHER -e $EXEC $SEED_DAT > $SEED_OUT 2> $SEED_ERR
##########################################################################################################################################################################

echo "--- srun finished at `date`."

# Check if ONETEP ran successfully

RESULT=$?
if [ $RESULT -ne 0 ]; then
    echo "!!! srun reported a non-zero exit code: $RESULT. Aborting!" >&2
    touch "%SRUN_ERROR"
    exit 6
fi

if [ -r $SEED.error_message ]; then
    echo "!!! ONETEP error message present. Aborting!" >&2
    touch "%ONETEP_ERROR_DETECTED"
    exit 7
fi

tail $SEED.out | grep completed >/dev/null 2>/dev/null
RESULT=$?
if [ $RESULT -ne 0 ]; then
    echo "!!! ONETEP calculation did not complete. Aborting!" >&2
    touch "%ONETEP_DID_NOT_COMPLETE"
    exit 8
fi

touch "%DONE"
echo "--- ONETEP finished successfully at `date`."
"""
    (workdir / "run_sub").write_text(script)