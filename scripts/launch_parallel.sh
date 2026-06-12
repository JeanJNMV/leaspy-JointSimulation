#!/bin/bash
#SBATCH --job-name=launcher
#SBATCH --partition=compute
#SBATCH --time=00:05:00
#SBATCH --mem=512M
#SBATCH --cpus-per-task=1
#SBATCH --output=out/launcher_%j.out
#SBATCH --error=err/launcher_%j.err

# Usage:  sbatch scripts/launch_slurm.sh <MODEL> <N> <M> <N_ITER> <N_PERSO>
# Example: sbatch scripts/launch_slurm.sh PULSE_JOINT_100_ALSFRS_BMI_VC_MUSC_NFL_SNIP_10 200 20 10000 1000
#
# MODEL is the filename (with or without .json) inside ./models/

set -euo pipefail

MODEL=${1:?"Usage: sbatch scripts/launch_slurm.sh MODEL N M N_ITER N_PERSO"}
N=${2:?}
M=${3:?}
N_ITER=${4:?}
N_PERSO=${5:?}

# Strip .json if the user passed it, for clean tag names
MODEL_STEM="${MODEL%.json}"

WORKDIR=/network/iss/home/jv.martini/leaspy-Joint_Model_Simu-Eval
cd "$WORKDIR"
mkdir -p out err

# Verify the model file exists before submitting anything
if [[ ! -f "${WORKDIR}/models/${MODEL_STEM}.json" ]]; then
    echo "ERROR: Model file not found: ${WORKDIR}/models/${MODEL_STEM}.json"
    exit 1
fi

LAST_TASK=$((M - 1))
TAG="${MODEL_STEM}_N${N}_M${M}_Niter${N_ITER}_Nperso${N_PERSO}"

# ── 1. Array job ──────────────────────────────────────────────────────────────
ARRAY_JOB=$(sbatch --parsable \
    --job-name="simu_${TAG}" \
    --partition=compute \
    --time=24:00:00 \
    --mem=16G \
    --cpus-per-task=4 \
    --exclude=sphpc-cpu37,sphpc-cpu30 \
    --array="0-${LAST_TASK}" \
    --output="${WORKDIR}/out/simu_${TAG}_%A_%a.out" \
    --error="${WORKDIR}/err/simu_${TAG}_%A_%a.err" \
    --wrap="
        module load python/3.12
        cd ${WORKDIR}
        source .venv/bin/activate || { echo 'ERROR: venv activation failed'; exit 1; }
        echo \"Config: ${TAG}  |  Task: \${SLURM_ARRAY_TASK_ID}  |  Started: \$(date)\"
        python -u scripts/simu_test_parallel.py \
            --model ${MODEL_STEM} \
            --N ${N} --M ${M} --N-iter ${N_ITER} --N-perso ${N_PERSO}
        echo \"Task finished at: \$(date)\"
    ")
echo "Array job submitted: ${ARRAY_JOB}  (${TAG}, tasks 0–${LAST_TASK})"

# ── 2. Aggregation job ───────────────────────────────────────────────────────
AGG_JOB=$(sbatch --parsable \
    --job-name="agg_${TAG}" \
    --partition=compute \
    --time=00:30:00 \
    --mem=4G \
    --cpus-per-task=1 \
    --exclude=sphpc-cpu37,sphpc-cpu30 \
    --dependency="afterany:${ARRAY_JOB}" \
    --output="${WORKDIR}/out/aggregate_${TAG}_%j.out" \
    --error="${WORKDIR}/err/aggregate_${TAG}_%j.err" \
    --wrap="
        module load python/3.12
        cd ${WORKDIR}
        source .venv/bin/activate || { echo 'ERROR: venv activation failed'; exit 1; }
        echo \"Aggregating ${TAG} — \$(date)\"
        python -u scripts/simu_test_parallel.py \
            --model ${MODEL_STEM} \
            --N ${N} --M ${M} --N-iter ${N_ITER} --N-perso ${N_PERSO} --aggregate
        echo \"Done — \$(date)\"
    ")
echo "Aggregation job submitted: ${AGG_JOB}  (depends on ${ARRAY_JOB})"
echo "Monitor with:  squeue -u \$USER"
echo "Cancel with:   scancel ${ARRAY_JOB} ${AGG_JOB}"