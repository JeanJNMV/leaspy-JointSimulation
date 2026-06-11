#!/bin/bash
#SBATCH --job-name=simu_test
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=out/file_output_%j.out
#SBATCH --error=err/file_output_%j.err

module load python/3.12

cd /network/iss/home/jv.martini/leaspy-Joint_Model_Simu-Eval

source .venv/bin/activate || { echo "ERROR: venv activation failed"; exit 1; }

echo "Python: $(which python)"
echo "Working dir: $(pwd)"
echo "Job started at: $(date)"

python -u scripts/simu_test.py

echo "Job finished at: $(date)"
