#!/bin/bash
set -e

if [ $# -ne 2 ]; then
    echo "Usage: $0 <job-name> <gpus>"
    exit 1
fi

JOB_NAME=$1
GPUS=$2
CPUS=$((GPUS * 15))

mkdir -p slurm_logs

sbatch \
    --job-name="$JOB_NAME" \
    --nodelist=worker-3 \
    --mem-per-cpu=8192 \
    --gpus="$GPUS" \
    --cpus-per-task="$CPUS" \
    --output="slurm_logs/${JOB_NAME}_%j.log" \
    --error="slurm_logs/${JOB_NAME}_%j.err" \
    --wrap="bash -c '
        set -e
        echo \"Job ID: \$SLURM_JOB_ID\"
        echo \"Node: \$SLURM_NODELIST\"
        echo \"Starting at: \$(date)\"
        echo \"GPUs allocated: \$CUDA_VISIBLE_DEVICES\"

        source /home/hapq/miniconda3/etc/profile.d/conda.sh
        conda activate mvinverse_all

        nvidia-smi
        export PYTHONUNBUFFERED=1
        ./scripts/gen_tartanair_albedo.sh > slurm_logs/tartanair.log 2>&1 &
        ./scripts/gen_flyingchairs_albedo.sh > slurm_logs/flyingchairs.log 2>&1 &
        ./scripts/gen_flyingthings_albedo.sh > slurm_logs/flyingthings.log 2>&1 &
        wait

        echo \"Job finished at: \$(date)\"
    '"
