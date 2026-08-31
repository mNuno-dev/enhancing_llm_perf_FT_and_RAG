#!/bin/bash
#SBATCH --job-name=gervasio_train_1B_0177724
#SBATCH --output=/projects/F202500002HPCVLABISTUL/nmarques/ft2/gervasio/scripts/logs/training_1B/%x-%j.out
#SBATCH --error=/projects/F202500002HPCVLABISTUL/nmarques/ft2/gervasio/scripts/logs/training_1B/%x-%j.err
#SBATCH --time=30:00:00
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --account=f202500002hpcvlabistulg

set -euo pipefail

module load CUDA/12.2.2

# Cluster-specific paths. Update REPO_DIR, VENV_DIR, and output paths for your environment.
REPO_DIR="/home/nmarques/project/ft2/gervasio"
VENV_DIR="/home/nmarques/project/ft2/.venv"

cd "$REPO_DIR"
source "$VENV_DIR/bin/activate"

export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

TRAIN_CONFIG="/home/nmarques/project/ft2/gervasio/train_configs/gerv-1b-RAGdocs/0177724/1b-train-config.toml"


OUTPUT_DIR="/home/nmarques/project/ft2/gervasio/models/1b_models/gervasio-1b-0177724"
LOGFILE="$OUTPUT_DIR/train-$(date +'%Y-%m-%d_%H-%M-%S').log"



mkdir -p "$OUTPUT_DIR"

{
    echo "==== HOST ===="
    hostname
    echo "==== DATE ===="
    date
    echo "==== GPU STATUS ===="
    nvidia-smi
    echo "==== MEMORY ===="
    free -hw
    echo "==== DISK ===="
    df -h
} | tee -a "$LOGFILE"

echo "==== Starting training at $(date +'%Y-%m-%d_%H-%M-%S') ====" | tee -a "$LOGFILE"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    --module gervasio.train \
    --train-config "$TRAIN_CONFIG" \
    >> "$LOGFILE" 2>&1

STATUS=$?

echo "==== torchrun exited with status $STATUS at $(date) ====" | tee -a "$LOGFILE"
exit "$STATUS"