#!/bin/bash
#SBATCH --job-name=gervasio_train_1B_1777701
#SBATCH --output=/projects/F202500002HPCVLABISTUL/nmarques/ft2/gervasios/scripts/logs/training_1B/%x-%j.out
#SBATCH --error=/projects/F202500002HPCVLABISTUL/nmarques/ft2/gervasios/scripts/logs/training_1B/%x-%j.err
#SBATCH --time=30:00:00
#SBATCH --partition=normal-a100-40
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --account=f202500002hpcvlabistulg

module load CUDA/12.2.2
source /home/nmarques/project/ft2/.venv/bin/activate

export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Cluster-specific paths. Update BASE_MODEL_NAME_DIR, OUTPUT_DIR, and output paths for your environment.
BASE_MODEL_NAME_DIR="/projects/F202500002HPCVLABISTUL/nmarques/pre_downloaded_models/meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR="/home/nmarques/project/ft2/gervasio/models/1b_models/gervasio-1b-1777701"
DATASETS_DIR="/home/nmarques/project/finetuning/gervasio-datasets"
DATASETS_CACHE="/home/nmarques/project/finetuning/gervasio-datasets/cache"
SLURM_LOG_DIR="/projects/F202500002HPCVLABISTUL/nmarques/ft2/gervasio/scripts/logs/training_1B"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$SLURM_LOG_DIR"

TIMESTAMP=$(date +'%Y-%m-%d_%H-%M-%S')
LOGFILE="$OUTPUT_DIR/train-${TIMESTAMP}.log"

{
    echo "==== GPU STATUS (nvidia-smi) ===="
    nvidia-smi
} | tee -a "$LOGFILE"

echo "==== Starting training at $TIMESTAMP ====" | tee -a "$LOGFILE"
free -hw >> "$LOGFILE"
df -h >> "$LOGFILE"

torchrun --standalone --nnodes=1 --nproc_per_node=4 --module \
    gervasio.train \
    --datasets-dir "$DATASETS_DIR" \
    --datasets-cache-dir "$DATASETS_CACHE" \
    --datasets-aliases "ptwikirag, ptwikirag" \
    --datasets-global-shuffle \
    --batch-max-examples 14 \
    --batch-max-tokens 1800 \
    --lr 1e-5 \
    --num-epochs 2 \
    --num-epoch-checkpoints 6 \
    --warmup-steps 500 \
    --auto-resume \
    --base-model-name "$BASE_MODEL_NAME_DIR" \
    --adapter-name "gervasio-1b-1777701" \
    --output-dir "$OUTPUT_DIR" \
    --lora-r 32 \
    --lora-alpha 128 \
    --enable-fsdp \
    --seed 999 \
    --verbose \
    --prompts-file /home/nmarques/project/ft2/gervasio/checkpoint_eval_gen/all_built_prompts_RAG_and_noRAG_2datasets_dev.jsonl \
    --generation-config /home/nmarques/project/ft2/gervasio/train_configs/example-gen-config.toml \
    >> "$LOGFILE" 2>&1

STATUS=$?
echo "==== torchrun exited with status $STATUS at $(date) ====" | tee -a "$LOGFILE"
free -hw >> "$LOGFILE"
df -h >> "$LOGFILE"

exit "$STATUS"