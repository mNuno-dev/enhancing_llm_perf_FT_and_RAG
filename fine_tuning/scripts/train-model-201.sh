#! /usr/bin/env bash


base_model_name=meta-llama/Llama-3.2-1B-Instruct
base_model_path="../models/$base_model_name"
output_dir="../models/PORTULAN/gervasio-1b-201"
datasets_dir="../gervasio-datasets"
datasets_cache_dir="../gervasio-datasets/cache"

mkdir -p "$output_dir"

export CUDA_VISIBLE_DEVICES=5
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export CUBLAS_WORKSPACE_CONFIG=":4096:8"


gervasio-download-model $base_model_name $base_model_path

while true; do
    logfile="$output_dir/train-$(date +'%Y-%m-%d_%H-%M-%S').log"
    date > $logfile
    free -hw >> $logfile
    df -h >> $logfile
    # nohup torchrun --standalone --nnodes=1 --nproc_per_node=10 --module \
    nohup python -m \
    gervasio.train \
        --datasets-dir $datasets_dir \
        --datasets-cache-dir $datasets_cache_dir \
        --datasets-aliases "wikipedia_pt,proverbios,extraglue_instruct,natinst,mmlu,mmlu" \
        --datasets-global-shuffle \
        --dataset-config "proverbios:sample_size:int:1000000" \
        --dataset-config "proverbios:example_size:int:36" \
        --dataset-config "natinst:whitelist:str:$datasets_dir/natinst/natinst_whitelist.txt" \
        --batch-max-examples 12 \
        --batch-max-tokens 1800 \
        --num-epochs 3 \
        --num-epoch-checkpoints 6 \
        --warmup-steps 100 \
        --auto-resume \
        --base-model-name $base_model_path \
        --output-dir $output_dir \
        --full-finetuning \
        --enable-gradient-checkpointing \
        --verbose \
    >> $logfile 2>&1
    date >> $logfile
    free -hw >> $logfile
    df -h >> $logfile
    sleep 120
done &

        # --enable-fsdp \