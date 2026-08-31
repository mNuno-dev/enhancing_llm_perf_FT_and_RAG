#! /bin/bash


attempt=$(basename $0 .sh)
attempt=${attempt##*-}
output_dir=output-dir/gervasio-1b-$attempt

deactivate

source ../llama3-finetune/venv/bin/activate

set -x

export BATCH_MAX_TOKENS=2000
export BATCH_MAX_EXAMPLES=12

mkdir -vp $output_dir

nohup \
    stdbuf -eL -oL \
        python3 -m llama_recipes.finetuning \
            --model_name meta-llama/Llama-3.2-1B-Instruct \
            --mixed_precision True \
            --use_peft \
            --peft_method lora \
            --lora_config.r 32 \
            --lora_config.lora_alpha 128 \
            --batch_size_training 1 \
            --gradient_accumulation_steps 1 \
            --batching_strategy padding \
            --num_epochs 1 \
            --dataset custom_dataset \
            --custom_dataset.file "../gervasio-datasets/mmlu_dataset.py" \
            --output_dir $output_dir \
> $output_dir/train.log 2>&1 &


