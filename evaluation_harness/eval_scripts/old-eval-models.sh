#! /usr/bin/bash
# Evaluation script used during the initial development phase with vLLM.
# Superseded by eval-models.sh, which additionally supports local completions servers.
# Configure model_paths and tasks below before running.

model_paths=""
tasks=""
seeds=""

# model_paths+=" $(readlink -f ../models/meta-llama/Llama-3.2-1B-Instruct)"
# model_paths+=" $(readlink -f ../models/meta-llama/Llama-3.2-3B-Instruct)"
# model_paths+=" $(readlink -f ../models/meta-llama/Llama-3.1-8B-Instruct)"
# model_paths+=" $(readlink -f ../models/meta-llama/Llama-3.3-70B-Instruct)"

# model_paths+=" $(readlink -f ../models/PORTULAN/gervasio-70b-091-epoch2-alpha32-merged)"
# model_paths+=" $(readlink -f ../models/PORTULAN/gervasio-70b-091-epoch2-alpha64-merged)"

# model_paths+=" $(readlink -f ../models/PORTULAN/gervasio-8b-092-epoch1-alpha16-merged)"
# model_paths+=" $(readlink -f ../models/PORTULAN/gervasio-8b-092-epoch2-alpha16-merged)"
# model_paths+=" $(readlink -f ../models/PORTULAN/gervasio-8b-092-epoch3-alpha16-merged)"

# model_paths+=" $(readlink -f ../models/PORTULAN/gervasio-8b-092-epoch1-alpha24-merged)"
# model_paths+=" $(readlink -f ../models/PORTULAN/gervasio-8b-portuguese-ptpt-decoder-quantized-4bit/gervasio-8b-portuguese-ptpt-decoder-Q4_K.gguf)"
# model_paths+=" $(readlink -f ../models/PORTULAN/gervasio-8b-092-epoch2-alpha24-merged)"
# model_paths+=" $(readlink -f ../models/PORTULAN/gervasio-8b-092-epoch3-alpha24-merged)"

# model_paths+=" $(readlink -f ../models/PORTULAN/gervasio-8b-092-epoch1-alpha32-merged)"
# model_paths+=" $(readlink -f ../models/PORTULAN/gervasio-8b-092-epoch2-alpha32-merged)"
# model_paths+=" $(readlink -f ../models/PORTULAN/gervasio-8b-092-epoch3-alpha32-merged)"

# model_paths+=" $(readlink -f ../models/PORTULAN/gervasio-8b-092-epoch1-alpha64-merged)"
# model_paths+=" $(readlink -f ../models/PORTULAN/gervasio-8b-092-epoch2-alpha64-merged)"
# model_paths+=" $(readlink -f ../models/PORTULAN/gervasio-8b-092-epoch3-alpha64-merged)"

# model_paths+=" $(readlink -f ../models/maritaca-ai/sabia-7b)"
# model_paths+=" $(readlink -f ../models/Qwen/Qwen3-32B)"
# model_paths+=" $(readlink -f /home/nmarques/dev/models/PORTULAN/Qwen/Qwen3-32B)"

# model_paths+=" $(readlink -f ../models/mistralai/Mistral-Small-24B-Instruct-2501)"
# model_paths+=" $(readlink -f ../models/deepseek-ai/deepseek-coder-1.3b-instruct)"

# GGUF models

# NOTE: Path below is environment-specific. Replace with the local path to your GGUF model file.
# model_paths+=" $(readlink -f /home/nmarques/dev/models/PORTULAN/gervasio-8b-portuguese-ptpt-decoder-quantized-4bit/gervasio-8b-portuguese-ptpt-decoder-Q4_K.gguf)"

# Thesis fine-tuned models (paths are environment-specific)
# model_paths+=" $(readlink -f /home/nmarques/thesisModels/gervasio-1b-101/gervasio-1b-101-alpha64-merged)"
# model_paths+=" $(readlink -f /home/nmarques/thesisModels/gervasio-3b-301/gervasio-3b-301-alpha64-merged)"



run_main_tasks=true
run_optional_tasks=true
gpu_memory_utilization=0.2


if $run_main_tasks; then
    # 6 min
    tasks+=" gervasio-extraglue_copa-ptpt"

    # 7 min
    tasks+=" gervasio-extraglue_rte-ptpt"

    # 17 min
    tasks+=" gervasio-meta_gpqa-diamond"

    # 20 min
    tasks+=" gervasio-extraglue_mrpc-ptpt"

    # 5h22m
    tasks+=" gervasio-meta_mmlu"

    # 15h
    tasks+=" gervasio-meta_mmlu-pro"
fi

if $run_optional_tasks; then
    # 7 min
    tasks+=" gervasio-meta_mbppplus"

    # 8 min
    tasks+=" gervasio-meta_mgsm"

    # 10 min
    tasks+=" gervasio-meta_humaneval"

    tasks+=" gervasio-meta_math"
fi

seeds+=" 2345"

# this is required because the dataset paths in the task configurations are relative
cd "../gervasio-eval"

export RAY_NUM_CPUS=40
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9"
export OMP_NUM_THREADS=2


for seed in $seeds; do

    for model_path in $model_paths; do
        if test -f $model_path; then
            eval_dir="$(dirname $model_path)/eval"
            echo "$model_path is a file; eval output will be saved in the parent directory" >&2
        else
            eval_dir="$model_path/eval"
        fi

        output_dir="$eval_dir/outputs"
        mkdir -p "$output_dir"

        for task in $tasks; do

            logfile="$eval_dir/eval-$task-seed$seed-$(date +'%Y-%m-%d_%H-%M-%S').log"

            lm_eval_args="--include_path task_configurations"
            lm_eval_args+=" --tasks $task"
            lm_eval_args+=" --output_path $output_dir"
            lm_eval_args+=" --confirm_run_unsafe_code"
            lm_eval_args+=" --batch_size auto"
            lm_eval_args+=" --log_samples"
            lm_eval_args+=" --seed $seed"

            vllm_args="pretrained=$model_path"
            vllm_args+=",max_model_len=4096"
            vllm_args+=",task=generate"
            vllm_args+=",dtype=auto"
            vllm_args+=",gpu_memory_utilization=$gpu_memory_utilization"
            # vllm_args+=",chat_template_kwargs={\"enable_thinking\":false}"


            case $model_path in
                *PORTULAN/gervasio-70b-*-merged|*meta-llama/Llama-3.3-70B-Instruct*)
                    vllm_args+=",max_model_len=4096,tensor_parallel_size=4,data_parallel_size=2"
                    lm_eval_args+=" --fewshot_as_multiturn"
                    lm_eval_args+=" --apply_chat_template"
                    ;;
                *PORTULAN/gervasio-8b-*-merged|*meta-llama/Llama-3.1-8B-Instruct*)
                    vllm_args+=",max_model_len=4096,tensor_parallel_size=4,data_parallel_size=2"
                    lm_eval_args+=" --fewshot_as_multiturn"
                    lm_eval_args+=" --apply_chat_template"
                    ;;
                *maritaca-ai/sabia-7b*)
                    vllm_args+=",max_model_len=2048,tensor_parallel_size=4,data_parallel_size=2"
                    ;;
                *deepseek-ai/deepseek-coder-1.3b-instruct*)
                    vllm_args+=",max_model_len=4096,tensor_parallel_size=2,data_parallel_size=5"
                    lm_eval_args+=" --fewshot_as_multiturn"
                    lm_eval_args+=" --apply_chat_template"
                    ;;
                *mistralai/Mistral-Small-24B-Instruct-2501*)
                    vllm_args+=",max_model_len=4096,tensor_parallel_size=4,data_parallel_size=2"
                    lm_eval_args+=" --fewshot_as_multiturn"
                    lm_eval_args+=" --apply_chat_template"
                    ;;
                *Qwen/Qwen3-32B*)
                    vllm_args+=",max_model_len=4096,tensor_parallel_size=4,data_parallel_size=2"
                    lm_eval_args+=" --fewshot_as_multiturn"
                    lm_eval_args+=" --apply_chat_template"
                    ;;
            esac

            cmd="lm-eval --model vllm --model_args $vllm_args $lm_eval_args"
            echo $cmd > $logfile
            $cmd >> $logfile 2>&1
        done
    done
done
