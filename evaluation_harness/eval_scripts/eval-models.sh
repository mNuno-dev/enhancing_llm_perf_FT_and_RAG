#! /usr/bin/bash
# Evaluation script supporting vLLM-based models and local completions servers.
# Model paths can be local file/directory paths or completions server URLs
# (see the NOTE below for the expected URL format).

model_paths=""
tasks=""
seeds=""


# NOTE: model_paths contains a list of models to be evaluated in one of three forms
#   1- path to a GGUF file
#   2- path to a directory containing a model in either pytorch or safetensors format
#   3- a triple joined with hash chars (#) consisting of a completions URL, the eval output dir and the tokenizer name
#        example: http://192.168.100.1:15010/v1/completions#../models/PORTULAN/gervasio-8b-portuguese-ptpt-decoder-evaristo#meta-llama/Llama-3.1-8B-Instruct

#####                    #####
#  Local Completions Models  #
#####                    #####

# model_paths+=" http://192.168.100.1:15010/v1/completions#../models/PORTULAN/gervasio-8b-portuguese-ptpt-decoder-evaristo#meta-llama/Llama-3.1-8B-Instruct"

# model_paths+=" http://192.168.100.1:15009/v1/completions#../models/PORTULAN/gervasio-8b-portuguese-ptpt-decoder-evaristo#meta-llama/Llama-3.1-8B-Instruct"

# NOTE: Path below pointed to a local completions server on the development LAN. Replace with your own server address.
# model_paths+=" http://192.168.100.1:15029/v1/completions#../models/Qwen32B_test#Qwen/Qwen3-32B"


# Thesis fine-tuned models (paths are environment-specific)

# model_paths+=" $(readlink -f /home/nmarques/thesisModels/gervasio-3b-301/gervasio-3b-301-alpha64-merged)"
# model_paths+=" $(readlink -f /home/nmarques/thesisModels/gervasio-3b-301/gervasio-3b-301-alpha128-merged)"


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
# model_paths+=" $(readlink -f ../models/mistralai/Mistral-Small-24B-Instruct-2501)"
# model_paths+=" $(readlink -f ../models/deepseek-ai/deepseek-coder-1.3b-instruct)"

run_main_tasks=true
run_optional_tasks=true
gpu_memory_utilization=0.2


if $run_main_tasks; then
    # 6 min
    tasks+=" gervasio-extraglue_copa-ptpt"

    # 7 min
    # tasks+=" gervasio-extraglue_rte-ptpt"

    # 17 min
    # tasks+=" gervasio-meta_gpqa-diamond"

    # 20 min
    # tasks+=" gervasio-extraglue_mrpc-ptpt"

    # 5h22m
    # tasks+=" gervasio-meta_mmlu"

    # 15h
    # tasks+=" gervasio-meta_mmlu-pro"
fi

if $run_optional_tasks; then
    # 7 min
    # tasks+=" gervasio-meta_mbppplus"

    # 8 min
    # tasks+=" gervasio-meta_mgsm"

    # 10 min
    # tasks+=" gervasio-meta_humaneval"

    # tasks+=" gervasio-meta_math"
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
        elif test -d $model_path; then
            eval_dir="$model_path/eval"
            echo "$model_path is a directory; eval output will be saved in ./eval sub-directory" >&2
        elif [[ "$model_path" =~ ^https?://[^#]+#[^#]+#[^#]+ ]]; then
            echo "extracting URL, eval_dir and tokenizer_name from $model_path" >&2

            eval_dir="${model_path#*#}"     # get part after '#'
            tokenizer_name="${eval_dir#*#}" # get part after '#'

            eval_dir="${eval_dir%%#*}"      # strip everything from '#' onward
            model_path="${model_path%%#*}"  # strip everything from '#' onward
            echo "completions URL: $model_path" >&2
            echo "eval output dir: $eval_dir" >&2
            echo "tokenizer: $tokenizer_name" >&2
        else
            echo "$model_path is not a file, nor a directory, nor a URL#local_output_path" >&2
            exit 1
        fi

        output_dir="$eval_dir/outputs"
        mkdir -p "$output_dir"


        for task in $tasks; do
            logfile="$eval_dir/eval-$task-seed$seed-$(date +'%Y-%m-%d_%H-%M-%S').log"

            lm_eval_args="--include_path task_configurations"
            lm_eval_args+=" --tasks $task"
            lm_eval_args+=" --output_path $output_dir"
            lm_eval_args+=" --confirm_run_unsafe_code"
            lm_eval_args+=" --log_samples"
            lm_eval_args+=" --seed $seed"
            lm_eval_args+=" --apply_chat_template"

            if test -d $model_path || test -f $model_path; then
                model_type="vllm"

                vllm_args="pretrained=$model_path"
                vllm_args+=",task=generate"
                vllm_args+=",dtype=auto"
                vllm_args+=",gpu_memory_utilization=$gpu_memory_utilization"

                lm_eval_args+=" --batch_size auto"

            elif [[ "$model_path" =~ ^https?:// ]]; then
                model_type="local-completions"

                lc_args="base_url=$model_path"
                lc_args+=",model=$tokenizer_name"
                lc_args+=",task=generate"
                lc_args+=",num_concurrent=1"
                lc_args+=",max_retries=3"
                lc_args+=",tokenized_requests=False"

            else
                echo "invalid model: not a file or directory or HTTP(S) URL: $model_path"
            fi
            
            case $model_path in
                # Thesis fine-tuned 3B models
                /home/nmarques/thesisModels/gervasio-3b-*/gervasio-3b-301-*-merged)
                    vllm_args+=",max_model_len=2048,tensor_parallel_size=4,data_parallel_size=2"
                    lm_eval_args+=" --fewshot_as_multiturn"
                    lm_eval_args+=" --apply_chat_template"
                    ;;
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

            case "$model_type" in
                vllm)
                model_args=$vllm_args
                ;;
                local-completions)
                model_args=$lc_args
                ;;
                *)
                echo "unexpected model type: $model_type" >&2
                exit 1
                ;;
            esac

            cmd="lm-eval --model $model_type --model_args $model_args $lm_eval_args"
            echo $cmd > $logfile
            $cmd >> $logfile 2>&1
        done
    done
done
