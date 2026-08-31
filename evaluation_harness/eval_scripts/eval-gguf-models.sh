#!/usr/bin/env bash

set -euo pipefail

model_paths=""
tasks=""
seeds=""

# Pick the GGUF model(s) you want to evaluate:
model_paths+=" $(readlink -f ../models/PORTULAN/gervasio-8b-portuguese-ptpt-decoder-quantized-4bit/gervasio-8b-portuguese-ptpt-decoder-Q4_K.gguf)"

run_main_tasks=true
run_optional_tasks=true

if $run_main_tasks; then
    tasks+=" gervasio-extraglue_copa-ptpt"
    tasks+=" gervasio-extraglue_rte-ptpt"
    tasks+=" gervasio-meta_gpqa-diamond"
    tasks+=" gervasio-extraglue_mrpc-ptpt"
    tasks+=" gervasio-meta_mmlu"
    tasks+=" gervasio-meta_mmlu-pro"
fi

if $run_optional_tasks; then
    tasks+=" gervasio-meta_mbppplus"
    tasks+=" gervasio-meta_mgsm"
    tasks+=" gervasio-meta_humaneval"
    tasks+=" gervasio-meta_math"
fi

seeds="2345"

cd "../decoder-eval"

export RAY_NUM_CPUS=40
export OMP_NUM_THREADS=2

port=8008
base_url="http://localhost:$port"

for seed in $seeds; do
    for model_path in $model_paths; do
        if ! test -f "$model_path"; then
            echo "Model file not found: $model_path"
            exit 1
        fi

        eval_dir="$(dirname "$model_path")/eval"
        output_dir="$eval_dir/outputs"
        mkdir -p "$output_dir"

        echo "Starting llama-cpp server for $model_path..."
        python3 -m llama_cpp.server --model "$model_path" --port $port > "$eval_dir/server.log" 2>&1 &
        server_pid=$!

        echo "Waiting for server to start..."
        until curl -s "$base_url/v1/completions" -o /dev/null; do
            sleep 1
        done
        echo "Server is up."

        for task in $tasks; do
            logfile="$eval_dir/eval-$task-seed$seed-$(date +'%Y-%m-%d_%H-%M-%S').log"

            lm_eval_args="--include_path ../task_configurations"
            lm_eval_args+=" --tasks $task"
            lm_eval_args+=" --output_path $output_dir"
            lm_eval_args+=" --confirm_run_unsafe_code"
            lm_eval_args+=" --batch_size auto"
            lm_eval_args+=" --log_samples"
            lm_eval_args+=" --seed $seed"

            model_args="base_url=$base_url"
            cmd="lm-eval --model gguf --model_args $model_args $lm_eval_args"

            echo "Running: $cmd" | tee "$logfile"
            $cmd >> "$logfile" 2>&1
        done

        echo "Shutting down server for $model_path..."
        kill $server_pid
        wait $server_pid 2>/dev/null || true
    done
done
