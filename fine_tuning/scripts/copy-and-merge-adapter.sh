#! /usr/bin/bash
set -e  # abort on error

source venv/bin/activate

if test $# != 3; then
    echo "usage: $0 OUTPUT_DIR ATTEMPT_NUMBER MODELS_DIR" >&2
    exit 2
fi

output_dir=$1
attempt_number=$2
models_dir=$3

source_dir="$output_dir/attempt-${attempt_number}"
base_model=$(grep base_model_name_or_path "$source_dir/adapter_config.json" | cut -f 2 -d ":" | tr -d '", ')

if test "$base_model" == 'meta-llama/Llama-3.2-1B-Instruct'; then
    model_size=1b
elif test "$base_model" == 'meta-llama/Llama-3.2-3B-Instruct'; then
    model_size=3b
elif test "$base_model" == 'meta-llama/Llama-3.1-8B-Instruct'; then
    model_size=8b
elif test "$base_model" == 'meta-llama/Llama-3.3-70B-Instruct'; then
    model_size=70b
else
    echo "Expected base_model_name_or_path to be one of:" >&2
    echo "meta-llama/Llama-3.2-1B-Instruct" >&2
    echo "meta-llama/Llama-3.2-3B-Instruct" >&2
    echo "meta-llama/Llama-3.1-8B-Instruct" >&2
    echo "meta-llama/Llama-3.3-70B-Instruct" >&2
    echo "But base_model is $base_model" >&2
    exit 1
fi

echo "Base model size: $model_size" >&2

alpha=$(grep lora_alpha "$source_dir/adapter_config.json" | cut -f 2 -d ":" | tr -d '", ')

echo "LoRA alpha is $alpha" >&2
echo "Creating alternate config with alpha=${half_alpha}" >&2
alpha2=$((alpha / 2))

dest_dir1="$models_dir/PORTULAN/gervasio-${model_size}-${attempt_number}-alpha${alpha}"
dest_dir2="$models_dir/PORTULAN/gervasio-${model_size}-${attempt_number}-alpha${alpha2}"

for dest_dir in "$dest_dir1" "$dest_dir2"; do
    if test -d $dest_dir; then
        echo "Model already copied from $source_dir into $dest_dir" >&2
    else
        echo "Copying model from $source_dir into $dest_dir" >&2
        cp -r "$source_dir" "$dest_dir"
    fi
done

sed -i "s/\"lora_alpha\": $alpha,/\"lora_alpha\": $alpha2,/" $dest_dir2/adapter_config.json

for dest_dir in "$dest_dir1" "$dest_dir2"; do
    if test -d "${dest_dir}-merged"; then
        echo "Adapter from ${dest_dir} already merged into ${dest_dir}-merged" >&2
    else
        echo "Merging adapter from ${dest_dir} into ${dest_dir}-merged" >&2
        gervasio merge-lora-adapter "$models_dir/$base_model" "$dest_dir" "${dest_dir}-merged"
    fi
done

