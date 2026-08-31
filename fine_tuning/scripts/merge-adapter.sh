#! /usr/bin/bash
set -e  # abort on error

source venv/bin/activate

if test $# != 1; then
    echo "usage: $0 ADAPTER_DIR" >&2
    exit 2
fi

adapter_dir=$1
base_model=$(grep base_model_name_or_path "$adapter_dir/adapter_config.json" | cut -f 2 -d ":" | tr -d '", ')
base_model_dir="$(realpath $(dirname $adapter_dir)/../../$base_model)"

echo "Base model: $base_model" >&2
if ! test -d "$base_model_dir"; then
    echo "Base model directory does not exist: $base_model_dir">&2
    exit 1
fi

dest_dir="$adapter_dir-merged"
if test -d $dest_dir; then
    echo "Merged adapter already exists in $dest_dir" >&2
else
    gervasio merge-lora-adapter "$base_model_dir" "$adapter_dir" "$dest_dir"
done
