#! /bin/bash

if test $# != 1; then
    echo "usage: MODEL_PATH" >&2
    exit 2
fi

MODEL_DIR=$(dirname $1)
MODEL_BASENAME=$(basename $1)
SPLIT_MODEL_BASENAME=$(basename $1 .gguf)-split.gguf

set -x

# see https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md
podman run --rm --volume "$MODEL_DIR:/model:rw" --entrypoint ./llama-gguf-split \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --split-max-size 5G /model/$MODEL_BASENAME /model/$SPLIT_MODEL_BASENAME

