#! /bin/bash

if test $# != 1; then
    echo "usage: MODEL_PATH" >&2
    exit 2
fi


MODEL_PATH=$(readlink -f $1)
if test -f $MODEL_PATH; then
    MODEL_DIR=$(dirname $MODEL_PATH)
    MODEL_NAME=$(basename $MODEL_PATH)
elif test -d $MODEL_PATH; then
    MODEL_DIR=$MODEL_PATH
    MODEL_NAME=$(basename $MODEL_PATH)
else
    echo "$MODEL_PATH is not a directory nor a file; aborting!" >&2
    exit 2
fi

set -x

# see https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md
podman run --rm --volume "$MODEL_DIR:/model:rw" \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --convert /model --model-name $MODEL_NAME --split-max-size 5G

