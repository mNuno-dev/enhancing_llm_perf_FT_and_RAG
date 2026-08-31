#! /bin/bash

if test $# != 2; then
    echo "usage: GGUF_PATH QUANTIZATION_TYPE" >&2
    echo "for QUANTIZATION_TYPE see https://huggingface.co/docs/hub/gguf" >&2
    echo "commonly used QUANTIZATION_TYPE: Q4_K" >&2
    exit 2
fi

GGUF_PATH=$(readlink -f $1)
GGUF_DIR=$(dirname $GGUF_PATH)
GGUF_BASENAME=$(basename $GGUF_PATH)

QUANTIZED_GGUF_BASENAME=$(echo $GGUF_BASENAME | sed 's/-f16.gguf$//i')-$2.gguf


set -x

# see https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md
podman run --rm --volume "$GGUF_DIR:/workdir:rw" \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --quantize /workdir/$GGUF_BASENAME $2

mv $GGUF_DIR/ggml-model-$2.gguf $GGUF_DIR/$QUANTIZED_GGUF_BASENAME

