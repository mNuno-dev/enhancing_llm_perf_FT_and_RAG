# Gervasio

Gervasio is a Python package for advanced model training and evaluation tooling, designed for efficient fine-tuning of large language models. It provides comprehensive infrastructure for LoRA (Low-Rank Adaptation) and FSDP (Fully Sharded Data Parallel) training, along with custom checkpoint management and model utilities.

## Key Features

- **Advanced Instruction Packing**: Sophisticated dataset collation that dynamically packs instruction-tuning examples to minimize padding, significantly improving training efficiency and reducing computational waste
- **LoRA Fine-tuning**: Full support for parameter-efficient fine-tuning with configurable LoRA parameters (rank, alpha, etc.)
- **FSDP Training**: Distributed training capabilities using Fully Sharded Data Parallel for handling large models across multiple GPUs
- **Checkpoint Management**: Comprehensive tools for inspecting, comparing, and managing training checkpoints, including continuation generation during checkpointing
- **Model Utilities**: Commands for downloading models from Hugging Face, comparing model files, and merging LoRA adapters with base models

## Installation

Use the following command to install gervasio and dependencies:


    pip install -e . --find-links https://download.pytorch.org/whl/cu121



## Usage

**Note:** All relative paths in configuration files and command-line arguments are interpreted relative to the current working directory where the program is executed.

### Training

Train a model using LoRA fine-tuning with a configuration file:

    gervasio-train --train-config example-train-config.toml

Train with custom parameters:

    gervasio-train \
        --datasets-dir /path/to/datasets \
        --datasets-cache-dir /path/to/cache \
        --datasets-aliases dataset1 dataset2 \
        --batch-max-examples 1000 \
        --batch-max-tokens 2048 \
        --base-model-name meta-llama/Llama-2-7b-hf \
        --adapter-name my_adapter \
        --output-dir /path/to/output \
        --num-epochs 2 \
        --lr 2e-5

Enable generation of continuations during checkpointing:

    gervasio-train \
        --train-config example-train-config.toml \
        --prompts-file prompts.jsonl \
        --generation-config example-gen-config.toml

### Plotting Training Loss

Plot the average loss from distributed training logs:

    gervasio-train-plot-loss /path/to/output/dir --num-ranks 4 --smooth 100

### Checkpoint Inspection

Inspect the structure and size of a checkpoint file:

    gervasio-checkpoint-inspector /path/to/checkpoint.pt

### Comparing Checkpoints

Compare two checkpoint files for differences:

    gervasio-compare-checkpoints checkpoint1.pt checkpoint2.pt --atol 1e-6 --rtol 1e-5

### Comparing Models

Compare two model files (safetensors or torch format):

    gervasio-compare-models model1.safetensors model2.safetensors --rtol 1e-5 --atol 1e-8 --top-k 10

### Downloading Models

Download a model from Hugging Face Hub:

    gervasio-download-model meta-llama/Llama-2-7b-hf /local/path/to/model

### Merging LoRA Adapters

Merge a LoRA adapter with its base model:

    gervasio-merge-lora-adapter my_adapter --base-model-name meta-llama/Llama-2-7b-hf --merged-model-name my_merged_model

Merge with quantization:

    gervasio-merge-lora-adapter my_adapter --quantization 4bit


## Author

Luís M. S. Gomes <lmdgomes@fc.ul.pt>, <luismsgomes@gmail.com>

## Copyright

Copyright 2025 NLX group, Faculdade de Ciências da Universidade de Lisboa


