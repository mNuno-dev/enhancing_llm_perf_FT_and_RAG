# Fine-Tuning

> **Note:** The `gervasio` package source code (`src/`, `tests/`, and related config files) was developed primarily by Luís M. S. Gomes at the [NLX research group](https://nlx.di.fc.ul.pt/) and remains within the research group. This folder contains the training configurations and SLURM scripts used in the thesis as a record of the experimental setup.

`gervasio` is a Python package for LoRA fine-tuning and FSDP distributed training of large language models, with support for checkpoint management, instruction packing, and adapter merging. It was used to fine-tune the Gervásio model family (1B, 3B, 8B) and also futher fine-tuning on the PTwikiRAG retrieval corpus.

## Contents

- `train_configs/` — TOML configuration files for each fine-tuning run, organised by run ID. See [`train_configs/README.md`](train_configs/README.md) for details on path conventions.
- `scripts/` — SLURM job scripts used to launch training on the HPC cluster. Each script corresponds to a run ID in `train_configs/`.
