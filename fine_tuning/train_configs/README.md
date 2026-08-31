# Training Configurations

This directory contains the TOML configuration files for each fine-tuning run
conducted in the thesis.

Each subdirectory corresponds to one training run, identified by its run ID.
Run IDs map to the SLURM job scripts in `../scripts/`.

## Path conventions

All configuration files contain cluster-specific paths (e.g. `datasets_dir`,
`base_model_name`, `output_dir`) that reference the HPC cluster filesystem
where the training was performed. These paths are preserved as a record of
the exact experimental setup and are not intended to be portable.

To reproduce a run, update `datasets_dir`, `datasets_cache_dir`,
`base_model_name`, and `output_dir` to match your local environment.

## Structure

```
train_configs/
├── example/                 ← annotated example configs for reference
├── gerv-1B-test/            ← initial 1B test runs
├── gerv-1b-RAGdocs/         ← 1B retrieval-corpus fine-tuning runs (0177701–0177726)
├── gerv-3b-RAGdocs/         ← 3B retrieval-corpus fine-tuning runs (0388802–0388809)
│   └── legacy/              ← earlier 3B run
└── gerv-8b-RAGdocs/         ← 8B retrieval-corpus fine-tuning runs (0877701–0877703)
```
