# Evaluation Harness

Custom additions on top of [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
used to evaluate the Gervásio model family on European-Portuguese benchmarks.

Only the thesis-specific files are included here. The full upstream harness
is a separate dependency.

## Setup

Install the upstream lm-evaluation-harness:

```bash
pip install lm-eval
```

## Contents

- `eval_scripts/` — shell scripts used to run evaluations on the HPC cluster.
  Paths in these scripts are VM-specific; adapt the `model_paths` variable to
  point to your local model directories before running.
- `task_configurations/` — the European-Portuguese task configurations used in
  the thesis were loaded from this directory. The configurations were developed
  by the NLX research group and remain within the research group; see
  `task_configurations/README.md` for details.

## Running

To evaluate using the same benchmarks, submit your model to the HuggingFace leaderboard:

**[PORTULAN/portuguese-llm-leaderboard](https://huggingface.co/spaces/PORTULAN/portuguese-llm-leaderboard)**

To run the eval scripts directly (once task configurations are in place):

```bash
lm-eval --model vllm \
  --model_args pretrained=/path/to/model \
  --tasks portuguese_bench \
  --include_path task_configurations \
  --output_path ./results
```

See `eval_scripts/eval-thesis-models.sh` for the full evaluation setup used
in the thesis.
