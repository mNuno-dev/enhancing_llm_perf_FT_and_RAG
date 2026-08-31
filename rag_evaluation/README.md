# RAG Evaluation

End-to-end RAG evaluation pipeline based on a customised fork of the
[RAGAS framework](https://github.com/explodinggradients/ragas), adapted for
European Portuguese with Portuguese-language prompts and an Answer Accuracy metric.

## Structure

```
rag_evaluation/
├── RAGAS_customized_for_PTPT/
│   ├── metrics_adapted_to_PTPT/    ← standalone copies of the PT-adapted metric files
│   └── ragas/                      ← RAGAS fork with PT metrics integrated into the package
├── auxiliary_scripts/
│   ├── add_reference.py            ← adds `reference` = `answer` to generation JSONL files
│   └── remove_vector.py            ← removes `vector` field from retrieval JSONL files
├── avg_ttr/
│   └── gen_avg_ttr.py              ← computes average time-to-response across generation runs
└── scripts/
    ├── RAGAS_full_dir_Gemini2.0.py                   ← 3-metric RAGAS eval (Gemini judge via local proxy)
    ├── RAGAS_AnswerCorrectness_full_dir_Gemini2.0.py ← adds Answer Accuracy (requires `reference` field)
    ├── RAGAS_full_dir_localHost.py                   ← 3-metric RAGAS eval (generic local LLM judge)
    └── legacy_scripts/   ← earlier approaches: BLEU, ROUGE, GPT-4, Groq
```

## Setup

Install the customised RAGAS fork (which includes the PT-adapted metrics):

```bash
pip install -e ./RAGAS_customized_for_PTPT/ragas/
```

Then install the remaining dependencies:

```bash
pip install langchain-openai langchain-groq google-generativeai datasets huggingface_hub
```

Set the following environment variables before running any evaluation script:

```bash
export OPENAI_API_KEY="..."    # required by langchain-openai
export GROQ_API_KEY="..."      # required by legacy Groq scripts
export HF_TOKEN="..."          # required for HuggingFace Hub access
```

## Custom Portuguese Metrics

The PT-adapted metrics are integrated into the RAGAS fork and imported via:

```python
from ragas.metrics import (
    Faithfulness_PT,
    AnswerRelevancy_PT,
    ContextRelevance_PT,
    AnswerAccuracy_PT,
    AnswerCorrectness_PT,
)
```

Standalone copies of the metric source files are in
`RAGAS_customized_for_PTPT/metrics_adapted_to_PTPT/` for reference.

## Running an Evaluation

1. Populate `p_list` in the chosen script with paths to your generation result directories.
2. Run the script:

```bash
python scripts/RAGAS_AnswerCorrectness_full_dir_Gemini2.0.py
```

Results are written to `ragas_results_logs/`.

## Auxiliary Scripts

**`auxiliary_scripts/add_reference.py`** — adds a `reference` field (copied from `answer`) to
each JSONL entry. Required before running `RAGAS_AnswerCorrectness_full_dir_Gemini2.0.py`.

```bash
python auxiliary_scripts/add_reference.py --src /path/to/gen_results
```

**`auxiliary_scripts/remove_vector.py`** — strips the `vector` field from JSONL files produced
by the retrieval engine before evaluation.

```bash
python auxiliary_scripts/remove_vector.py --src /path/to/gen_results
```
