# Thesis Code: Enhancing LLM Performance through Fine-Tuning and RAG

This repository contains the code that supported the Master's thesis
**"Enhancing LLM Performance through Fine-Tuning and RAG"**,
which investigates European-Portuguese language adaptation and
Retrieval-Augmented Generation (RAG) across three model scales (1B, 3B, 8B).

## Repository Structure

| Folder | Description | Key Dependencies |
|--------|-------------|------------------|
| [`fine_tuning/`](fine_tuning/) | `gervasio` Python package for LoRA + FSDP fine-tuning of Llama 3 models. Includes training configs and SLURM scripts used on the HPC cluster. | `peft`, `transformers`, `accelerate`, `torch` |
| [`retrieval_engine/`](retrieval_engine/) | RAG retrieval pipeline: chunking strategies (DPC, LumberChunker, sentence-level), embedding model evaluation, hybrid BM25+dense retrieval, and Weaviate-based indexing. | `weaviate-client`, `sentence-transformers` |
| [`rag_evaluation/`](rag_evaluation/) | End-to-end RAG evaluation using a customised [RAGAS framework](https://github.com/vibrantlabsai/ragas) with European-Portuguese prompts and an Answer Accuracy metric. Includes evaluation scripts, preprocessing utilities, and custom Portuguese metrics. | `ragas`, `google-generativeai` |
| [`evaluation_harness/`](evaluation_harness/) | Custom additions on top of [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness): evaluation shell scripts and Portuguese NLP benchmark task definitions. | `lm-eval` |

## Models

The Gervásio model family (1B, 3B, 8B) trained in this thesis will be released
on HuggingFace under the PORTULAN/Gervásio line.

## Dataset

The PTwikiRAG benchmark (92 documents, 1006 QA pairs) will be released on
HuggingFace as `nunoFcul/PTwikiRAG`.

## Notes

- Model weights are not included in this repository.
- The full `lm-evaluation-harness` and `ragas` libraries are upstream dependencies;
  only thesis-specific additions are included here.
- API keys and tokens are not included. Set `OPENAI_API_KEY`, `GROQ_API_KEY`,
  and `HF_TOKEN` as environment variables before running evaluation scripts.
- Each component was developed and run independently on separate servers, selected
  based on hardware requirements (HPC cluster and A100 GPU enabled GCP cloud VMs for fine-tuning, GPU VMs for retrieval
  and evaluation). Some components ran in parallel to make efficient use of compute
  resources and running costs.  
