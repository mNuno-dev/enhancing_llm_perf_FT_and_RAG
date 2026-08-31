import os
import asyncio
import datetime

# pip install langchain-openai
from langchain_openai import ChatOpenAI

from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness_PT, AnswerRelevancy_PT, ContextRelevance_PT

from ragas import EvaluationDataset, evaluate
from ragas import RunConfig


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")  # set via environment variable
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")  # set via environment variable
os.environ["HF_API_TOKEN"] = os.getenv("HF_API_TOKEN", "")  # set via environment variable


async def main():
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
        base_url="http://localhost:25012/v1",
        api_key="not-needed"
    ))

    faithfu_metric_PT = Faithfulness_PT(name="faithfulness_PT", llm=evaluator_llm)
    ans_rel_metric_PT = AnswerRelevancy_PT(name="answer_relevance_PT", llm=evaluator_llm)
    context_rel_metric_PT = ContextRelevance_PT(name="context_relevance_PT", llm=evaluator_llm)

    # RunConfig can be used to control concurrency and retry behaviour:
    # run_cfg = RunConfig(
    #     max_workers=1,        # limit concurrent requests
    #     max_retries=12,       # retry on 429 rate limits
    #     max_wait=3,           # exponential backoff upper bound (seconds)
    #     timeout=180,
    #     log_tenacity=True
    # )

    # Set this to the directory containing the generation result JSONL files.
    # Path format: /path/to/your/generation_results/<model>/<retrieval_config>/<emb_model>/<top_k>
    # Example: /path/to/your/generation_results/Gervasio3B/fin_hnsw_PTwikiRAG-Pro_meanPooling_hybrid0.3/google/embeddinggemma-300m/top5
    og_path = "/path/to/your/generation_results/<model>/<retrieval_config>/<emb_model>/<top_k>"

    list_f_dirs = [os.path.join(og_path, file) for file in os.listdir(og_path)]

    for gen_result_file in list_f_dirs:

        print()
        print(f"Using Generation file --> {gen_result_file}")
        print()

        eval_dataset = EvaluationDataset.from_jsonl(gen_result_file)

        print("------- REPORT: -------\n")
        print("Features in dataset:", eval_dataset.features())
        print("Total samples in dataset:", len(eval_dataset))
        print("-----------------------\n")

        results = evaluate(eval_dataset, metrics=[faithfu_metric_PT, ans_rel_metric_PT, context_rel_metric_PT])

        print("\n\n")
        print("------- Printing Results of RAGAS -------")
        print("///**start**///")
        print(f"Day --> {datetime.datetime.now()}")
        print(f"Generation file --> {os.path.basename(gen_result_file)}")
        print()
        print(f"Ragas Metrics: {results}")
        print()
        print("------- Printing Results of RAGAS -------")
        print("\n\n")

        log_dir = "ragas_results_logs"
        os.makedirs(log_dir, exist_ok=True)
        save_log_file_dir = os.path.join(log_dir, f"RAGAS_OUT_{datetime.datetime.now().date()}.txt")
        with open(save_log_file_dir, "a", encoding="UTF-8") as outfile:
            outfile.write("\n\n")
            outfile.write("------- Printing Results of RAGAS -------\n")
            outfile.write("///**start**///\n")
            outfile.write(f"Day --> {datetime.datetime.now()}\n")
            outfile.write(f"Full file dir --> {gen_result_file}\n")
            outfile.write(f"Generation file --> {os.path.basename(gen_result_file)}\n")
            outfile.write(f"Ragas Metrics: {results}\n")
            outfile.write("------- Printing Results of RAGAS -------\n")


asyncio.run(main())
