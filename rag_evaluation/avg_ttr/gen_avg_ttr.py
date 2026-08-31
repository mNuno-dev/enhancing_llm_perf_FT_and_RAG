import os
import json
import argparse


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Compute the average time-to-response (TTR) across generation result "
            "files in a given directory and write a summary report."
        )
    )
    ap.add_argument(
        "--files-dir",
        required=True,
        help=(
            "Directory containing the generation result JSONL files to average. "
            "Example: /path/to/generation_results/Gervasio8B/"
            "fin_hnsw_PTwikiRAG_meanPooling_hybrid0.4/Qwen/Qwen3-Embedding-8B/top3"
        ),
    )
    ap.add_argument(
        "--model-name",
        required=True,
        help="Model label used as the output subdirectory (e.g. Gervasio8B, Llama1B).",
    )
    ap.add_argument(
        "--ret-type",
        required=True,
        help=(
            "Retrieval configuration label used in the output filename "
            "(e.g. PTwikiRAG_meanPooling_hybrid0.4, PTwikiRAG-Pro_meanPooling_hybrid0.3)."
        ),
    )
    ap.add_argument(
        "--emb-model",
        default="Qwen3-Embedding-8B",
        help="Embedding model name written into the report header (default: Qwen3-Embedding-8B).",
    )
    ap.add_argument(
        "--top-k",
        required=True,
        help="Top-k retrieval setting written into the output filename (e.g. top1, top3, top5).",
    )
    ap.add_argument(
        "--out-dir",
        default="ttr_results",
        help="Base output directory for TTR reports (default: ttr_results).",
    )
    args = ap.parse_args()

    files_dir = args.files_dir
    model_name = args.model_name
    ret_type = args.ret_type
    emb_model = args.emb_model
    top_k = args.top_k

    # Example --files-dir values (PTwikiRAG and PTwikiRAG-Pro variants):
    #   /path/to/generation_results/Gervasio1B/fin_hnsw_PTwikiRAG_meanPooling_hybrid0.4/Qwen/Qwen3-Embedding-8B/top1
    #   /path/to/generation_results/Gervasio8B/fin_hnsw_PTwikiRAG-Pro_meanPooling_hybrid0.3/Qwen/Qwen3-Embedding-8B/top5
    #   /path/to/generation_results/Llama3B/fin_hnsw_PTwikiRAG_meanPooling_hybrid0.5/Qwen/Qwen3-Embedding-8B/top5

    list_f_dirs = [os.path.join(files_dir, f) for f in os.listdir(files_dir)]
    print(list_f_dirs)

    final_res = {}

    for file in list_f_dirs:
        total_time = 0.0
        count = 0

        with open(file, "r", encoding="UTF-8") as f:
            for line in f:
                entry = json.loads(line)
                total_time += entry["tt_response"]
                count += 1

        final_res[file] = total_time / count

    out_path = os.path.join(args.out_dir, model_name)
    os.makedirs(out_path, exist_ok=True)

    file_name = ret_type + "_" + top_k + ".txt"
    final_path = os.path.join(out_path, file_name)

    with open(final_path, "w", encoding="UTF-8") as out_file:
        out_file.write(f"Embedding Model = Qwen/{emb_model}")

        for key, value in final_res.items():
            out_file.write("\n\n---------- // ----------\n")
            out_file.write(
                f"Chunk-Strategy --> {os.path.basename(key).replace('Qwenqwen3embedding8b', '')}"
            )
            out_file.write("\n")
            out_file.write(f"Avg Time to Response --> {value}")
            out_file.write("\n---------- // ----------\n")


if __name__ == "__main__":
    main()
