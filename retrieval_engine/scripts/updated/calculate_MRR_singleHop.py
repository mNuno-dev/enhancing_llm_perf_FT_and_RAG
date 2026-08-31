"""
python3 scripts/NEW/calculate_MRR_singleHop.py \
    /path/to/retrieval_evaluation/results \
    "['BAAI/bge-m3','PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir']" \
    /path/to/retrieval_evaluation/metrics

"""
import time
import os
import sys
# import retrieval_evaluation as metrics
import argparse
import ast
import json
from utils import loading_PT_wikiRAG_from_HF
import mlflow

from utils import save_results_csv, append_to_google_sheet

def load_results(path:str):
    multihop = []
    not_multihop = []
    with open(path, "r", encoding="UTF-8") as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            if data["is_multi_hop"] in ("FALSE","INTER"):
                not_multihop.append(data)
            elif data["is_multi_hop"] == "TRUE":
                multihop.append(data)
            else:
                print(data)
                raise ValueError(f'Unexpected value in x["is_multi_hop"] --> ----{data["is_multi_hop"]}----')
    
    return {"multihop": multihop, "not_multihop": not_multihop}





def main():
    parser = argparse.ArgumentParser(description="Parsing the cli arguments for MRR metric calculation")

    parser.add_argument(
        "retrieval_results_dir",
        type=str,
        help="The directory where the .jsonl retrieval results are saved."
    )
 

    parser.add_argument(
        "model_list",
        type=ast.literal_eval,
        help="List of model names to be evaluated, each model has a folder inside retrieval_evaluation/results"
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="The output directory for the MRR results to be logged and saved."
    )

    # # Maybe for another script that can take different metrics as a parameter
    # parser.add_argument(
    #     "metric",
    #     choices=["MRR","Precision", "Recall", "MAP"],
    #     type=str,
    #     help=""
    # )

    args = parser.parse_args()

    print("Arguments received:")
    print(f"retrieval_results_dir: {args.retrieval_results_dir}")
    print(f"model_list: {args.model_list}")
    print(f"output_dir: {args.output_dir}")

    ####                                                      ####
    #### ---- Getting dir_paths per model for evaluation ---- ####
    ####                                                      ####
    
    model_saved_results_files = {model_name: [] for model_name in args.model_list}

    for model_name in args.model_list:
        model_path = os.path.join(args.retrieval_results_dir, model_name)
        entries = os.listdir(model_path)

        for entry in entries:
            full_path = os.path.join(model_path, entry)
            
            if os.path.isfile(full_path) and entry.lower().startswith("top") and entry.endswith(".jsonl"):
                # case: top{k} JSONL file directly in the model folder
                model_saved_results_files[model_name].append(full_path)

            elif os.path.isdir(full_path) and entry.lower().startswith("top"):
                # case: top{k} is a folder → get all JSONL files inside it
                for f in os.listdir(full_path):
                    if f.endswith(".jsonl"):
                        model_saved_results_files[model_name].append(os.path.join(full_path, f))

    print(model_saved_results_files)

    
    score_results = {model_name: {} for model_name in args.model_list}
    for model_name in args.model_list:
        dir_list = model_saved_results_files[model_name]  # all JSONL files
        for path in dir_list:
            print("Processing:", path)
            results = load_results(path)
            if results["multihop"]:
                top_k = len(results["multihop"][0]["relevant_chunks"])
            else:
                top_k = len(results["not_multihop"][0]["relevant_chunks"])
            
            RR_scores = []
            for index, entry in enumerate(results["not_multihop"]):
                if entry["is_multi_hop"] == "INTER":
                    l_cit = entry["citations"].split(" <SEP> ")
                    gt = l_cit
                else:
                    gt = []
                    gt.append(entry["citations"])
                
                rel_chunks = [x[0]["text"] for x in entry["relevant_chunks"]]

                # flag to see if we found a match
                found = False
                for idx, chunk in enumerate(rel_chunks):
                    if any(citation in chunk for citation in gt):
                        score = 1.0 / (idx + 1) # rank is 1-based for MRR evaluation
                        RR_scores.append(score)
                        found = True
                        break # stop at first relevant (MRR only cares about the first one)
                
                if not found:
                    RR_scores.append(0.0)
                
                MRR_score = sum(RR_scores) / len(RR_scores)
                # Save a dictionary with the result in score_results
                score_results[model_name][os.path.basename(path)+"_results_"+"MRR@"+str(top_k)] = {f"MRR@{top_k}":MRR_score}

    
    # Save locally (CSV,TSV)
    save_results_csv(score_results, model_list=args.model_list, output_dir=args.output_dir, output_file="retrieval_eval_results.csv", delimiter=",")


if __name__ == "__main__":
    main()







if __name__ == "__main__":
    main()
