"""
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

HF_TOKEN = os.getenv("HF_TOKEN", "")  # set HF_TOKEN as an environment variable


# Setting the MLflow experiment for logs
# 🟢 Always set the experiment at the top
mlflow.set_experiment("retrieval-eval-experiments")


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

    ####                                                  ####
    #### ---- Getting model dir_paths for evaluation ---- ####
    ####                                                  ####


    model_saved_results_files = {model_name:[] for model_name in args.model_list}

    # WORKS!
    for model_name in args.model_list:
        eval_path = args.retrieval_results_dir + "/" + model_name
        files = os.listdir(args.retrieval_results_dir + "/" + model_name)
        model_saved_results_files[model_name].extend([os.path.join(eval_path,file) for file in files])
    
    # # For testing
    # print(model_saved_results_files)
    score_results = {}
    for model_name in args.model_list:
        # dir_list = model_saved_results_files[model_name]
        dir_list = [x for x in model_saved_results_files[model_name] if "top3" in x]
        for path in dir_list:
            results = load_results(path)
            top_k = len(results["multihop"][0]["relevant_chunks"])
            with mlflow.start_run(run_name=f"MRR@{top_k} for model {model_name}") as run:
                print("Run ID:", run.info.run_id)
                run_id = run.info.run_id
                mlflow.log_param("run_id", run_id)
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("top_k", top_k)
                mlflow.log_param("chunking_strat", os.path.basename(path))

                RR_scores = []
                for index, entry in enumerate(results["not_multihop"]):
                    if entry["is_multi_hop"] == "INTER":
                        l_cit = entry["citations"].split(" <SEP> ")
                        # gt = [cit for cit in l_cit]
                        # break
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
                mlflow.log_metric("MRR_score", MRR_score)

                # print("Attention!!")
                # print("Attention!!")
                # print()
                # print()
                # print(f"Performance on file {os.path.basename(path)}")
                # print(f"MRR_@{top_k}: {MRR_score}")
                # print()
                # print()
                # print("Attention!!")
                # print("Attention!!")

                # time.sleep(5)
                score_results[os.path.basename(path)] = {f"MRR@{top_k}": MRR_score}
        print(score_results)
    






if __name__ == "__main__":
    main()


# def get_topK():
#     ...



