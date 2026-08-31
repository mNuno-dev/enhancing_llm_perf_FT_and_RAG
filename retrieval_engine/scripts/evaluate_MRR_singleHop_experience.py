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
mlflow.set_experiment("TEST_evaluate_MRR_exp1")


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
 

    # parser.add_argument(
    #     "model_list",
    #     type=ast.literal_eval,
    #     help="List of model names to be evaluated, each model has a folder inside retrieval_evaluation/results"
    # )

    # parser.add_argument(
    #     "output_dir",
    #     type=str,
    #     help="The output directory for the MRR results to be logged and saved."
    # )

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
    # print(f"model_list: {args.model_list}")
    # print(f"output_dir: {args.output_dir}")

    ####                                                  ####
    #### ---- Getting model dir_paths for evaluation ---- ####
    ####                                                  ####
    results_files = os.listdir(args.retrieval_results_dir)

    print(results_files)

#     model_saved_results_files = {model_name:[] for model_name in args.model_list}

    # WORKS!
    overall_results_on_dir = []
    for file in results_files:
        results = load_results(os.path.join(args.retrieval_results_dir, file))
        top_k = len(results["multihop"][0]["relevant_chunks"])
        with mlflow.start_run(run_name=f"EVAL-{file}") as run:
            mlflow.log_param("top_k", top_k)

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
            mlflow.log_metric(f"MRR-at-{top_k}", MRR_score)
            overall_results_on_dir.append({"file" : file, "top_K": top_k, f"MRR@{top_k}": MRR_score})
    
    # Saving the results for every model + chunkstrat
    save_file_path = args.retrieval_results_dir + "/" + f"RESULTS_MRR@{top_k}.jsonl"
    with open(save_file_path, "w", encoding="UTF-8") as f:
        for entry in overall_results_on_dir:
            f.write(json.dumps(entry) + "\n") # for jsonl
    
        f.close()
    print(f"Saved MRR results locally at --> {save_file_path}")
    print("ALSO")
    mlflow_ui_url = "http://127.0.0.1:5000"
    print(f"Saved results in a MLflow local, you can visualize results at: \n ---> {mlflow_ui_url}")


                
if __name__ == "__main__":
    main()


# def get_topK():
#     ...



