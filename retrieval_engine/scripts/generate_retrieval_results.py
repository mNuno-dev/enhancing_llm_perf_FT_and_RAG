"""
Example of calling the script:
----- (required args) -----
python3 scripts/generate_retrieval_results.py 3 /path/to/retrieval_evaluation/results "['BAAI/bge-m3','PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir']" --batch_size 32

"""


import os
import argparse
import ast  # to safely parse list-like strings
import copy
import numpy as np
import tqdm
import json

from sentence_transformers import SentenceTransformer, util
import weaviate
import logging

from embeddings import generate_embeddings_w_model
from utils import make_alias, load_jsonl_PT_wikiRAG_locally, loading_PT_wikiRAG_from_HF, remove_special_characters
import weaviate_database as weaviate_utils

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

HF_TOKEN = os.getenv("HF_TOKEN", "")  # set HF_TOKEN as an environment variable

def main():
    parser = argparse.ArgumentParser()

    # Position Arguments (required)
    # parser.add_argument(
    #     "collection_names_list",
    #     type=ast.literal_eval, # allows safe parsing of a Python list from a string
    #     help="The list of names for vector index databases (collections) in the weaviate database."
    # )

    parser.add_argument(
        "top_k",
        type=int, 
        help="The number of relevant chunks to retrieve for each question."
    )

    parser.add_argument(
        "results_save_dir",
        type=str,
        default="/retrieval_evaluation/results",
        help="The path where the results will be saved"
    )

    parser.add_argument(
        "model_list",
        type=ast.literal_eval, # allows safe parsing of a Python list from a string
        help="List of models for which there is a collection to evaluate. Example: \"['model1','model2']\""
    )

    parser.add_argument(
        "--batch_size",
        type=int, # allows safe parsing of a Python list from a string
        default=32,
        help="Batch size for embedding generation function -> it is passed to the SentenceTransformer during embedding generation."
    )


    # parser.add_argument(
    #     "--q_embeddings_save_dir",
    #     type=str, 
    #     default=None,
    #     help="The directory where the pre-computed embeddings for every question in the subsplit2 is saved"
    # )

    args = parser.parse_args()

    # Argument parsing sanity checks
    print("Arguments received:")
    # print(f"collection_names_list: {args.collection_names_list}")
    print(f"results_save_dir: {args.results_save_dir}")
    print(f"top_k: {args.top_k}")
    print(f"model_list: {args.model_list}")
    print(f"batch_size: {args.batch_size}")
    # print(f"q_embeddings_save_dir: {args.q_embeddings_save_dir}")

    #####                                         #####
    #####   Getting the dataset subsplit2 q+a+c   #####
    #####                                         #####
    dataset = loading_PT_wikiRAG_from_HF(token=HF_TOKEN)
    qac_data = dataset["subsplit2"]


    # Create a list with all the questions in the dataset subsplit2
    q_list = [q["question"] for q in qac_data]


    #####                                                     #####
    #####   Getting the collection names for a certain model. #####
    #####                                                     #####

    aliases = { model_name:make_alias(model_name) for model_name in args.model_list}

    # Retrieve collection names for a certain model
    collection_names_for_model = {model_name:[] for model_name in args.model_list}

    available_collections = weaviate_utils.list_collections()
    print(available_collections)

    for model_name in args.model_list:
        for col_name in available_collections:
            if aliases[model_name] in col_name:
                collection_names_for_model[model_name].append(col_name)

    # print(collection_names_for_model)
    # print("BGE-M3:")
    # print()
    # print(collection_names_for_model["BAAI/bge-m3"])
    # print("Serafim100mir:")
    # print()
    # print(collection_names_for_model["PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir"])
    for model_name in args.model_list:
        q_embeddings = generate_embeddings_w_model(model=model_name, text=q_list, batch_size=args.batch_size)
        with_embeddings = [
            {**question_dict, "vector": q_embeddings[i]}
            for i, question_dict in enumerate(qac_data)
        ]
        with weaviate.connect_to_local() as client:
            for collection in collection_names_for_model[model_name]:
                # There is no need for a deepcopy (think of a better way later)
                with_rel_chunks = copy.deepcopy(with_embeddings)

                for i, entry in tqdm.tqdm(enumerate(with_embeddings), desc=f"Querying for Relevant Chunks in collection={collection}."):
                    q_embedding = entry["vector"]

                    # Vector similarity with pre-computed embeddings
                    relevant_chunks = weaviate_utils.query_with_own_embedding(
                        collection_name = collection,
                        query_embedding = q_embedding,
                        top_k = args.top_k
                    )

                    # NOTE:
                    # relevant_chunks is a list with multiple tuples:
                    # index 0 -> a dict with the properties of the relevant chunk
                    # index 1 -> the cosine similarity distance to the query
                    with_rel_chunks[i]["relevant_chunks"] = relevant_chunks
                save_path = args.results_save_dir + "/" + model_name + "/" + f"top{args.top_k}" + collection + ".jsonl"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "w", encoding="UTF-8") as f:
                    for item in with_rel_chunks:
                        f.write(json.dumps(item) + "\n")
                print(f"Finished saving retrieval results for collection={collection}")
                print(f"Save dir ----> {save_path}")

        



if __name__ == "__main__":
    main()

