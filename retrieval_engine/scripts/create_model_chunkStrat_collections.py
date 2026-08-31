"""
NOTE: THIS SCRIPT CREATES 20 Collections for each model ... (it's a high computational script due to the generation of embeddings for every chunk)

Example of calling the script:
----- (required args) -----
python3 scripts/create_model_chunkStrat_collections.py hnsw "['BAAI/bge-m3','PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir']"

---- (with optional args) ----
python3 scripts/create_model_chunkStrat_collections.py hnsw "['BAAI/bge-m3','PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir']" --token [HF_TOKEN] --batch_size 32
"""
import re
import argparse
import ast  # to safely parse list-like strings
import utils
import torch
from sentence_transformers import SentenceTransformer, util
import chunking
import embeddings
import weaviate_database
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
import time
import json

import weaviate_database as weaviate_utils


# Defining test set parameters for each chunking strategy
aux_fixed_size = [("fixedSize200",200),("fixedSize400",400),("fixedSize600",600)]
aux_fixed_size_overlap = [
    ("overlap200and01",[200,0.1]),("overlap200and02",[200,0.2]),("overlap200and03",[200,0.3]),
    ("overlap400and01",[400,0.1]),("overlap400and02",[400,0.2]),("overlap400and03",[400,0.3]),
    ("overlap600and01",[600,0.1]),("overlap600and02",[600,0.2]),("overlap600and03",[600,0.3])
]
aux_variable_size = [
    ("variableSizeT1", r"\n{1,}"),
    ("variableSizeT2", r"\n{2,}"),
    ("variableSizeT3", r"\n{3,}")
]
aux_mixed_chunking = [
    ("special30",30),("special50",50), ("special70",70), ("special90",90), ("special110",110)
]

params_set_dict = {
    "fixed_size": aux_fixed_size,
    "fixed_size_overlap": aux_fixed_size_overlap,
    "variable_size": aux_variable_size,
    "mixed_chunking": aux_mixed_chunking
}



def main():


    # Clear all available collections before anything
    weaviate_utils.delete_all_collections()


    parser = argparse.ArgumentParser(
        description="Handles cli arguments for a script that can be used to experiment for deciding chunking_size and model."
    )

    # Positional arguments (required)
    parser.add_argument(
        "DB_index",
        choices=["hnsw","flat"],
        help="Type of VectorDB indexation type hnsw/flat"
    )

    parser.add_argument(
        "model_list",
        type=ast.literal_eval, # allows safe parsing of a Python list from a string
        help="List of models for which to create a collection. Example: \"['model1','model2']\""
    )

    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace token for models that need it."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding generation in case memory problem occur"
    )



    args = parser.parse_args()

    # Sanity checks from cli parameters
    print("Arguments received:")
    print(f"DB_index: {args.DB_index}")
    print(f"model_list: {args.model_list}")
    if args.token: print(f"token: SECRET_TOKEN")
    print(f"batch_size: {args.batch_size}")

    aliases = {model_name: utils.make_alias(model_name) for model_name in args.model_list}

    chunks_embeddings = {}
    # 1 - Calculate all chunks with embeddings for different chunking_types
    for model_name in args.model_list:
        # Create all fixed_size collections for this model
        aux_list = [] # -> will recieve a list with [<collection_name>,embeddings]
        for parameters in params_set_dict["fixed_size"]:
            print("parameters:",parameters)
            embedded_chunks = embeddings.generate_PT_wikiRAG_chunks_embeddings(
                model_name=model_name,
                chunking_type="fixed_size",
                chunk_size=parameters[1],
                batch_size=args.batch_size
            )
            collection_name = aliases[model_name] + parameters[0]
            aux_list.append([collection_name, embedded_chunks]) # [<collection_name>,embeddings]

        for parameters in params_set_dict["fixed_size_overlap"]:
            print("parameters:",parameters)
            embedded_chunks = embeddings.generate_PT_wikiRAG_chunks_embeddings(
                model_name=model_name,
                chunking_type="fixed_size_overlap",
                chunk_size=parameters[1][0],
                overlap_factor=parameters[1][1],
                batch_size=args.batch_size
            )
            collection_name = aliases[model_name] + parameters[0]
            aux_list.append([collection_name,embedded_chunks]) # [<collection_name>,embeddings]

        for parameters in params_set_dict["variable_size"]:
            print("parameters:",parameters)
            embedded_chunks = embeddings.generate_PT_wikiRAG_chunks_embeddings(
                model_name=model_name,
                chunking_type="variable_size",
                regex=parameters[1],
                batch_size = args.batch_size,
            )
            collection_name = aliases[model_name] + parameters[0]
            aux_list.append([collection_name,embedded_chunks]) # [<collection_name>,embeddings]

        for parameters in params_set_dict["mixed_chunking"]:
            print("parameters:",parameters)
            embedded_chunks = embeddings.generate_PT_wikiRAG_chunks_embeddings(
                model_name=model_name,
                chunking_type="mixed_chunking",
                min_char_chunk=parameters[1],
                # regex=r"\n{3,}", # the regex is unused when chunking type is mixed_chunking. (it's always regex=r"\n{3,}")
                batch_size = args.batch_size, 
            )
            collection_name = aliases[model_name] + parameters[0]
            aux_list.append([collection_name,embedded_chunks]) # [<collection_name>,embeddings]
        
        for x in aux_list:
            weaviate_database.create_collection(
                collection_name=x[0],
                hnsw = args.DB_index == "hnsw",
                # distance_metric=VectorDistances.COSINE # The default,
                properties = [
                    Property(name="doc_name", data_type=DataType.TEXT),
                    Property(name="text", data_type=DataType.TEXT)
                ]
            )
            nr_chunk_added = weaviate_database.add_own_vectors(
                collection_name=x[0],
                vectors=x[1],
                batch_size=200, # This parameter has to be manipulated on the script (there is no CLI handler for it)
                raise_on_failure=True,
            )
            print(f"Added {nr_chunk_added} vectors to the collection --> {collection_name}")



    list_collections = weaviate_database.list_collections()
    print("List of Collections:")
    print()
    print(list_collections)




if __name__ == "__main__":
    main()