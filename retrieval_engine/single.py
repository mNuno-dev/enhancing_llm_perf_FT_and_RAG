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

def main():
    parser = argparse.ArgumentParser(
        description="script description..."
    )

    # Positional arguments (required)
    parser.add_argument(
        "DB_index",
        choices=["hnsw","flat"],
        help="Type of VectorDB indexation type hnsw/flat"
    )

    parser.add_argument(
        "chunk_size",
        type=int,
        help="Chunk size (by characters) used in both Fixed-Size-Chunking and Fixed-Size-Chunking-With-Overlap"
    )

    parser.add_argument(
        "overlap_factor",
        type=float,
        help="Overlap factor (by characters) used in Fixed-Size-Chunking-With-Overlap [0,1.0], ex: overlap_factor = 0.2 and chunk_size = 200 // overlap = 0.2*200 = 40"
    )

    parser.add_argument(
        "min_char_chunk",
        type=int,
        help="minimum number of characters that a chunk should have to be considered as a chunk on the Special-Mixed chunking strategy"
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
    # Example usage
    print("Arguments received:")
    print(f"DB_index: {args.DB_index}")
    print(f"chunk_size: {args.chunk_size}")
    print(f"overlap_factor: {args.overlap_factor}")
    print(f"min_char_chunk: {args.min_char_chunk}")
    print(f"model_list: {args.model_list}")
    if args.token: print(f"token: SECRET_TOKEN")
    print(f"batch_size: {args.batch_size}")

    # need a function here that takes all the arguments and creates all chunks ...
    # return chunks
    # args 
   
    print(f"DB_index: {args.DB_index}")
    print(f"chunk_size: {args.chunk_size}")
    print(f"overlap_factor: {args.overlap_factor}")
    print(f"min_char_chunk: {args.min_char_chunk}")
    print(f"model_list: {args.model_list}")


    chunks_embeddings = {}

    # 1 - Calculate all chunks with embeddings for different chunking_types
    for model_name in args.model_list:
        fixed_size = embeddings.generate_PT_wikiRAG_chunks_embeddings(
            model_name=model_name,
            chunking_type="fixed_size",
            chunk_size=args.chunk_size,
            batch_size=args.batch_size
        )

        # print()
        # print()
        # print(fixed_size[0])
        # print(fixed_size[0]["vector"])
        # print()
        # print()

        # fixed_size_overlap = embeddings.generate_PT_wikiRAG_chunks_embeddings(
        #     model_name=model_name,
        #     chunking_type="fixed_size_overlap",
        #     chunk_size=args.chunk_size,
        #     overlap_factor=args.overlap_factor,
        #     batch_size=args.batch_size,
        # )

        # variable_size = embeddings.generate_PT_wikiRAG_chunks_embeddings(
        #     model_name=model_name,
        #     chunking_type="variable_size",
        #     regex=r"\n{3,}",
        #     batch_size=4,
        # )

        # special = embeddings.generate_PT_wikiRAG_chunks_embeddings(
        #     model_name=model_name,
        #     chunking_type="mixed_chunking",
        #     min_char_chunk=60,
        #     regex=r"\n{3,}",
        #     batch_size=4,            
        # )

        # chunks_embeddings[model_name] = (
        #     ["fixed_size", fixed_size],
        #     # "fixed_size_overlap": fixed_size_overlap,
        #     # "variable_size": variable_size,
        #     # "mixed_chunking": special
        # )


    #with open("out_debug.txt", "w", encoding="UTF-8") as f:
    #    f.write(json.dumps(chunks_embeddings))
    # print(chunks_embeddings["BAAI/bge-m3"].keys())
    # print(chunks_embeddings["BAAI/bge-m3"]["variable_size"])
    
    # print(type(chunks_embeddings["BAAI/bge-m3"]["fixed_size"]))
    # time.sleep(5)
    # print(type(chunks_embeddings["BAAI/bge-m3"]["fixed_size"][0]))
    # time.sleep(5)
    # print(chunks_embeddings["BAAI/bge-m3"]["fixed_size"])
    # time.sleep(20)

    # 2 - Create all weaviate.Collections for each model and chunking strategy

    aliases = {model_name: utils.make_alias(model_name) for model_name in args.model_list}
    chunking_types = ['fixed_size']
    chunking_types_aliases = ["fixedSize"]
    for model_name in args.model_list:
        # for index, chunking_type in enumerate(chunking_types):
        collection_name = "fdscrlh"
        weaviate_database.create_collection(
            collection_name=collection_name,
            hnsw = True,
            # distance_metric=VectorDistances.COSINE # The default,
            # properties=[Property(name="doc_name", data_type=DataType.TEXT), Property(name="text", data_type=DataType.TEXT)]
        )
        weaviate_database.add_own_vectors(
            collection_name=collection_name,
            vectors=fixed_size,
            batch_size=200, # This parameter has to be manipulated on the script (there is no CLI handler for it)
            raise_on_failure=True,
        )
        # print(f"Added {n_vec_added} vectors to the collection --> {collection_name}")


list_collections = weaviate_database.list_collections()

print("List of Collections:")
print()
print(list_collections)




if __name__ == "__main__":
    main()