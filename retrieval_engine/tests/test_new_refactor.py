import requests


from embeddings import (
    request_local_STserver_embeddings,
    request_local_TEI_embeddings,
    generate_local_STserver_PT_wikiRAG_embeddings,
    generate_TEI_PT_wikiRAG_embeddings,
    local_STserver_info,
    local_TEI_server_info

)
import weaviate_database as weaviate_utils


###
### TEMPORARY VARIABLES
###

"""
Testing developmente of get_local_TEI_embbedding_default_API
"""

TOKEN_HF = os.getenv("HF_TOKEN", "")  # set HF_TOKEN as an environment variable
TEIserver_url = "http://localhost:8089/embed"
STserver_url = "http://localhost:8000/embed" 



###
### Small test setup
###

"""
Testing Fixed Size Retrieval Results on a small set of questions from PT_wikiRAG
    -  Only with query_simple_vec_similarity
"""

q_1 = "Quantos edifícios religiosos foram danificados ou destruídos em Myanmar devido ao sismo?"

q_2 = "Qual foi a intensidade máxima do sismo de Myanmar em 2024, segundo a Escala de Mercalli Modificada?"

q_3 = "Myanmar"

q_4 = "Qual era o nome do Ministro do Interior da Turquia em março de 2025?"


print("Checking Collections:")
collections = weaviate_utils.list_collections()
print(collections)

print("Deleting all Collections")
print(weaviate_utils.delete_all_collections())

print("Verifying empty dict after all collections deletion")
collections = weaviate_utils.list_collections()
print(collections)

###
### Small test setup
###


###
### Getting local STserver info
###

# info = local_STserver_info(server_url="http://localhost:8000")
# print(info)
# model_name = info["model_name"]
# device = info["device"]
# embedding_dimension = info["embedding_dim"]




"""
1 - Create own vectors
2 - Create collection
3 - Add vectors to collection
4 - Query test on the collection
"""

# Generating the embeddings for all the chunks of Fixed Size
# print("Creating fixed size embeddings")
# vectors_fixed_size = generate_local_STserver_PT_wikiRAG_embeddings(
#     inference_url=STserver_url,
#     dataset_jsonl_subset1_path=None,
#     dataset_jsonl_subset2_path=None,
#     chunking_type="fixed_size",
#     chunk_size=200,
#     batch_size=16
# )

# collection_name = "localSTbgem3FIXED-Size"

# # Creating the collection
# weaviate_utils.create_collection(
#     collection_name="localSTbgem3FIXED-Size"
# )

# # Adding own vectors to the collection
# weaviate_utils.add_own_vectors(
#     collection_name=collection_name,
#     vectors=vectors_fixed_size,
# )

# # Vector similarity with server embedding model
# relevant_chunks = weaviate_utils.query_with_external_embedding_call(
#     request_local_STserver_embeddings,
#     inference_url=STserver_url,
#     collection_name=collection_name,
#     query_text=q_2,
# )


# print("Query:",q_2)
# print("Results for Fixed Size Chunking with overlap -> embedding_model=bge-m3")
# for i in range(len(relevant_chunks)):
#     print("------ /// ------")
#     print(f"Rel Chunk {i+1}:\n")
#     print(relevant_chunks[i])
#     print("------ /// ------")


