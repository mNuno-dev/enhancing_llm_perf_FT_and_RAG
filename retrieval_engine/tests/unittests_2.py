from weaviate_database import create_doc_collection, add_own_vectors, check_collection, check_local_collections, delete_all_collections, delete_doc_collection, collection_exists, query_simple_vec_similarity, create_doc_collection_flat, check_collection_embedding_size, check_first_3_embeddings, debug_vectors, query_TEI_server_semanic_search, query_STserver_semanic_search
from embeddings import generate_PT_wikiRAG_embeddings, get_local_TEI_embbedding_default_API, get_local_TEI_embbedding_OpenAPI_compatible, generate_TEI_PT_wikiRAG_embeddings, generate_local_STserver_PT_wikiRAG_embeddings
from openai import OpenAI

###
### TEMPORARY VARIABLES
###

"""
Testing developmente of get_local_TEI_embbedding_default_API
"""

TOKEN_HF = os.getenv("HF_TOKEN", "")  # set HF_TOKEN as an environment variable

# host_url = "http://localhost:8089/embed"
# # openAPI_host_url = "http://localhost:8089/v1/embeddings" # OpenAI-compatible endpoint

# embeddings = get_local_TEI_embbedding_default_API(url=host_url, text=["Sismo em Myanmar em 2025\n\nEm 28 de março de 2025 às 12:50:54 MMT (06:20:54 UTC), um sismo de 7,7 na escala de magnitude de momento atingiu a região de Sagaingue, em Myanmar, com epicentro próximo de Mandalai, a segunda maior cidade do país. O terremoto atingiu uma intensidade máxima de IX (violento) na Escala de Mercalli Modificada. Foi o terremoto mais forte que atingiu Myanmar desde 1912 e o segundo mais mortífero da história moderna de Myanmar, ultrapassado apenas pelas estimativas superiores do terremoto de Bago de 1930. O terremoto causou grandes danos em Myanmar e danos significativos na vizinha Tailândia. Centenas de casas foram também danificadas em Iunã, na China.\n\nO terremoto causou mais de 4 390 mortes em Myanmar e 36 na Tailândia.\n\nMais de 6 200 pessoas ficaram feridas. Centenas de outras pessoas foram dadas como desaparecidas, incluindo num local de construção que desabou em Bangkok, que, devido à sua geologia superficial, é mais vulnerável a ondas sísmicas vindas de longe, e à falta de sensibilização, aumentando a suscetibilidade da cidade aos impactos relacionados com terremotos.\n\nEm 28 de março, as autoridades declararam estado de emergência e esperam que o número de mortos aumente.\n\nA atual guerra civil em Myanmar agravou","nos estados de Caime Xã, VII (Muito Forte) na região de Maguai e VI (Forte) nas divisões administrativas de Rangum, Cachim, Mom, Caiá e Irauádi. MMI V (Moderado) foi registrada em Bangkok e Chiang Mai na Tailândia, Dehong Dai e Prefeitura Autônoma de Jingpo na China, e Infal na Índia. No geral, estima-se que a grande maioria da população de Myanmar tenha sido exposta a níveis de tremores de pelo menos MMI V (Moderado) e estima-se que cerca de 22 milhões de pessoas, ou mais de 40% da população do país, tenham sido expostas a níveis de tremores superiores a MMI VI (Forte) em 12 das 15 divisões administrativas de Myanmar. Na Tailândia, o tremor foi sentido em 57 províncias do país. Impacto\n\nMyanmar:\nPelo menos 1 644 pessoas morreram, 130 estão desaparecidas e 3 408 ficaram feridas no país. A emissora disse que as estatísticas de vítimas ainda estavam sendo obtidas.\n\nForam relatadas mortes em Bago, Mandalai, Xã, Nepiedó e Sagaingue.\n\nA infraestrutura de telefonia e internet foi interrompida em todo o país. O terremoto abalou cidades e vilas, incluindo Rangum, Mandalai, Nepiedó, Sagaingue, Aungban, Bago, Kalay, Muse, Kyaukse e Yinmabin. As estimativas de vítimas nas áreas rurais continuaram escassas O Governo de"])

# for x in embeddings:
#     print(x[:3])


"""
    Chunking + Vector generation
    local TEI model
"""
TEIserver_url = "http://localhost:8089/embed" 
"""
-------HEY-------
"""
"""
This is for TEI server instance
"""
"""
-------HEY-------
"""


print("Creating fixed size embeddings")
vectors_fixed_size = generate_TEI_PT_wikiRAG_embeddings(
    inference_url=TEIserver_url,
    dataset_jsonl_subset1_path=None,
    dataset_jsonl_subset2_path=None,
    chunking_type="fixed_size",
    chunk_size=200,
    batch_size=16
)


# print("Creating fixed size overlap embeddings")
# vectors_f_size_overlapp = generate_PT_wikiRAG_embeddings(
#                         model_name="BAAI/bge-m3",
#                         chunking_type="fixed_size_overlap",
#                         hf_token=TOKEN_HF,
#                         chunk_size=200,
#                         batch_size=32
#                     )


# print("Creating variable size embeddings")
# vectors_variable_size = generate_PT_wikiRAG_embeddings(
#                         model_name="BAAI/bge-m3",
#                         chunking_type="variable_size",
#                         hf_token=TOKEN_HF,
#                         batch_size=4
#                     )

# print("Creating special mixed chunking embeddings")
# vectors_special_mixed_chunking = generate_PT_wikiRAG_embeddings(
#                         model_name="BAAI/bge-m3",
#                         chunking_type="mixed_chunking",
#                         hf_token=TOKEN_HF,
#                         min_char_chunk=80,
#                         batch_size=4,
#                     )


"""
Creating flat collection (BGE/bge-m3) on local TEI server
"""
# NOTE: WITH -> model_name="BGE/bge-m3" on a local TEI server

# check_local_collections()
# delete_all_collections()
# check_local_collections()


# # Creating Fixed-Size chunks Collection with PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir
# create_doc_collection_flat("FixedSizeLOCALbgem3")
# check_collection("FixedSizeLOCALbgem3")
# add_own_vectors(vectors=vectors_fixed_size,name="FixedSizeLOCALbgem3")
# check_collection("FixedSizeLOCALbgem3")

"""
-------HEY-------
"""
"""
This is for TEI server instance
"""
"""
-------HEY-------
"""



"""
-------HEY-------
"""
"""
This is for STserver instance
"""
"""
-------HEY-------
"""
STserver_url = "http://localhost:8000/embed" 

# print("Creating fixed size embeddings STserver")
# vectors_fixed_size = generate_local_STserver_PT_wikiRAG_embeddings(
#     inference_url=STserver_url,
#     dataset_jsonl_subset1_path=None,
#     dataset_jsonl_subset2_path=None,
#     chunking_type="fixed_size",
#     chunk_size=200,
#     batch_size=16
# )


# print("Creating fixed size overlap embeddings")
# vectors_f_size_overlapp = generate_PT_wikiRAG_embeddings(
#                         model_name="BAAI/bge-m3",
#                         chunking_type="fixed_size_overlap",
#                         hf_token=TOKEN_HF,
#                         chunk_size=200,
#                         batch_size=32
#                     )


# print("Creating variable size embeddings")
# vectors_variable_size = generate_PT_wikiRAG_embeddings(
#                         model_name="BAAI/bge-m3",
#                         chunking_type="variable_size",
#                         hf_token=TOKEN_HF,
#                         batch_size=4
#                     )

# print("Creating special mixed chunking embeddings")
# vectors_special_mixed_chunking = generate_PT_wikiRAG_embeddings(
#                         model_name="BAAI/bge-m3",
#                         chunking_type="mixed_chunking",
#                         hf_token=TOKEN_HF,
#                         min_char_chunk=80,
#                         batch_size=4,
#                     )


"""
Creating flat collection (BAAI/bge-m3) for local embeddings generated by local STserver
"""
# NOTE: WITH -> model_name="BAAI/bge-m3" on a local TEI server

# check_local_collections()
# delete_all_collections()
# check_local_collections()


# # Creating Fixed-Size chunks Collection with BAAI/bge-m3
# create_doc_collection_flat("FixedSizeSTserverBGEm3")
# check_collection("FixedSizeSTserverBGEm3")
# add_own_vectors(vectors=vectors_fixed_size,name="FixedSizeSTserverBGEm3")
# check_collection("FixedSizeSTserverBGEm3")

"""
-------HEY-------
"""
"""
This is for STserver instance
"""
"""
-------HEY-------
"""





"""
Testing Fixed Size Retrieval Results on a small set of questions from PT_wikiRAG
    -  Only with query_simple_vec_similarity
"""

q_1 = "Quantos edifícios religiosos foram danificados ou destruídos em Myanmar devido ao sismo?"

q_2 = "Qual foi a intensidade máxima do sismo de Myanmar em 2024, segundo a Escala de Mercalli Modificada?"

q_3 = "Myanmar"

q_4 = "Qual era o nome do Ministro do Interior da Turquia em março de 2025?"

print("Checking Collections:")
check_local_collections()

"""
"""
###
###### -> Test for query_TEI_server_semanic_search(
###


# relevant_chunks = query_TEI_server_semanic_search(
#     inference_url=TEIserver_url,
#     collection_name="FixedSizeLOCALbgem3",
#     query=q_4,
#     top_k=3
# )
# # x = query_simple_vec_similarity(name="FixedSizebgem3", query=q_2, top_k=5, model_name="BAAI/bge-m3")

# print("Query:",q_4)
# print("Results for Fixed Size Chunking -> embedding_model=bge-m3")
# for i in range(len(relevant_chunks)):
#     print("------ /// ------")
#     print(f"Rel Chunk {i+1}:\n")
#     print(relevant_chunks[i])
#     print("------ /// ------")

###
###### -> Test for query_TEI_server_semanic_search(
###


# relevant_chunks = query_TEI_server_semanic_search(
#     inference_url=TEIserver_url,
#     collection_name="FixedSizeLOCALbgem3",
#     query=q_3,
#     top_k=3
# )
# # x = query_simple_vec_similarity(name="FixedSizebgem3", query=q_2, top_k=5, model_name="BAAI/bge-m3")

# print("Query:",q_2)
# print("Results for Fixed Size Chunking with overlap -> embedding_model=bge-m3")
# for i in range(len(relevant_chunks)):
#     print("------ /// ------")
#     print(f"Rel Chunk {i+1}:\n")
#     print(relevant_chunks[i])
#     print("------ /// ------")


###
###### -> Test for query_STserver_semanic_search(
###


# relevant_chunks = query_STserver_semanic_search(
#     inference_url=STserver_url,
#     collection_name="FixedSizeSTserverBGEm3",
#     query=q_4,
#     top_k=3
# )
# # x = query_simple_vec_similarity(name="FixedSizebgem3", query=q_2, top_k=5, model_name="BAAI/bge-m3")

# print("Query:",q_4)
# print("Results for Fixed Size Chunking with overlap -> embedding_model=bge-m3")
# for i in range(len(relevant_chunks)):
#     print("------ /// ------")
#     print(f"Rel Chunk {i+1}:\n")
#     print(relevant_chunks[i])
#     print("------ /// ------")
