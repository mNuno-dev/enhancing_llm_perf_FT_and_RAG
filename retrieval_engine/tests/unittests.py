from embeddings.embeddings import generate_PT_wikiRAG_embeddings
from weaviate_database.weaviate_database import create_doc_collection, add_own_vectors, check_collection, check_local_collections, delete_all_collections, delete_doc_collection, collection_exists, query_simple_vec_similarity, create_doc_collection_flat, check_collection_embedding_size, check_first_3_embeddings, debug_vectors
###
### TEMPORARY VARIABLES
###

TOKEN_HF = os.getenv("HF_TOKEN", "")  # set HF_TOKEN as an environment variable

# Dataset location
subsplit1_path = None  # Path to subsplit1 docs JSONL file
subsplit2_path = None  # Path to subsplit2 QAC JSONL file

###
### TEMPORARY VARIABLES
###

"""
----------------------------//----------------------------
"""

# Testing local loading
# generate_PT_wikiRAG_embeddings(chunking_type="fixed_size_overlap",dataset_jsonl_subset1_path=subsplit1_path, dataset_jsonl_subset2_path=subsplit2_path)

# Testing HF loading
# generate_PT_wikiRAG_embeddings(chunking_type="fixed_size_overlap", hf_token=TOKEN_HF)
# generate_PT_wikiRAG_embeddings(chunking_type="fixed_size", hf_token=TOKEN_HF)
# generate_PT_wikiRAG_embeddings(chunking_type="variable_size", hf_token=TOKEN_HF)
# generate_PT_wikiRAG_embeddings(chunking_type="mixed_chunking", hf_token=TOKEN_HF)

# generate_PT_wikiRAG_embeddings(model_name="PORTULAN/serafim-100m-portuguese-pt-sentence-encoder", chunking_type="fixed_size_overlap", hf_token=TOKEN_HF)
# generate_PT_wikiRAG_embeddings(model_name="PORTULAN/serafim-100m-portuguese-pt-sentence-encoder",chunking_type="fixed_size", hf_token=TOKEN_HF)
# generate_PT_wikiRAG_embeddings(model_name="PORTULAN/serafim-100m-portuguese-pt-sentence-encoder",chunking_type="variable_size", hf_token=TOKEN_HF)
# generate_PT_wikiRAG_embeddings(model_name="PORTULAN/serafim-100m-portuguese-pt-sentence-encoder", chunking_type="mixed_chunking", hf_token=TOKEN_HF)




# vectors = generate_PT_wikiRAG_embeddings(model_name="PORTULAN/serafim-100m-portuguese-pt-sentence-encoder",chunking_type="fixed_size", hf_token=TOKEN_HF)

# create_doc_collection(name="wikiTest3")

# add_own_vectors(vectors=vectors, db_name="wikiTestMariliazita")

# check_collection("wikiTestTESTMAD")

# print(len(vectors))


# Checking local collections
# check_local_collections()
# check_collection("WikiTestMarilia")


# Testing if trying to get a collection that doesn't exist works
# collection = get_database_collection("askdhakldhas")


# Checking available collections
#check_local_collections()

# Testing getting a collection object by it's name
#collection = get_database_collection("WikiTestMariliazita")

# vectors = generate_PT_wikiRAG_embeddings(model_name="PORTULAN/serafim-100m-portuguese-pt-sentence-encoder",chunking_type="fixed_size", hf_token=TOKEN_HF)


# Testing create_doc_collection()
# delete_all_collections()
# create_doc_collection("col1")
# check_collection("col1")


# Testing edgecases - add_own_vectors()

# Case 1 -> The Collection Name is Invalid and doesn't exist
# add_own_vectors(vectors=vectors, db_name="mriliaJoaquina")

# Case 2 -> Everything is OK (with an existing collection)
# Check PT_wikiRAG state
# check_collection("PT_wikiRAG")
# add_own_vectors(vectors=vectors, db_name="PT_wikiRAG")
# check_collection("PT_wikiRAG")


# Case 3 -> Create a database first, then add own vectors

# Create doc collection vec_db
# create_doc_collection("PT_wikiRAG_323")
# print("First check:")
# check_collection("PT_wikiRAG_323")

# Add own vectors to doc collection
# add_own_vectors(vectors=vectors, db_name="PT_wikiRAG_323")
# print("Second check:")
# check_collection("PT_wikiRAG_323")

# Add own vectors (again) to doc collection (THIS IS NOT SUPPOSED TO BE DONE)
# but it's just a check
# add_own_vectors(vectors=vectors, db_name="PT_wikiRAG_323")
# print("Third check:")
# check_collection("PT_wikiRAG_323")



"""
Testing consecutive collection operations:
1 - Create Doc Collection
2 - Create own vectors
2 - Add own vectors to collection
"""

# NOTE: WITH -> model_name="nomic-ai/nomic-embed-text-v1.5"

# delete_all_collections()
# create_doc_collection("TiaMariaz")
# check_collection("TiaMariaz")

# vectors = generate_PT_wikiRAG_embeddings(model_name="nomic-ai/nomic-embed-text-v1.5",chunking_type="fixed_size", hf_token=TOKEN_HF)

# add_own_vectors(vectors=vectors, name="TiaMariaz")
# check_collection("TiaMariaz")

# NOTE: WITH -> model_name="PORTULAN/serafim-900m-portuguese-pt-sentence-encoder"
# (NOT IR FOCUSED)

# delete_all_collections()
# create_doc_collection("TiaMariaz")
# check_collection("TiaMariaz")

# vectors = generate_PT_wikiRAG_embeddings(model_name="PORTULAN/serafim-900m-portuguese-pt-sentence-encoder",chunking_type="fixed_size", hf_token=TOKEN_HF)

# add_own_vectors(vectors=vectors, name="TiaMariaz")
# check_collection("TiaMariaz")

# NOTE: WITH -> model_name="PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir"
# (IR FOCUSED)

# delete_all_collections()
# create_doc_collection("TiaMariaz")
# check_collection("TiaMariaz")

# vectors = generate_PT_wikiRAG_embeddings(model_name="PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir",chunking_type="fixed_size", hf_token=TOKEN_HF)

# add_own_vectors(vectors=vectors, name="TiaMariaz")
# check_collection("TiaMariaz")


#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#                           Flat Collection (not default HNSW)
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------


"""
Creating flat collection (PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir)
"""
# NOTE: WITH -> model_name="PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir"
# (IR FOCUSED)

# check_local_collections()
# delete_all_collections()
# create_doc_collection_flat("TiaMariazFLAT")
# check_collection("TiaMariazFLAT")

# vectors = generate_PT_wikiRAG_embeddings(model_name="PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir",chunking_type="fixed_size", hf_token=TOKEN_HF)

# add_own_vectors(vectors=vectors, name="TiaMariazFLAT")
# check_collection("TiaMariazFLAT")

"""
Creating flat collection (nomic-ai/nomic-embed-text-v1.5)
"""
# NOTE: WITH -> model_name="nomic-ai/nomic-embed-text-v1.5"
# (sentence-embedder)

# check_local_collections()
# delete_all_collections()
# create_doc_collection_flat("TiaMariazFLAT")
# check_collection("TiaMariazFLAT")

# vectors = generate_PT_wikiRAG_embeddings(model_name="nomic-ai/nomic-embed-text-v1.5",chunking_type="fixed_size", hf_token=TOKEN_HF)

# add_own_vectors(vectors=vectors, name="TiaMariazFLAT")
# check_collection("TiaMariazFLAT")






"""
Testing collection exists
"""
# check_local_collections()
# print(collection_exists("TiaMariaz"))
# print(collection_exists("TiaJoana"))


# """
# Testing query_simple_vec_similarity
# """
# q_1 = "Quantos edifícios religiosos foram danificados ou destruídos em Myanmar devido ao sismo?"

# q_2 = "Qual foi a intensidade máxima do sismo de Myanmar em 2024, segundo a Escala de Mercalli Modificada?"

# q_3 = "Myanmar"

# q_4 = "Qual era o nome do Ministro do Interior da Turquia em março de 2025?"

# print("Checking Collections:")
# check_local_collections()

# x = query_simple_vec_similarity(name="TiaMariazFLAT", query=q_2, top_k=3, model_name="nomic-ai/nomic-embed-text-v1.5")

# for i in range(len(x)):
#     print("------ /// ------")
#     print(f"Rel Chunk {i+1}:\n")
#     print(x[i])
#     print("------ /// ------")


# for i in range(len(x[1])):
#     print("------ /// ------")
#     print(f"Rel Chunk {i+1}:\n")
#     print(x[1][i])
#     print("------ /// ------")


# print(x[1])

# x="""sustentando que o judiciário operava independentemente de influência política. Quando questionados especificamente sobre as alegações de que a detenção foi motivada politicamente, os representantes do gabinete do presidente Erdogan não forneceram comentários imediatos. Dias depois, Erdogan falou a favor da independência judicial e disse que os protestos eram uma "perturbação da ordem pública".\n\nAs linhas de metro e o transporte de autocarro foram encerrados na Estação da Universidade Técnica do Oriente Médio, em Ancara, durante os protestos estudantis. Repressão nas redes sociais\n\nO Ministro do Interior da Turquia, Ali Yerlikaya, anunciou na manhã de 20 de março que as autoridades policiais identificaram 261 “suspeitos gestores de contas” que alegadamente partilharam conteúdos “incitando o público ao ódio e à hostilidade” e “ incitando à prática de um crime”. As autoridades detiveram 37 indivíduos em conexão com estas alegações, continuando os esforços para deter suspeitos adicionais.\n\nAs autoridades turcas solicitaram o bloqueio de 700 contas na rede social X (antigamente Twitter), mas a rede recusou os pedidos. Internacional\n\nA detenção de İmamoğlu e os protestos que se seguiram atraíram a atenção internacional, com vários governos e organizações de direitos humanos a expressarem preocupação com o retrocesso democrático na Turquia. A Diretora Regional Adjunta da Anistia Internacional'}, {'doc_name': 'Artigo - Protestos na Turquia 2025.txt', 'text': 'Protestos na Turquia 2025\n\nUma série de grandes protestos estão ocorrendo na Turquia, começando em 19 de março de 2025, após a detenção e prisão do prefeito de Istambul, Ekrem İmamoğlu, centenas de outros membros da oposição e manifestantes pelas autoridades turcas. As manifestações representaram uma oposição pública significativa ao que os seus participantes caracterizaram como ações legais politicamente motivadas contra İmamoğlu, que foi o principal candidato da oposição para as eleições presidenciais turcas de 2028 e o principal rival político do presidente turco Recep Tayyip Erdoğan.\n\nAs manifestações foram apoiadas pelo CHP e muitos outros partidos políticos, organizações e associações. Centenas de milhares de pessoas estão protestando em quase todas as cidades da Turquia (especialmente em Istambul, Ancara e Esmirna ), com a maior multidão se concentrando em frente à sede da Prefeitura Metropolitana de Istambul. Os estudantes universitários estão a desempenhar um papel importante nestes protestos.\n\nOs manifestantes representam um amplo espectro ideológico, incluindo indivíduos de direita e de esquerda. Nesse contexto, símbolos da República Turca — particularmente Atatürk — são frequentemente usados como uma estrutura unificadora e ponto de referência simbólico ao longo das manifestações. Os protestos estão ocorrendo no contexto de uma crise econômica. O valor da lira turca"""

# print("Ali Yerlikaya" in x)


#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#                        Testing with different chunking techniques
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------

"""
    Chunking + Vector generation
    model -> PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir
"""
# vectors_fixed_size = generate_PT_wikiRAG_embeddings(
#                         model_name="PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir",
#                         chunking_type="fixed_size",
#                         hf_token=TOKEN_HF
#                     )

# vectors_f_size_overlapp = generate_PT_wikiRAG_embeddings(
#                         model_name="PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir",
#                         chunking_type="fixed_size_overlap",
#                         hf_token=TOKEN_HF
#                     )

# vectors_variable_size = generate_PT_wikiRAG_embeddings(
#                         model_name="PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir",
#                         chunking_type="variable_size",
#                         hf_token=TOKEN_HF
#                     )

# vectors_special_mixed_chunking = generate_PT_wikiRAG_embeddings(
#                         model_name="PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir",
#                         chunking_type="mixed_chunking",
#                         hf_token=TOKEN_HF
#                     )


"""
    Chunking + Vector generation
    model -> BAAI/bge-m3
"""
# print("Creating fixed size embeddings")
# vectors_fixed_size = generate_PT_wikiRAG_embeddings(
#                         model_name="BAAI/bge-m3",
#                         chunking_type="fixed_size",
#                         hf_token=TOKEN_HF,
#                         chunk_size=200,
#                         batch_size=32
#                     )


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
Creating flat collection (PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir)
"""
# NOTE: WITH -> model_name="PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir"
# (IR FOCUSED)

# check_local_collections()
# delete_all_collections()
# check_local_collections()


# # Creating Fixed-Size chunks Collection with PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir
# create_doc_collection_flat("FixedSize900Serafim")
# check_collection("FixedSize900Serafim")
# add_own_vectors(vectors=vectors_fixed_size,name="FixedSize900Serafim")
# check_collection("FixedSize900Serafim")


# # Creating Fixed-Size-with-Overlap chunks Collection with PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir
# create_doc_collection_flat("FixedSizeOverlap900Serafim")
# check_collection("FixedSizeOverlap900Serafim")
# add_own_vectors(vectors=vectors_f_size_overlapp,name="FixedSizeOverlap900Serafim")
# check_collection("FixedSizeOverlap900Serafim")


# # Creating Variable-Size chunks Collection with PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir
# create_doc_collection_flat("VariableSize900Serafim")
# check_collection("VariableSize900Serafim")
# add_own_vectors(vectors=vectors_variable_size,name="VariableSize900Serafim")
# check_collection("VariableSize900Serafim")


# # Creating Mixed-Chunking (techniques) Collection with PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir
# create_doc_collection_flat("MixedChunking900Serafim")
# check_collection("MixedChunking900Serafim")
# add_own_vectors(vectors=vectors_special_mixed_chunking,name="MixedChunking900Serafim")
# check_collection("MixedChunking900Serafim")



"""
Creating flat collection (BAAI/bge-m3)
"""
# NOTE: WITH -> model_name="BAAI/bge-m3"

# check_local_collections()
# delete_all_collections()
# check_local_collections()


# # # Creating Fixed-Size chunks Collection with BAAI/bge-m3
# print("Creating fixed size embeddings - COLLECTION")
# create_doc_collection_flat("FixedSizebgem3")
# check_collection("FixedSizebgem3")
# add_own_vectors(vectors=vectors_fixed_size,name="FixedSizebgem3")
# check_collection("FixedSizebgem3")


# # # Creating Fixed-Size-with-Overlap chunks Collection with BAAI/bge-m3
# create_doc_collection_flat("FixedSizeOverlapbgem3")
# check_collection("FixedSizeOverlapbgem3")
# add_own_vectors(vectors=vectors_f_size_overlapp,name="FixedSizeOverlapbgem3")
# check_collection("FixedSizeOverlapbgem3")


# # # Creating Variable-Size chunks Collection with BAAI/bge-m3
# create_doc_collection_flat("VariableSizeOverlapbgem3")
# check_collection("VariableSizeOverlapbgem3")
# add_own_vectors(vectors=vectors_variable_size,name="VariableSizeOverlapbgem3")
# check_collection("VariableSizeOverlapbgem3")


# # # Creating Mixed-Chunking (techniques) Collection with BAAI/bge-m3
# create_doc_collection_flat("MixedChunkingbgem3")
# check_collection("MixedChunkingbgem3")
# add_own_vectors(vectors=vectors_special_mixed_chunking,name="MixedChunkingbgem3")
# check_collection("MixedChunkingbgem3")




# # Checking the state of the database
# print("How does the database look:\n")
# print()
# check_local_collections()
# print()


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

# """
# dict_keys(['FixedSize900Serafim', 'FixedSizeOverlap900Serafim', 'MixedChunking900Serafim', 'VariableSize900Serafim'])
# """
"""
dict_keys(['FixedSizeOverlapbgem3', 'FixedSizebgem3', 'MixedChunkingbgem3', 'VariableSizeOverlapbgem3'])
# """
x = query_simple_vec_similarity(name="FixedSizebgem3", query=q_2, top_k=5, model_name="BAAI/bge-m3")

print("Query:",q_2)
print("Results for Fixed Size Chunking with overlap -> embedding_model=bge-m3")
for i in range(len(x)):
    print("------ /// ------")
    print(f"Rel Chunk {i+1}:\n")
    print(x[i])
    print("------ /// ------")



# check_collection("FixedSizebgem3")
# check_collection_embedding_size("FixedSizebgem3")
# check_first_3_embeddings("FixedSizebgem3")




# how_many_duplicate_vectors = debug_vectors("FixedSizebgem3")

# print(how_many_duplicate_vectors)

# check_collection("FixedSizebgem3")