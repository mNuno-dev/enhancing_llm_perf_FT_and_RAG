####
# LEGACY CODE
####
# def create_doc_collection(name: str):   
#     # NOTE: By default it creates a HNSW - Hierarchical Navigatable Small World
#     with weaviate.connect_to_local() as client:
#         if client.collections.exists(name=name):
#             client.close() # Closing the client
#             raise RuntimeError(f"There is already a collection named '{name}'. Please choose another name, or delete the current collection before creating a new one with the same name.")
        

#         client.collections.create(
#             name=name,
#             properties=[
#                 Property(name="doc_name",data_type=DataType.TEXT),
#                 Property(name="text", data_type=DataType.TEXT)
#             ],
#             vector_config=wvc.config.Configure.Vectors.self_provided()
#             # Additional parameters ..
#         )


# def create_doc_collection_flat(name:str):
#     # NOTE: By deafault it creates a HNSW - Hierarchical Navigatable Small World
#     #           - To counter this because my chunk collection is smaller than 10K 
#     #             using a flat index might be the best option.
#     with weaviate.connect_to_local() as client:
#         if client.collections.exists(name=name):
#             raise RuntimeError(
#                 f"There is already a collection named '{name}'. "
#                 "Please choose another name, or delete the current collection."
#             )

#         # Creating a flat_index collection in the database
#         client.collections.create(
#             name=name,
#             vector_config=Configure.Vectors.self_provided( # no model; you supply vectors
#                 vector_index_config=Configure.VectorIndex.flat( # other options # or hnsw, dynamic
#                     distance_metric=VectorDistances.COSINE,
#                     # Optional: enable BQ to speed up flat search
#                     # quantizer=Configure.VectorIndex.Quantizer.bq(cache=True),
#                     # vector_cache_max_objects=1_000_000,                
#                 ) # other options # or hnsw, dynamic
#             ),
#             properties=[
#                 Property(name="doc_name",data_type=DataType.TEXT),
#                 Property(name="text", data_type=DataType.TEXT)
#             ],

#         )
    

# def delete_doc_collection(name: str):
#     with weaviate.connect_to_local() as client:
#         if not client.collections.exists(name=name):
#             raise RuntimeError(f"There no collection named '{name}'.")
#         client.collections.delete(name)


# def delete_all_collections():
#     with weaviate.connect_to_local() as client:
#         collections_list = client.collections.list_all().keys()
#         client.collections.delete_all()
#         print("All weaviate collections deleted!")
#         print(f"Deleted:\n {collections_list}")


# # def get_database_collection(db_name: str):
# #     with weaviate.connect_to_local() as client:
# #         if db_name not in client.collections.list_all().keys():
# #             print(f"\n\n(Debugging)\n\nError! --> There is no database with called '{db_name}'.\n\n(Debugging)\n\n")
# #             return None
        
# #         return client.collections.get(db_name)


# def add_own_vectors(vectors, name: str):
#     with weaviate.connect_to_local() as client:

#         # Does the collection exist? If not, exit the operation
#         if not client.collections.exists(name=name):
#             raise RuntimeError(f"The collection name '{name}' doesn't exist.\n\n Available Collections: {client.collections.list_all().keys()}")
#             # return
        


#         # # NOTE: Justification for (second condition fo the if statement)
#         # # Need to use the .lower() approach because Weaviate doesn't let collections with
#         # # the same name (lowercase) be created (it's not case sensitive).
#         # # It's something to do with GraphQL
#         # #
#         # # Create collection if it doesn't exist, configured for self-provided vectors
#         # if exists == False and db_name.lower() not in [key.lower() for key in client.collections.list_all().keys()]:
#         #     client.collections.create(
#         #         name=db_name,
#         #         vector_config=wvc.config.Configure.Vectors.self_provided(),
#         #     ) # Collection accepts user-provided vectors (vectorizer=none)
#         #     # (Ref: “Bring your own vectors – collection definition”) 
#         #     # (https://docs.weaviate.io/weaviate/starter-guides/custom-vectors#collection-definition)
#         #     exists = True
    

#         # Get the collection object
#         collection = client.collections.get(name)
        
#         #OLD
#         """
#         # for vec in vectors:
#         #     collection.data.insert(
#         #         properties={
#         #             "doc_name": vec["doc_name"],
#         #             "text": vec["text"],
#         #         },
#         #         vector=vec["vector"]
#         #     )
#         """

#         # Batch insert with explicit vectors
#         with collection.batch.fixed_size(batch_size=200) as batch:
#             for vec in vectors:
#                 batch.add_object(
#                     properties={
#                         "doc_name": vec["doc_name"],
#                         "text": vec["text"],
#                     },
#                     vector=vec["vector"]
#                 )
#                 # (Ref: “Batch import – specify a vector”) 
#                 # (https://docs.weaviate.io/weaviate/manage-objects/import#specify-a-vector)

#         # Optional: check for failed objects
#         failed = collection.batch.failed_objects
#         if failed:
#             raise RuntimeError(f"Failed to import {len(failed)} objects; first: {failed[0]}")
            

        
# def check_collection(name):
#     client = weaviate.connect_to_local()
#     count = 0
#     try:
#         collection = client.collections.get(name)
#         print("Collection Name:", collection.name)
#         print("Size: ",len(collection))
#         print()
#         # for item in collection.iterator(include_vector=True):
#         #     count += 1
#         #     # print("uuid: ", item.uuid)
#         #     # print("properties: ", item.properties)
#         #     # print("vector: ", item.vector)
#         #     # print()
#         # print("total_count: ", count)    
#     finally:
#         client.close()

# def check_collection_embedding_size(name):
#     client = weaviate.connect_to_local()
#     count = 0
#     try:
#         collection = client.colections.get(name)
#         print("Collection Name:", collection.name)
#         iterator = collection.iterator(include_vector=True)
#         first = next(iterator)
#         # print("First vector:",first)
#         print("Embedding Size: ",len(first.vector["default"]))
#         print()
#         # for item in collection.iterator(include_vector=True):
#         #     count += 1
#         #     # print("uuid: ", item.uuid)
#         #     # print("properties: ", item.properties)
#         #     # print("vector: ", item.vector)
#         #     # print()
#         # print("total_count: ", count)    
#     finally:
#         client.close()

# def check_first_3_embeddings(name):
#     with weaviate.connect_to_local() as client:
#         collection = client.collections.get(name)
#         iterator = collection.iterator(include_vector=True)
#         first = next(iterator)
#         second = next(iterator)
#         third = next(iterator)
#         print(first.vector["default"][:10])
#         print(second.vector["default"][:10])
#         print(third.vector["default"][:10])

# def debug_vectors(name):
#     client = weaviate.connect_to_local()
#     count = 0
#     try:
#         collection = client.collections.get(name)
#         print("Collection Name:", collection.name)

#         seen = set()   # will hold tuples of vectors
#         for item in collection.iterator(include_vector=True):
#             vector = tuple(item.vector["default"])  # make hashable

#             if vector in seen:
#                 print("THIS IS VERY BAD!! Duplicate vector found.")
#                 count += 1
#             else:
#                 seen.add(vector)

#     finally:
#         client.close()
#         return count

# def check_local_collections():
#     with weaviate.connect_to_local() as client:
#         collections = client.collections.list_all()
#         # # print(collections.keys())
#         # for x in collections:
#         #     print("SEP")
#         #     print("---------------------------- SEP ----------------------------")
#         #     print()
#         #     print()
#         #     print(x)
#         #     print()
#         #     print()
#         #     print("SEP")
#         #     print("---------------------------- SEP ----------------------------")
        
#         print("---------------------------- Collections ----------------------------")
#         print("\n\n\n")
#         print(collections.keys())
#         print("\n\n\n")
#         print("---------------------------- Collections ----------------------------")
        
#         # for x in collections.keys():
#         #     print(x)

# def collection_exists(name: str):
#     exists=False
#     with weaviate.connect_to_local() as client:
#         if client.collections.exists(name=name):
#             exists=True
    
#     return exists


# def query_simple_vec_similarity(
#         collection_name: str, 
#         query: str, 
#         top_k: int = 3, 
#         model_name: str ="PORTULAN/serafim-900m-portuguese-pt-sentence-encoder-ir", 
#         hf_token: str=None
#     ):
#     # !!! OF COURSE !!!
#     # I need to vectorize the query before doing similarity search
#     # The embedding model must be the same as used when the document embeddings were 
#     # generated
#     model = SentenceTransformer(model_name, token=hf_token, trust_remote_code=True)

    
#     query_embedding = model.encode(
#         sentences=query,
#         batch_size=32, #This is the default setting
#         convert_to_numpy=False,
#         normalize_embeddings=False # L2 normalization
#     )


#     query_embedding = query_embedding.tolist()



#     # Variable to hold relevant chunks
#     # relevant_chunks = []
        
#     with weaviate.connect_to_local() as client:

#         collection = client.collections.get(name=collection_name)

#         results = collection.query.near_vector(
#             near_vector=query_embedding,
#             # distance=0.6,  # only return objects within this distance
#             limit=top_k,
#             return_metadata=MetadataQuery(distance=True)  # optionally return distance
#         )
        
#         relevant_chunks = [(obj.properties,obj.metadata.distance) for obj in results.objects] 

#     return relevant_chunks

# def query_bm45_retrieve(name:str, query: str, top_k: int=3):

#     with weaviate.connect_to_local() as client:
#         collection = client.collections.get(name=name)

#         results = collection.query.bm25(
#             query=query,
#             limit=top_k,
#             return_metadata=MetadataQuery(score=True)
#         )
#         relevant_chunks = [(obj.properties, obj.metadata.distance) for obj in results.objects]

#     return relevant_chunks



# def query_hybrid_retrieval(
#         name: str, 
#         query: str, 
#         top_k: int = 3, 
#         model_name: str ="PORTULAN/serafim-100m-portuguese-pt-sentence-encoder",
#         hf_token: str=None
#         ):

#     # Defining the embedding model
#     model = SentenceTransformer(model_name)

#     # Getting the embedding for the query
#     query_embedding = model.encode_query(
#         sentences=query,
#         batch_size=32,
#         convert_to_numpy=False,
#         normalize_embeddings=True # L2 normalization
#     )

#     # Converting the query_embedding from a Tensor into a simple python list as it is what it expects
#     query_embedding = query_embedding.tolist()

#     with weaviate.connect_to_local() as client:
#         collection = client.collections.get(name=name)

#         results = collection.query.hybrid(
#             query=query,
#             vector=query_embedding,
#             alpha = 0.4,
#             # max_vector_distance= 0.8
#             limit=top_k,
#             return_metadata=MetadataQuery(distance=True)  # optionally return distance
#         )        

#         relevant_chunks = [obj.properties for obj in results.objects] 

#     return relevant_chunks

# def query_filter_title_contains():
#     # A function that filters the relevant documents retrived with metadata properies
#     # But witch ones?
#     ...


# def query_my_solution():
#     # My retrieval solution designed specially for PT_wikiRAG
#     ...



# def query_semantic_search_w_reranking():
#     ...

# def query_TEI_server_semanic_search(
#         inference_url:str,
#         collection_name: str, 
#         query: str, 
#         top_k: int = 3,
#     ):
#     # !!! OF COURSE !!!
#     # I need to vectorize the query before doing similarity search
#     # The embedding model must be the same as used when the document embeddings were 
#     # generated
    
#     # Returns a list with each embedding which is also a list
#     query_embedding = get_local_TEI_embbedding_default_API(
#         url=inference_url,
#         text=query
#     )

#     # Getting the query embedding from the list
#     query_embedding = query_embedding[0]

#     # print(type(query_embedding))
#     # print("#####           #####\n#####   DEBUG   #####\n#####           #####")
#     # print("query_embedding:", query_embedding)
#     # print("len(query_embedding):", len(query_embedding))
#     # print("query_embedding[0]:", query_embedding[0])
#     # print("#####           #####\n#####   DEBUG   #####\n#####           #####")
        
#     with weaviate.connect_to_local() as client:

#         collection = client.collections.get(name=collection_name)

#         results = collection.query.near_vector(
#             near_vector=query_embedding,
#             # distance=0.6,  # only return objects within this distance
#             limit=top_k,
#             return_metadata=MetadataQuery(distance=True)  # optionally return distance
#         )
        
#         relevant_chunks = [(obj.properties,obj.metadata.distance) for obj in results.objects] 

#     return relevant_chunks


# def query_STserver_semanic_search(
#         inference_url:str,
#         collection_name: str, 
#         query: str, 
#         top_k: int = 3,
#     ):
#     # !!! OF COURSE !!!
#     # I need to vectorize the query before doing similarity search
#     # The embedding model must be the same as used when the document embeddings were 
#     # generated
    
#     # Returns a list with each embedding which is also a list
#     query_embedding = get_local_STserver_embeddings(
#         inference_url=inference_url,
#         text=query
#     )

#     query_embedding = query_embedding["embeddings"][0]
#     print("query_embedding: ", query_embedding)

#     """
#     might not be needed because get_local_STserver_embeddings -> the server handles passing only one string and getting only an embeddings instead of a list of embeddings
#     """
#     # Getting the query embedding from the list
#     # query_embedding = query_embedding[0]

#     # print(type(query_embedding))
#     # print("#####           #####\n#####   DEBUG   #####\n#####           #####")
#     # print("query_embedding:", query_embedding)
#     # print("len(query_embedding):", len(query_embedding))
#     # print("query_embedding[0]:", query_embedding[0])
#     # print("#####           #####\n#####   DEBUG   #####\n#####           #####")
        
#     with weaviate.connect_to_local() as client:

#         collection = client.collections.get(name=collection_name)

#         results = collection.query.near_vector(
#             near_vector=query_embedding,
#             # distance=0.6,  # only return objects within this distance
#             limit=top_k,
#             return_metadata=MetadataQuery(distance=True)  # optionally return distance
#         )
        
#         relevant_chunks = [(obj.properties,obj.metadata.distance) for obj in results.objects] 

#     return relevant_chunks


