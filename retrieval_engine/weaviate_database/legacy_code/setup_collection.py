import weaviate
import logging
from weaviate.classes.config import Configure

client = weaviate.connect_to_local()
if client.is_ready():
    print("The server is ready!")


docs = client.collections.create(
    name="PT_wikiRAG_Docs_Collection",
    vector_config=[
        Configure.Vectors.text2vec_transformers(
            name="doc-chunk-vec",
            source_properties=["content"],
            pooling_strategy = "masked_mean", # can also be cls https://arxiv.org/abs/1908.10084
            
            # NOTE: Should I make a flask server or something and put it in a container to generate embeddings for my data?
            #       - I think I can just calculate all the embeddings with different models and then save the collection so I don't have to be always calculating embeddings
            # inference_url="http://localhost:8080" # instead of computing embeddings internally, it sends a resquest to this url for the embedding


        )
    ],
    # Additional parameters
)

client.close()