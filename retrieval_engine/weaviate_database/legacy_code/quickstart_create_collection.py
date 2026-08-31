import weaviate
from weaviate.classes.config import Configure

client = weaviate.connect_to_local()

# questions = client.collections.create(
#     name="Question",
#     vector_config=Configure.Vectors.text2vec_ollama(  # Configure the Ollama embedding integration
#         api_endpoint="http://ollama:11434",  # If using Docker you might need: http://host.docker.internal:11434
#         model="nomic-embed-text",  # The model to use
#     ),
#     generative_config=Configure.Generative.ollama(  # Configure the Ollama generative integration
#         api_endpoint="http://ollama:11434",  # If using Docker you might need: http://host.docker.internal:11434
#         model="llama3.2",  # The model to use
#     ),
# )

# client.close()  # Free up resources

questions = client.collections.create(
    name="DemoCollecitonclear",
    vector_config=[
        Configure.Vectors.text2vec_transformers(
            name="title_vector",
            source_properties=["title"],
            pooling_strategy = "masked_mean", # can also be cls https://arxiv.org/abs/1908.10084

            # NOTE: Should I make a flask server or something and put it in a container to generate embeddings for my data?
            #       - I think I can just calculate all the embeddings with different models and then save the collection so I don't have to be always calculating embeddings
            # inference_url="http://localhost:8080" # instead of computing embeddings internally, it sends a resquest to this url for the embedding

            # It might be useful IF -> You want to integrate with existing inference servers like Ollama, vLLM, Hugging Face TGI, or custom Flask/FastAPI embedding services.

        )
    ],
    # Additional parameters
)

client.close()