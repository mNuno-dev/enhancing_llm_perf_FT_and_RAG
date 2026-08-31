import weaviate

from weaviate.classes.config import Configure, Property, DataType
import weaviate.classes as wvc
import numpy as np


# Openning the client
client = weaviate.connect_to_local()


document_chunks = client.collections.create(
    name="PT_wikiRAG",
    properties=[
        Property(name="doc_name",data_type=DataType.TEXT),
        Property(name="text", data_type=DataType.TEXT)
    ],
    vector_config=wvc.config.Configure.Vectors.self_provided()
    # Additional parameters ...
)

# Example vector from an embedding model
dummy_vector = np.random.rand(384).tolist()

# with client: opens and closes the client automatically
# with client:
collection = client.collections.get("PT_wikiRAG")
collection.data.insert(
    properties={
        "doc_name": "Neural Networks",
        "text": "A neural network is a series of algorithms..."
    },
    vector=dummy_vector
)

query_vector_dummy = np.random.rand(384).tolist()


# with client:
collection = client.collections.get("PT_wikiRAG")
results = collection.query.near_vector(
    near_vector=query_vector_dummy,
    limit=3,
    return_properties=["doc_name","text"]
)

for o in results.objects:
    print(o.properties)


# Closing the client
client.close()