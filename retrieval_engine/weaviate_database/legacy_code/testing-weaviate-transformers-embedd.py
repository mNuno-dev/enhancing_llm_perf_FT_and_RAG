import weaviate
from weaviate.classes.config import Property, DataType, Configure

client = weaviate.connect_to_local()

# create test collection if it doesn't exist
if not client.collections.exists("SanityCheck"):
    client.collections.create(
        name="SanityCheck",
        properties=[Property(name="content", data_type=DataType.TEXT)],
        vector_config=Configure.Vectors.text2vec_transformers()
    )

col = client.collections.get("SanityCheck")

# insert a doc
# obj_id = col.data.insert({"content": "hello embeddings world","uuid":131231})
obj_id = col.data.insert({"content": "Hoje é dia de comer batatas","uuid":131231})
print(obj_id)


# fetch back with vector
out = col.query.fetch_object_by_id(obj_id, include_vector=True)

results = col.query.fetch_objects(limit=4, include_vector=True)

print("Vector length:", len(out.vector["default"]))  # expect 384 for MiniLM-L6


for obj in results.objects:
    print("content:", obj.properties)
    print("uuid:", obj.uuid)
    # print("embedding:",obj.vector)


client.close()
