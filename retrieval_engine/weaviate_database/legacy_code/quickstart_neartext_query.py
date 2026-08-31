import weaviate
import json

# client = weaviate.connect_to_local()

# questions = client.collections.use("Question")

# response = questions.query.near_text(
#     query="biology",
#     limit=2
# )

# for obj in response.objects:
#     print(json.dumps(obj.properties, indent=2))

# client.close()  # Free up resources

client = weaviate.connect_to_local()
collection = client.collections.use("DemoCollection")

response = collection.query.near_text(
    query="A holiday film",  # The model provider integration will automatically vectorize the query
    limit=30
)

for obj in response.objects:
    print(obj.properties["title"])

client.close()  # Free up resources