# Weaviate Database — Legacy Code

## How to create a Weaviate database

- Write the `docker-compose.yml` file according to the type of embedding model you want to use, particularly if you want the database itself to generate the embeddings.

- The Weaviate vector database also supports loading your own pre-computed embeddings, which allows use of models not available through the OpenAI API or `text2vec-transformers` (e.g. HuggingFace models).

- This folder contains examples for both embedding variants: OpenAI API and Transformers.

---

## Example `docker-compose.yml`

```yaml
services:
  weaviate:
    image: cr.weaviate.io/semitechnologies/weaviate:1.32.4
    restart: on-failure:0
    ports:
      - "8080:8080"
      - "50051:50051"
    environment:
      QUERY_DEFAULTS_LIMIT: 20
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: "./data"
      DEFAULT_VECTORIZER_MODULE: text2vec-transformers
      ENABLE_MODULES: text2vec-transformers
      TRANSFORMERS_INFERENCE_API: http://text2vec-transformers:8080
      CLUSTER_HOSTNAME: 'node1'
    volumes:
      - weaviate_data:/var/lib/weaviate

  text2vec-transformers:
    image: cr.weaviate.io/semitechnologies/transformers-inference:sentence-transformers-multi-qa-MiniLM-L6-cos-v1
    environment:
      ENABLE_CUDA: "0"   # set to "1" to enable GPU
      # NVIDIA_VISIBLE_DEVICES: all  # uncomment if using GPU
    restart: on-failure:0

volumes:
  weaviate_data:
```
