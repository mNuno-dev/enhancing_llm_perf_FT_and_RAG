from typing import List, Dict, Literal, Union, Callable, Optional, Any, Sequence, Iterable, Tuple

import weaviate
from weaviate.classes.query import MetadataQuery
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
import weaviate.classes as wvc
import numpy as np

from sentence_transformers import SentenceTransformer
from embeddings.embeddings import get_local_TEI_embbedding_default_API, get_local_STserver_embeddings

import logging
import hashlib

# Configuring a logger for this module
logging.basicConfig(level=logging.info)
logger = logging.getLogger(__name__)



# -----------------------
# Custom exceptions
# -----------------------
class WeaviateError(Exception):
    """Base class for module-specific exceptions."""




class CollectionNotFound(WeaviateError):
    pass




class CollectionExists(WeaviateError):
    pass




class VectorValidationError(WeaviateError):
    pass



### Weaviate Collection Operations Functions ###


# -------------------------
# Helpers
# -------------------------

def _connect():
    """Returns a weaviate client connected to local instance.
    If you want to use a remote instance change this helper in one place.
    """
    # keep context manager usage at call sites. This function just returns the client.
    return weaviate.connect_to_local()


# -------------------------
# Collection management
# -------------------------

def create_collection(
    collection_name:str,
    hnsw: bool = True,
    distance_metric: VectorDistances = VectorDistances.COSINE,
    properties: Optional[List[Property]] = None,
) -> None:
    if properties is None:
        properties = [Property(name="doc_name", data_type=DataType.TEXT), Property(name="text", data_type=DataType.TEXT)]
    
    with _connect() as client:
        if client.collections.exists(name=collection_name):
            raise CollectionExists(f"Collection '{collection_name}' already exists.")
        
        if hnsw:
            vector_config = wvc.config.Configure.Vectors.self_provided()
        else:
            vector_config = Configure.Vectors.self_provided(
                vector_index_config=Configure.VectorIndex.flat(distance_metric=distance_metric)
            )

        client.collections.create(name=collection_name, properties=properties, vector_config=vector_config)
        logger.info("Created collection '%s' (hnsw=%s)", collection_name, hnsw)
    

def delete_connection(collection_name:str) -> None:
    """Delete a collection; raise if not exists."""
    with _connect() as client:
        if not client.collections.exists(name=collection_name):
            raise CollectionNotFound(f"Collection '{collection_name}' could not be found.")
        
        client.collections.delete(name=collection_name)
        logger.info("Deleted collection '%s'", collection_name)
    

def delete_all_collections() -> List[str]:
    """"""
    with _connect() as client:
        collections = List(client.collections.list_all().keys())
        client.collections.delete_all()
        logger.info("Deleted all collections: %s", collections)
        return collections
    

def list_collections() -> List[str]:
    with _connect as client:
        return List(client.collections.list_all().keys())


def collection_exists(collection_name):
    with _connect() as client:
        return client.collections.exists(name=collection_name)

# -------------------------
# Data insertion
# -------------------------     
        
def add_own_vectors(
    collection_name:str,
    vectors: Iterable[Dict[str, Any]],
    batch_size: int = 200,
    raise_on_failure: bool = True,
) -> ... :
    """
    Add vectors to a collection in batches

    Returns (added_count, failed_items_or_None)
    """
    added = 0
    failed = []

    with _connect() as client:
        if not client.collections.exists(name=collection_name):
            raise CollectionNotFound(f"Collection '{collection_name}' does not exist. Available Collections: {list(client.collections.list_all().keys())}")
        
        collection = client.collections.get(name=collection_name)

        with collection.batch.fixed_size(batch_size=batch_size) as batch:
            for item in vectors:
                try:
                    batch.add_object(
                        properties= {"doc_name": item["doc_name"], "text": item["text"]},
                        vector = item["vector"]
                    )
                except Exception as e: # catch-weaviate errors per vector
                    logger.exception("Failed to add object to batch: %s", e)
                    failed.append({"item": item, "error": str(e)})

        # inspect batch failures reported by the SDK
        sdk_failed = collection.batch.failed_objects
        if sdk_failed:
            logger.error("Weaviate SDK reported failed objects: %s", sdk_failed[:3])
            for f in sdk_failed:
                failed.append({"item": getattr(f, "object", None), "error": getattr(f, "error", None)})

    if failed and raise_on_failure:
        raise RuntimeError(f"Failed to insert {len(failed)} items; first error: {failed[0]}")
    
    return added, (failed if failed else None)

# -------------------------
# Collection inspection
# -------------------------

def get_collection_info(name: str) -> Dict[str, Any]:
    """Return a small summary about the collection (name and length).


    Note: len(collection) may be implemented by the SDK and could require a round-trip.
    """
    with _connect() as client:
        if not client.collections.exists(name=name):
            raise CollectionNotFound(name)
        coll = client.collections.get(name)
        try:
            size = len(coll)
        except Exception:
            size = None
        return {"name": coll.name, "size": size}
    

def first_n_embeddings(name: str, n: int = 3, vector_name: str = "default") -> List[List[float]]:
    """Return the first n vectors for inspection. Raises if collection empty.


    Handles StopIteration and absent vector field gracefully.
    """
    with _connect() as client:
        if not client.collections.exists(name=name):
            raise CollectionNotFound(name)
        coll = client.collections.get(name)
        iterator = coll.iterator(include_vector=True)


        results: List[List[float]] = []
        for i in range(n):
            try:
                item = next(iterator)
            except StopIteration:
                break


            if not getattr(item, "vector", None):
                logger.warning("Item has no .vector attribute; skipping")
                continue


            vec = item.vector.get(vector_name)
            if vec is None:
                raise WeaviateError(f"Vector name '{vector_name}' not found on object")
            results.append(list(vec))


        if not results:
            raise WeaviateError("No vectors found in collection")


        return results


def detect_duplicate_vectors(name: str, sample: Optional[int] = 10_000, vector_name: str = "default") -> int:
    """Detect duplicate vectors in a memory-efficient manner.

    - If sample is None, check all items (dangerous for very large collections).
    - Otherwise check up to `sample` items.
    - Uses a hash of float32 bytes for constant memory per stored hash.
    Returns the number of duplicates seen in the inspected set.
    """
    seen = set()
    duplicates = 0


    with _connect() as client:
        if not client.collections.exists(name=name):
            raise CollectionNotFound(name)
        coll = client.collections.get(name)
        iterator = coll.iterator(include_vector=True)


        checked = 0
        for item in iterator:
            if sample is not None and checked >= sample:
                break


            checked += 1
            vec = getattr(item, "vector", None)
            if not vec:
                continue
            arr = vec.get(vector_name)
            if not arr:
                continue


            # convert to float32 bytes for consistent hashing
            try:
                arr_np = np.asarray(arr, dtype=np.float32)
            except Exception:
                continue


            h = hashlib.sha1(arr_np.tobytes()).hexdigest()
            if h in seen:
                duplicates += 1
            else:
                seen.add(h)

    return duplicates


# -------------------------
# Query helpers
# -------------------------

def _query_near_vector(
    collection_name:str,
    vector: Sequence[float],
    top_k: int = 3,
    return_distance: bool = True
) ->  List[Tuple[Dict[str, Any], float]]: 
    """Generic near_vector query wrapper returning (properties, distance) tuples."""

    with _connect() as client:
        if not client.collections.exists(name=collection_name):
            raise CollectionNotFound(collection_name)
        
        collection = client.collections.get(collection_name)

        results = collection.query.near_vector(
            near_vector=vector,
            limit=top_k,
            return_metadata=MetadataQuery(distance=return_distance)
        )

        objs = []
        for obj in getattr(results, "objects", []) or []:
            dist = getattr(obj, "metadata", None)
            if dist is not None and hasattr(dist, "distance"):
                objs.append((obj.properties, dist.distance))
            else:
                objs.append((objs.properties,None))
        
        return objs


def query_simple_vec_similarity(
    collection_name:str,
    query_text:str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    trust_remote_code: bool = False,
    hf_token: Optional[str] = None,
    top_k: int = 3,
    normalize: bool = False,
) -> List[Tuple[Dict[str, Any], Optional[float]]]:
    """Encode a query with SentenceTransformer and perform near_vector search.


    Notes:
    - convert_to_numpy=True returns a numpy array which is then converted to list.
    - normalize controls L2 normalization before sending the vector to Weaviate.
    """
    # load model (trust_remote_code only if needed by model)
    model = SentenceTransformer(model_name, use_auth_token=hf_token, trust_remote_code=trust_remote_code)

    embedding = model.encode([query_text], convert_to_numpy=True)
    vec = np.asarray(embedding[0], dtype=float)
    if normalize:
        norm = np.linalg.norm(vec)
        if norm != 0:
            vec = vec / norm

    return _query_near_vector(collection_name, vec.tolist(), top_k=top_k)

def query_bm25(collection_name: str, query: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], Optional[float]]]:
    with _connect() as client:
        if not client.collections.exists(name=collection_name):
            raise CollectionNotFound(collection_name)
        coll = client.collections.get(collection_name)

        results = coll.query.bm25(query=query, limit=top_k, return_metadata=MetadataQuery(score=True))
        objs = []
        for obj in getattr(results, "objects", []) or []:
            score = getattr(getattr(obj, "metadata", None), "score", None)
            objs.append((obj.properties, score))
        return objs
    

def query_hybrid(
collection_name: str,
query_text: str,
model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
hf_token: Optional[str] = None,
top_k: int = 3,
alpha: float = 0.5,
normalize: bool = True,
) -> List[Tuple[Dict[str, Any], Optional[float]]]:
    """Perform a hybrid semantic+BM25 query.

    alpha -> controls the weighting between text and vector similarity (SDK-specific semantics).
    """
    model = SentenceTransformer(model_name, use_auth_token=hf_token)
    embedding = model.encode([query_text], convert_to_numpy=True)
    vec = np.asarray(embedding[0], dtype=float)
    if normalize:
        n = np.linalg.norm(vec)
        if n != 0:
            vec = vec / n

    with _connect() as client:
        if not client.collections.exists(name=collection_name):
            raise CollectionNotFound(collection_name)

        collection = client.collections.get(collection_name)

        results = collection.query.hybrid(query=query_text, vector=vec.tolist(), alpha=alpha, limit=top_k, return_metadata=MetadataQuery(distance=True))
        objs = []
        for obj in getattr(results, "objects", []) or []:
            dist = getattr(getattr(obj, "metadata", None), "distance", None)
            objs.append((obj.properties, dist))

    return objs

def query_with_external_embedding(
    embedder_fn, # callable(inference_url, text) -> sequence or dict containing embeddings
    inference_url: str,
    collection_name: str,
    query_text: str,
    top_k: int = 3,
    embeddings_key: Optional[str] = None,
) -> List[Tuple[Dict[str, Any], Optional[float]]]:
    """Generic wrapper to handle locally hosted embedding servers (TEI / STserver).

    - embedder_fn should accept (inference_url, text) and return either a sequence of
    floats or a dict containing an embeddings list (common patterns supported).

    - embeddings_key: if the response is a dict, use this key to extract the vector list.
    """
    
    resp = embedder_fn(inference_url=inference_url, text=query_text)

    # try to extract vector in flexible ways
    if isinstance(resp, dict) and embeddings_key:
        # e.g. {'embeddings': [[...]]}
        embedding = resp.get(embeddings_key)
        if embedding is None:
            raise WeaviateError(f"Key '{embeddings_key}' not found in embedder response")
        # if embeddings is a list of lists, take first
        if isinstance(embedding, list) and len(embedding) and isinstance(embedding[0], (list, tuple)):
            vec = embedding[0]
        else:
            vec = embedding
    elif isinstance(resp, (list, tuple)):
        # assume embedder returned a list-of-lists or list vector
        if len(resp) and isinstance(resp[0], (list, tuple)):
            vec = resp[0]
        else:
            vec = resp
    else:
        raise WeaviateError("Unrecognized embedding response format")

    return _query_near_vector(collection_name, list(map(float, vec)), top_k=top_k)


# -------------------------
# Unimplemented / placeholder functions
# -------------------------


def query_filter_title_contains(*args, **kwargs):
    raise NotImplementedError("This helper was left intentionally unimplemented. Please describe the filter semantics you need.")




def query_my_solution(*args, **kwargs):
    raise NotImplementedError("Custom retrieval solution placeholder")




def query_semantic_search_w_reranking(*args, **kwargs):
    raise NotImplementedError("Reranking pipeline placeholder")


