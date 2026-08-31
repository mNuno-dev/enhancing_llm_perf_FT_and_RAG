from typing import List, Dict, Literal, Union, Callable, Optional, Any
import logging
import re
import os
import glob
import torch
import unicodedata
import numpy as np
import time

import torch

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

from utils.utils import load_jsonl_PT_wikiRAG_locally, loading_PT_wikiRAG_from_HF, normalize_str
from chunking.chunking import fixed_size_chunking, fixed_size_chunking_with_overlap, variable_size_chunking_character_based, special_mixed_chunking

import pprint

import requests
import json
from openai import OpenAI
import tqdm


###
### Initializing the logger
###
logger = logging.getLogger(__name__)


def batched_sliding_window_chunk_embedding(
    text: str,
    model_name: Optional[str] = None,
    model: Optional[SentenceTransformer] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    max_tokens: Optional[int] = None,
    stride: Optional[int] = None,
    pooling: str = "mean",
    batch_size: int = 32,
    verbose: bool = False,
    hf_token: Optional[str] = None
) -> np.ndarray:
    """
    Generate embeddings for long text using sliding windows + stream pooling.
    Optimizied for both speed (batching) and memory (no big number of arrays)
    """
    
    # device detection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"


    # load model/tokenizer if not provided
    if model is None:
        if model_name is None:
            raise ValueError("Either model or model_name must be provided.")
        model = SentenceTransformer(model_name_or_path=model_name, device=device, token=hf_token)
    if tokenizer is None:
        if model_name is None:
            raise ValueError("Either tokenizer or model_name must be provided.")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    # stride default and safety clamp
    if stride is None:
        stride = max(1, max_tokens // 2)  # 50% overlap default
    else:
        stride = max(1, min(stride, max_tokens))
        
    # tokenize input
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    seq_len = len(input_ids)
    if seq_len == 0:
        raise ValueError("Tokenized text is empty.")
    
    pooled = None
    count = 0
    # sliding window batching
    start = 0
    batch_chunks = []
    while start < seq_len:
        end = min(start + max_tokens, seq_len)
        chunk_ids = input_ids[start:end].tolist()
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()

        if chunk_text:
            batch_chunks.append(chunk_text)
            
            # process when batch is full
            if len(batch_chunks) >= batch_size:
                embeddings = model.encode(batch_chunks, convert_to_numpy=True, show_progress_bar=True)
                for emb in embeddings:
                    if pooled is None:
                        pooled = emb
                    else:
                        if pooling in ("mean", "sum"):
                            pooled += emb
                        elif pooling == "max":
                            pooled = np.maximum(pooled, emb)
                    count += 1
                batch_chunks = []  # reset

        if end == seq_len:
            break
        start += stride

    # process any leftovers
    if batch_chunks:
        embeddings = model.encode(batch_chunks, convert_to_numpy=True, show_progress_bar=False)
        for emb in embeddings:
            if pooled is None:
                pooled = emb
            else:
                if pooling in ("mean", "sum"):
                    pooled += emb
                elif pooling == "max":
                    pooled = np.maximum(pooled, emb)
            count += 1
    
    if pooled is None:
        raise ValueError("No non-empty chunks to encode.")

    # final pooling
    if pooling == "mean":
        return pooled / count
    elif pooling == "sum":
        return pooled
    elif pooling == "max":
        return pooled
    else:
        raise ValueError(f"Unknown pooling '{pooling}'")


def generate_sliding_window_chunk_embedding(
    text: str,
    model_name: Optional[str] = None,
    model: Optional[SentenceTransformer] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    max_tokens: Optional[int] = None,
    stride: Optional[int] = None,
    pooling: str = "mean",
    verbose: bool = False,
    hf_token: Optional[str] = None
) -> np.ndarray:
    """
    Generate embeddings for long text using sliding windows.

    You can either pass `model` and `tokenizer` objects (recommended when calling many times),
    or pass `model_name` to load them inside the function.

    Returns a single pooled embedding (or raises on invalid inputs).
    """
    
    """
    Example:
    max_tokens = 8
    stride = 4
    20 tokens chunk:
    T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15 T16 T17 T18 T19 T20
    
    Iteration 1
    [T1 T2 T3 T4 T5 T6 T7 T8] T9 T10 T11 T12 T13 T14 T15 T16 T17 T18 T19 T20
    
    Iteration 2
    T1 T2 T3 T4 [T5 T6 T7 T8 T9 T10 T11 T12] T13 T14 T15 T16 T17 T18 T19 T20

    --------------   ////   --------------
    NOTE: 
    - max_tokens should normally be <= the model/tokenizer model_max_length to avoid truncation
    - stride controls overlap: overlap = max_tokens - stride
        - if stride == max_tokens -> no overlap
    """    

    # device detection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # load model/tokenizer if not provided
    if model is None:
        if model_name is None:
            raise ValueError("Either model or model_name must be provided.")
        model = SentenceTransformer(model_name_or_path=model_name, device=device, token=hf_token)
    if tokenizer is None:
        if model_name is None:
            raise ValueError("Either tokenizer or model_name must be provided.")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    # determine max_tokens if not provided, clamp to tokenizer/model limit
    # model_max = getattr(tokenizer, "model_max_length", 512)
    # print("model_max", model_max)
    # if model_max is None or model_max > 10**9:
    #     # sometimes tokenizers give huge placeholder; use a safe default
    #     model_max = 4096
    # if max_tokens is None:
    #     max_tokens = model_max
    # else:
    #     max_tokens = min(max_tokens, model_max)
    
    # print("\n\nMAX TOKENS: \n\n", max_tokens)

    # stride default and safety clamp
    if stride is None:
        stride = max(1, max_tokens // 2)  # 50% overlap default
    else:
        stride = max(1, min(stride, max_tokens))

    # tokenize and prepare chunk texts
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    seq_len = len(input_ids)
    if seq_len == 0:
        raise ValueError("Tokenized text is empty.")

    # NOTE: Non streaming pooling (problematic since it stores many chunks into memory)
    # chunk_texts: List[str] = []
    # start = 0
    # while start < seq_len:
    #     end = min(start + max_tokens, seq_len)
    #     chunk_ids = input_ids[start:end].tolist()
    #     chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
    #     if chunk_text:
    #         chunk_texts.append(chunk_text)
    #     if end == seq_len:
    #         break
    #     start += stride
    # if verbose:
    #     print(f"seq_len={seq_len}, max_tokens={max_tokens}, stride={stride}, n_chunks={len(chunk_texts)}")

    # if len(chunk_texts) == 0:
    #     raise ValueError("No non-empty chunks to encode.")
    
    
    # Streaming pooling
    pooled = None
    count = 0
    start = 0
    while start < seq_len:
        end = min(start + max_tokens, seq_len)
        chunk_ids = input_ids[start:end].tolist()
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        
        if chunk_text:
            emb = model.encode(chunk_text, convert_to_numpy=True)
            # NOTE! This step optimizes memory usage by not saving all the diffrent numpy.arrays
            #           and instead using a single array to sum or get the maximum according to the pooling strategy
            if pooled is None:
                pooled = emb
            else:
                if pooling == "mean" or pooling == "sum":
                    pooled += emb # adding the embedding tensor into a single (pooled) numpy.array that can also be pooled in the end
                elif pooling == "max":
                    pooled = np.maximum(pooled, emb)
            count += 1 # count each time a "stream pooling" operation is carried through
            
            if end == seq_len:
                break
            start += stride
        
    if pooled is None:
        raise ValueError("No non-empty chunks to encode.")

    # Final pooling with the saved ("stream pooling")
    if pooling == "mean":
        return pooled / count
    elif pooling == "sum":
        return pooled
    elif pooling == "max":
        return pooled
    else:
        raise ValueError(f"Unknown pooling '{pooling}'")
    
    
    
    
    
    
# def generate_sliding_window_chunk_embedding(
#     text: str,
#     model_name: Optional[str] = None,
#     model: Optional[SentenceTransformer] = None,
#     tokenizer: Optional[AutoTokenizer] = None,
#     max_tokens: Optional[int] = None,
#     stride: Optional[int] = None,
#     pooling: str = "mean",
#     batch_encode: bool = True,
#     batch_size: int = 32,
#     verbose: bool = False,
# ) -> np.ndarray:
#     """
#     Generate embeddings for long text using sliding windows.

#     You can either pass `model` and `tokenizer` objects (recommended when calling many times),
#     or pass `model_name` to load them inside the function.

#     Returns a single pooled embedding (or raises on invalid inputs).
#     """
    
#     """
#     Example:
#     max_tokens = 8
#     stride = 4
#     20 tokens chunk:
#     T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15 T16 T17 T18 T19 T20
    
#     Iteration 1
#     [T1 T2 T3 T4 T5 T6 T7 T8] T9 T10 T11 T12 T13 T14 T15 T16 T17 T18 T19 T20
    
#     Iteration 2
#     T1 T2 T3 T4 [T5 T6 T7 T8 T9 T10 T11 T12] T13 T14 T15 T16 T17 T18 T19 T20

#     --------------   ////   --------------
#     NOTE: 
#     - max_tokens should normally be <= the model/tokenizer model_max_length to avoid truncation
#     - stride controls overlap: overlap = max_tokens - stride
#         - if stride == max_tokens -> no overlap
#     """    
    
#     # device detection (I should make a utils function for this)
#     if torch.backends.mps.is_available():
#         device = "mps"
#     elif torch.cuda.is_available():
#         device = "cuda"
#     else:
#         device = "cpu"

#     # load model/tokenizer if not provided
#     if model is None:
#         if model_name is None:
#             raise ValueError("Either model or model_name must be provided.")
#         model = SentenceTransformer(model_name_or_path=model_name, device=device)
#     if tokenizer is None:
#         if model_name is None:
#             raise ValueError("Either tokenizer or model_name must be provided.")
#         tokenizer = AutoTokenizer.from_pretrained(model_name)

#     # determine max_tokens if not provided, clamp to tokenizer/model limit
#     # model_max = getattr(tokenizer, "model_max_length", 512)
#     # print("model_max", model_max)
#     # if model_max is None or model_max > 10**9:
#     #     # sometimes tokenizers give huge placeholder; use a safe default
#     #     model_max = 4096
#     # if max_tokens is None:
#     #     max_tokens = model_max
#     # else:
#     #     max_tokens = min(max_tokens, model_max)
#     # print("\n\nMAX TOKENS: \n\n", max_tokens)

#     # stride default and safety clamp
#     if stride is None:
#         stride = max(1, max_tokens // 2)  # 50% overlap default
#     else:
#         stride = max(1, min(stride, max_tokens))

#     # tokenize and prepare chunk texts
#     input_ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
#     seq_len = len(input_ids)
#     if seq_len == 0:
#         raise ValueError("Tokenized text is empty.")

#     # NOTE: Non streaming pooling (problematic since it stores many chunks into memory)
#     chunk_texts: List[str] = []
#     start = 0
#     while start < seq_len:
#         end = min(start + max_tokens, seq_len)
#         chunk_ids = input_ids[start:end].tolist()
#         chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
#         if chunk_text:
#             chunk_texts.append(chunk_text)
#         if end == seq_len:
#             break
#         start += stride
#     if verbose:
#         print(f"seq_len={seq_len}, max_tokens={max_tokens}, stride={stride}, n_chunks={len(chunk_texts)}")

#     if len(chunk_texts) == 0:
#         raise ValueError("No non-empty chunks to encode.")
#     # encode all chunks (batch)
#     if batch_encode:
#         chunk_embeddings = model.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=True, batch_size=batch_size)
#         # print("l->chunk_texts",len(chunk_texts))
#         # print("l->chunk_embeddings",len(chunk_embeddings))
#     else:
#         chunk_embeddings = np.array([model.encode(t, convert_to_numpy=True) for t in chunk_texts])

#     # pooling options
#     if pooling == "mean":
#         pooled = np.mean(chunk_embeddings, axis=0)
#     elif pooling == "sum":
#         pooled = np.sum(chunk_embeddings, axis=0)
#     elif pooling == "max":
#         pooled = np.max(chunk_embeddings, axis=0) #max is an alias for amax
#     elif pooling == "none":
#         return chunk_embeddings  # return the raw chunk embeddings
#     else:
#         raise ValueError(f"Unknown pooling '{pooling}'")

#     return pooled







def get_PTwikiRAG_docSegmentation_embeddings(
    model_name: str,
    batch_size: int = 32,
    max_tokens: Optional[int] = 128,
    sliding_window: bool = True,
    pooling: Optional[Literal["mean", "sum", "max", None]] = "mean",
    stride: Optional[int] = None,
    hf_token: Optional[str] = None
):
    """
    Generates embeddings for PT-wikiRAG documents using the documents present on the PTwikiRAG HFdataset
    ----> Generates a chunk for each "section" ("\n\n\n" splitting)
    ---> This criteria takes into account a pattern found across all files in the dataset where "\n\n\n" represents a change of subject or related content
    --> Each embedding corresponds to a secction criteria of "\n\n\n" for splitting
    -> It is advised to use sliding_window, even for models with long max_seq, to prevent truncation of text during the embedding calculation, and therefore preventing loss of context
    Note: It starts the embedding model locally on CPU/GPU/MPS. (So take caution for larger models)
    Args:
        files_dir: Directory where the LXsentenceSplit files are located
        model_name: Name of embedding model.
        batch_size: Encoding batch_size, for larger chunks it might be needed to lower the batch_size if memory is contrained
        max_tokens: A parameter that decides the maximum number of tokens a chunk can take (for sliding window) it should respect the model's max_seq_length
        sliding_window: Choose between applying sliding window on embeddings or default embeddings that do truncation when the embedding is too long
        pooling: Choose a type of pooling for sliding window embeddings
        stride: Optional stride value, the amount of tokens that are moved forward during sliding window -> Defaults to half of max_tokens if not declared
        hf_token: Optional HuggingFace token if loading from HF.

    Returns:
        A list of dictionaries containing:
        {
            "doc_id": str,
            "doc_name": str,
            "text": str,
            "vector": List[float]
        }
    """
    dataset = loading_PT_wikiRAG_from_HF(token=hf_token)
    
    data = dataset["subsplit1"]

    list_chunk_dicts = [
        {
            "doc_id": doc["doc_id"],
            "doc_name": doc["doc_name"],
            "text": chunk
        
        }
        for doc in data
        for chunk in variable_size_chunking_character_based(text=doc["content"], characters=["\n\n\n"], regex_mode=False)
    ]    
    
    chunks_text_only = [chunk["text"] for chunk in list_chunk_dicts]
    
    if not chunks_text_only:
        raise ValueError("No valid text chunks produced. Check chunking or input data.")
    
    
    
    # device detection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    # initialize model and tokenizer only once
    model = SentenceTransformer(model_name_or_path=model_name, device=device, trust_remote_code=True, token=hf_token)
    
    if sliding_window:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, token=hf_token)
        embeddings = [
            batched_sliding_window_chunk_embedding(text=t,model=model,tokenizer=tokenizer,batch_size=batch_size, max_tokens=max_tokens, stride=stride, pooling=pooling).tolist()
            for t in chunks_text_only
        ]
    else:
        embeddings = generate_embeddings_w_model(text=chunks_text_only, model=model, batch_size=batch_size, hf_token=hf_token)
        
    
    final_chunks_w_embeddings = [
        {**chunk_dict, "vector": embeddings[i]}
        for i, chunk_dict in enumerate(list_chunk_dicts)
    ]
    
    return final_chunks_w_embeddings
    




def get_PTwikiRAG_simple_line_chunking_embeddings(
    model_name: str,
    batch_size: int = 32,
    max_tokens: Optional[int] = 128,
    sliding_window: bool = True,
    pooling: Optional[Literal["mean", "sum", "max", None]] = "mean",
    stride: Optional[int] = None,
    hf_token: Optional[str] = None
):
    """
    Generate embeddings for PT-wikiRAG documents using the documents present on the PTwikiRAG HFdataset
    ---> Generates a chunk for each line/paragraph ("\n" splitting)
    --> Each embedding corresponds to a line/paragraph using the criteria of "\n" for splitting
    Note: It starts the embedding model locally on CPU/GPU/MPS. (So take caution for larger models)
    Args:
        files_dir: Directory where the LXsentenceSplit files are located
        model_name: Name of embedding model.
        batch_size: Encoding batch_size, for larger chunks it might be needed to lower the batch_size if memory is contrained
        max_tokens: A parameter that decides the maximum number of tokens a chunk can take (for sliding window) it should respect the model's max_seq_length
        sliding_window: Choose between applying sliding window on embeddings or default embeddings that do truncation when the embedding is too long
        pooling: Choose a type of pooling for sliding window embeddings
        stride: Optional stride value, the amount of tokens that are moved forward during sliding window -> Defaults to half of max_tokens if not declared
        hf_token: Optional HuggingFace token if loading from HF.

    Returns:
        A list of dictionaries containing:
        {
            "doc_id": str,
            "doc_name": str,
            "text": str,
            "vector": List[float]
        }
    """
    
    dataset = loading_PT_wikiRAG_from_HF(token=hf_token)
    
    data = dataset["subsplit1"]

    list_chunk_dicts = [
        {
            "doc_id": doc["doc_id"],
            "doc_name": doc["doc_name"],
            "text": chunk
        
        }
        for doc in data
        for chunk in variable_size_chunking_character_based(text=doc["content"], characters=["\n"], regex_mode=False)
    ]    

    
    # NOTE: variable_size_chunking_character_based already handles empty chunks, so I don't need to worry about that here
    chunks_text_only = [chunk["text"] for chunk in list_chunk_dicts] 
    
    if not chunks_text_only:
        raise ValueError("No valid text chunks produced. Check chunking or input data.")
    
    
    
    # device detection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    # initialize model and tokenizer only once
    model = SentenceTransformer(model_name_or_path=model_name, device=device, trust_remote_code=True, token=hf_token)
    
    if sliding_window:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
        embeddings = [
            batched_sliding_window_chunk_embedding(text=t,model=model,tokenizer=tokenizer,batch_size=batch_size, max_tokens=max_tokens, stride=stride, pooling=pooling).tolist()
            for t in chunks_text_only
        ]
    else:
        embeddings = generate_embeddings_w_model(text=chunks_text_only, model=model, batch_size=batch_size, hf_token=hf_token)
        
    
    final_chunks_w_embeddings = [
        {**chunk_dict, "vector": embeddings[i]}
        for i, chunk_dict in enumerate(list_chunk_dicts)
    ]
    
    return final_chunks_w_embeddings


# def get_PTwikiRAG_docSegmentation_embeddings(model: str, batch_size: int):
#     """
#     The one where you take into account the document format ("\n\n\n") // ("\n\n")
#     """
#     dataset = loading_PT_wikiRAG_from_HF(token=TOKEN_HF)
    
#     data = dataset["subsplit1"]

#     list_chunk_dicts = [
#         {
#             "doc_id": doc["doc_id"],
#             "doc_name": doc["doc_name"],
#             "text": chunk
        
#         }
#         for doc in data
#         for chunk in variable_size_chunking_character_based(text=doc["content"], characters=["\n\n\n"], regex_mode=False)
#     ]    
    
#     chunks_text_only = [chunk["text"] for chunk in list_chunk_dicts]
#     if not chunks_text_only:
#         raise ValueError("No valid text chunks produced. Check chunking or input data.")
    
#     chunk_embeddings = generate_embeddings_w_model(text=chunks_text_only, model=model, batch_size=batch_size)
    
#     final_l_chunks = [
#         {**chunk_dict, "vector": chunk_embeddings[i]}  for i, chunk_dict in enumerate(list_chunk_dicts) 
#     ]
    
    
#     return final_l_chunks
    
    
    
        



def get_PT_wikiRAG_LXsentenceSplit_embeddings(
    files_dir: Optional[str] = None,
    model_name: str = "PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir",
    batch_size: int = 32,
    max_tokens: Optional[int] = 128,
    sliding_window: bool = True,
    pooling: Optional[Literal["mean", "sum", "max", None]] = "mean",
    stride: Optional[int] = None,
    hf_token: Optional[str] = None
):
    """
    Generate embeddings for PT-wikiRAG documents using files processed by the LX-Sentence-Spliter (PORTULAN)
    -- This means that each embedding with this strategy corresponds to a sentence (decided by the LX-Sentence-Spliter)
    Note: It starts the embedding model locally on CPU/GPU/MPS. (So take caution for larger models)
    Args:
        files_dir: Directory where the LXsentenceSplit files are located
        model_name: Name of embedding model.
        batch_size: Encoding batch_size, for larger chunks it might be needed to lower the batch_size if memory is contrained
        max_tokens: A parameter that decides the maximum number of tokens a chunk can take (for sliding window) it should respect the model's max_seq_length
        sliding_window: Choose between applying sliding window on embeddings or default embeddings that do truncation when the embedding is too long
        pooling: Choose a type of pooling for sliding window embeddings
        stride: Optional stride value, the amount of tokens that are moved forward during sliding window -> Defaults to half of max_tokens if not declared
        hf_token: Optional HuggingFace token if loading from HF.

    Returns:
        A list of dictionaries containing:
        {
            "doc_id": str,
            "doc_name": str,
            "text": str,
            "vector": List[float]
        }
    """
    
    if files_dir is None:
        raise ValueError(
            "files_dir must be provided. Set it to the directory containing "
            "the LX-Sentence-Splitter output .txt files."
        )

    dataset = loading_PT_wikiRAG_from_HF(token=hf_token)

    data = dataset["subsplit1"]

    doc_name_and_id_dict = { normalize_str(doc["doc_name"]): doc["doc_id"] for doc in data }

    if not os.path.exists(files_dir):
        raise ValueError(f"files_dir '{files_dir}' does not exist.")

    file_paths = glob.glob(files_dir + "/*")

    # device detection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    model = SentenceTransformer(model_name_or_path=model_name, device=device, token=hf_token)
    
    if sliding_window: 
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, token=hf_token)

    all_embeddings = [] # collect results across all files
    
    for file_path in file_paths:
        file = normalize_str(os.path.basename(file_path))
        if file in doc_name_and_id_dict.keys():
            # print("LEZGO")
            doc_name = file
            doc_id = doc_name_and_id_dict[file]
            # print("doc_name: ", doc_name)
            # print("doc_id: ", doc_id)
            # count += 1
        # else:            
            # raise ValueError(f"Unexpected file found at {file_path}")
        with open(file_path, "r", encoding="UTF-8") as f:
            # strip whitespace and skip empty lines
            non_empty_lines = [line.strip() for line in f if line.strip()]
        # print(f"{file_path}: {len(non_empty_lines)} non-empty lines")
        
        chunks_in_dict = [
            {
            "doc_name": doc_name,
            "doc_id": doc_id,
            "text": text,
            }
            for text in non_empty_lines
        ]
        
        # print(len(chunks_in_dict) == len(non_empty_lines))
        text_chunks = non_empty_lines
        
        if sliding_window:
            text_chunks_embeddings = [
                batched_sliding_window_chunk_embedding(
                    text=t,
                    model=model,
                    tokenizer=tokenizer,
                    max_tokens=max_tokens,
                    batch_size=batch_size,
                    stride=stride,
                    pooling=pooling,
                    hf_token=hf_token       
                ).tolist() # tolist() because batched_sliding_window_chunk_embeddings returns np.array 
                
                for t in text_chunks
            ]
        else:
            text_chunks_embeddings = generate_embeddings_w_model(
                text=text_chunks,
                model=model,
                batch_size=batch_size,
                hf_token=hf_token
            )
        
        # # chunk_embeddings = generate_embeddings_w_model(model=model, text=non_empty_lines, batch_size=batch_size)
        # text_chunks_embeddings = []
        # for text in text_chunks:
        #     embedding = batched_sliding_window_chunk_embedding(text=text,model_name=model,max_tokens=128,batch_size=32)
        #     text_chunks_embeddings.append(embedding.tolist())
            
        
        final_chunk_embeddings = [
            {**chunk_dict, "vector": text_chunks_embeddings[i]}
            for i, chunk_dict in enumerate(chunks_in_dict)
        ]
        
        # print(final_chunk_embeddings[0])
        # print(len(final_chunk_embeddings))
        # print(final_chunk_embeddings[len(final_chunk_embeddings)//2])
        
        all_embeddings.extend(final_chunk_embeddings) # accumulate embeddings for each file directory
        
        
    return all_embeddings
        

        
def get_PTwikiRAG_chunkingStrat_embeddings(
    model_name: str,
    chunking_type: Literal["fixed_size", "fixed_size_overlap", "variable_size", "mixed_chunking", None] = "fixed_size_overlap",
    dataset_jsonl_subset1_path: Optional[str] = None,
    dataset_jsonl_subset2_path: Optional[str] = None,
    chunk_size: Optional[int] = 200,
    overlap_factor: Optional[float] = 0.2,
    min_char_chunk: Optional[int] = 30,
    regex: Optional[str] = r"\n{3,}", # Default regex for variable_size // can also pass a specific token/tokens
    batch_size: Optional[int] = 32,
    max_tokens: Optional[int] = 128,
    sliding_window: bool = True,
    pooling: Optional[Literal["mean", "sum", "max", None]] = "mean",
    stride: Optional[int] = None,
    hf_token: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Generate embeddings for PT-wikiRAG documents using different chunking strategies.
    Note: It starts the embedding model locally on CPU/GPU/MPS. (So take caution for larger models)
    Args:
        model_name: Name of embedding model.
        chunking_type: Chunking strategy ("fixed_size", "fixed_size_overlap", "variable_size", "mixed_chunking").
        dataset_jsonl_subset1_path: Optional path to local dataset subset1.
        dataset_jsonl_subset2_path: Optional path to local dataset subset2.
        chunk_size: Chunk size for fixed-size chunking.
        min_char_chunk: Minimum characters for mixed chunking.
        regex: Regular expression for splitting with the "variable_size" chunking strategy
        batch_size: Encoding batch_size, for larger chunks it might be needed to lower the batch_size if memory is contrained
        sliding_window: Choose between applying sliding window on embeddings or default embeddings that do truncation when the embedding is too long
        stride: Optional stride value, the amount of tokens that are moved forward during sliding window -> Defaults to half of max_tokens if not declared
        hf_token: Optional[str] = None

    Returns:
        A list of dictionaries containing:
        {
            "doc_id": str,
            "doc_name": str,
            "text": str,
            "vector": List[float]
        }
    """
    # --- Dataset loading ---
    if dataset_jsonl_subset1_path and dataset_jsonl_subset2_path:
        dataset = load_jsonl_PT_wikiRAG_locally(dataset_jsonl_subset1_path,dataset_jsonl_subset2_path)
    else:
        dataset = loading_PT_wikiRAG_from_HF(hf_token)

    data = dataset["subsplit1"]
    
    
    # --- Chunking strategy mapping ---
    chunking_strategies: Dict[str, Callable[[str], List[str]]] = {
        "fixed_size": lambda text: fixed_size_chunking(text, chunk_size=chunk_size),
        "fixed_size_overlap": lambda text: fixed_size_chunking_with_overlap(text, chunk_size=chunk_size, overlap_factor=overlap_factor),
        "variable_size": lambda text: variable_size_chunking_character_based(
            text, characters=[regex], regex_mode=True
        ),
        "mixed_chunking": lambda text: special_mixed_chunking(text, min_char_chunk=min_char_chunk),
    }
    

    # Safe handling chunking stratery option
    if chunking_type not in chunking_strategies:
        raise ValueError(f"Invalid chunking_type: {chunking_type}")
    
    chunk_fn = chunking_strategies[chunking_type]


    # ---  Build List of Chunk Dictionaries ---
    list_chunks_dicts = [
        {
            "doc_id": doc["doc_id"],
            "doc_name": doc["doc_name"],
            "text":  chunk
         
        }
        for doc in data
        for chunk in chunk_fn(doc["content"])
    ]
    
    # A list with only the "text" part of the chgunks
    text_chunks = [chunk["text"] for chunk in list_chunks_dicts]
    

    # ---                      ---
    # ---      Load model      ---    
    # ---                      ---

    # Select device: MPS (Apple Silicon), CUDA (NVIDIA), util
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    
    model = SentenceTransformer(model_name_or_path=model_name, device=device, trust_remote_code=True, token=hf_token)

    # ---                            ---
    # ---    Generate embeddings     ---    
    # ---                            ---

    if sliding_window:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        embeddings = [  
            batched_sliding_window_chunk_embedding(
                text = text_chunk,
                model = model,
                tokenizer = tokenizer,
                max_tokens = max_tokens,
                batch_size = batch_size,
                stride = stride,
                pooling = pooling,
                hf_token=hf_token
            ).tolist() # tolist() because batched_sliding_window_chunk_embedding() returns a numpy.array
            
            for text_chunk in text_chunks
        ]
    else:
        embeddings = generate_embeddings_w_model(
            text=text_chunks, # can take all text_chunks list
            model=model,
            batch_size=batch_size,
            hf_token=hf_token
        )
        # embeddings = model.encode(
        #     sentences=text_chunks,
        #     batch_size=batch_size,
        #     convert_to_numpy=True
        # )

    
    final_chunks = [
        {**chunk_dict, "vector": embeddings[i]} for i, chunk_dict in enumerate(list_chunks_dicts)
    ]

    return final_chunks




def get_PTwikiRAG_DPCchunking_embeddings(
    files_dir: Optional[str] = None,
    model_name: str = "PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir",
    batch_size: int = 32,
    max_tokens: Optional[int] = 128,
    sliding_window: bool = True,
    pooling: Optional[Literal["mean", "sum", "max", None]] = "mean",
    stride: Optional[int] = None,
    hf_token: Optional[str] = None
):
    """
    Generate embeddings for PT-wikiRAG documents using files processed by the LX-Sentence-Spliter (PORTULAN)
    -- This means that each embedding with this strategy corresponds to a sentence (decided by the LX-Sentence-Spliter)
    Note: It starts the embedding model locally on CPU/GPU/MPS. (So take caution for larger models)
    Args:
        files_dir: Directory where the LXsentenceSplit files are located
        model_name: Name of embedding model.
        batch_size: Encoding batch_size, for larger chunks it might be needed to lower the batch_size if memory is contrained
        max_tokens: A parameter that decides the maximum number of tokens a chunk can take (for sliding window) it should respect the model's max_seq_length
        sliding_window: Choose between applying sliding window on embeddings or default embeddings that do truncation when the embedding is too long
        pooling: Choose a type of pooling for sliding window embeddings
        stride: Optional stride value, the amount of tokens that are moved forward during sliding window -> Defaults to half of max_tokens if not declared
        hf_token: Optional HuggingFace token if loading from HF.

    Returns:
        A list of dictionaries containing:
        {
            "doc_id": str,
            "doc_name": str,
            "text": str,
            "vector": List[float]
        }
    """
    dataset = loading_PT_wikiRAG_from_HF(token=hf_token)

    data = dataset["subsplit1"]

    doc_name_and_id_dict = {normalize_str(doc["doc_name"]): doc["doc_id"] for doc in data}

    l_json_names = [normalize_str(doc_name.replace(".txt", ".json")) for doc_name in doc_name_and_id_dict.keys()]
    l_ids = [id for id in doc_name_and_id_dict.items()]

    dict_json_files = {file_name:l_ids[i] for i,file_name in enumerate(l_json_names)}

    if files_dir is None:
        raise ValueError(
            "files_dir must be provided. Set it to the directory containing "
            "the DPC chunking output .json files."
        )

    if not os.path.exists(files_dir):
        raise ValueError(f"files_dir '{files_dir}' does not exist.")

    file_paths = glob.glob(files_dir + "/*")
    
    
    # LOADING MODEL AND TOKENIZER ONLY ONECE
    # device detection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    model = SentenceTransformer(model_name_or_path=model_name, device=device, token=hf_token)
    
    if sliding_window: 
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, token=hf_token)

    all_embeddings = [] # collect results across all files
    
    for file_path in file_paths:
        file = normalize_str(os.path.basename(file_path))
        if file in dict_json_files.keys():
            doc_name = file
            doc_id = dict_json_files[file]
        else:
            raise ValueError(f"Unexpected file_name {file}")
        
        if not doc_name:
            print("\n\n\n-------ERROR-------\n\n\n")
            print("file:", file)
            print("file_path:", file_path)
            
        with open(file_path, "r", encoding="UTF-8") as f:
            data = json.load(f)
            
            # Create a list with the chunks
            text_chunks_list = [chunk for chunk in data["chunks"]]
            
            # Can I close the file here already?
            f.close()
        
        if sliding_window:
            embeddings = [
                batched_sliding_window_chunk_embedding(
                    text=text_chunk,
                    model=model,
                    tokenizer=tokenizer,
                    max_tokens= max_tokens,
                    stride = stride,
                    pooling= pooling,
                    batch_size=batch_size,
                    hf_token=hf_token
                ).tolist() # tolist() because batched_sliding_window_chunk_embedding() returns a numpy.array
                for text_chunk in text_chunks_list
            ]
        else:
            embeddings = generate_embeddings_w_model(
                text=text_chunks_list,
                model=model,
                batch_size=batch_size,
                hf_token=hf_token
            )
        
        final_chunk_embeddings = [
            {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "text": text,
                "vector": embeddings[i]
            }
            for i,text in enumerate(text_chunks_list)
        ]
        all_embeddings.extend(final_chunk_embeddings)
    
    return all_embeddings

            
            

    
    



def generate_PT_wikiRAG_chunks_embeddings(
    model_name: str,
    chunking_type: Literal["fixed_size", "fixed_size_overlap", "variable_size", "mixed_chunking", None] = "fixed_size_overlap",
    dataset_jsonl_subset1_path: Optional[str] = None,
    dataset_jsonl_subset2_path: Optional[str] = None,
    hf_token: Optional[str] = None,
    chunk_size: Optional[int] = 200,
    overlap_factor: Optional[float] = 0.2,
    min_char_chunk: Optional[int] = 30,
    regex: Optional[str] = r"\n{3,}",
    batch_size: Optional[int] = 32
    
) -> List[Dict[str, Any]]:
    """
    Generate embeddings for PT-wikiRAG documents using different chunking strategies.
    Note: It starts the embedding model locally on CPU/GPU/MPS. (So take caution for larger models)
    Args:
        model_name: Name of embedding model.
        chunking_type: Chunking strategy ("fixed_size", "fixed_size_overlap", "variable_size", "mixed_chunking").
        dataset_jsonl_subset1_path: Optional path to local dataset subset1.
        dataset_jsonl_subset2_path: Optional path to local dataset subset2.
        hf_token: Optional HuggingFace token if loading from HF.
        chunk_size: Chunk size for fixed-size chunking.
        min_char_chunk: Minimum characters for mixed chunking.
        regex: Regular expression for splitting with the "variable_size" chunking strategy
        batch_size: Encoding batch_size, for larger chunks it might be needed to lower the batch_size if memory is contrained

    Returns:
        A list of dictionaries containing:
        {
            "doc_id": str,
            "doc_name": str,
            "text": str,
            "vector": List[float]
        }
    """

    # --- Dataset loading ---
    if dataset_jsonl_subset1_path and dataset_jsonl_subset2_path:
        dataset = load_jsonl_PT_wikiRAG_locally(dataset_jsonl_subset1_path,dataset_jsonl_subset2_path)
    else:
        dataset = loading_PT_wikiRAG_from_HF(hf_token)


    data = dataset["subsplit1"]


    # --- Chunking strategy mapping ---
    chunking_strategies: Dict[str, Callable[[str], List[str]]] = {
        "fixed_size": lambda text: fixed_size_chunking(text, chunk_size=chunk_size),
        "fixed_size_overlap": lambda text: fixed_size_chunking_with_overlap(text, chunk_size=chunk_size, overlap_factor=overlap_factor),
        "variable_size": lambda text: variable_size_chunking_character_based(
            text, characters=[regex], regex_mode=True
        ),
        "mixed_chunking": lambda text: special_mixed_chunking(text, min_char_chunk=min_char_chunk),
    }


    # Safe handling chunking stratery option
    if chunking_type not in chunking_strategies:
        raise ValueError(f"Invalid chunking_type: {chunking_type}")
    
    chunk_fn = chunking_strategies[chunking_type]

    # ---  Build List of Chunk Dictionaries ---
    list_chunks_dicts = [
        {
            "doc_id": doc["doc_id"],
            "doc_name": doc["doc_name"],
            "text":  chunk
         
        }
        for doc in data
        for chunk in chunk_fn(doc["content"])
    ]

    # A list with only the "text" part of the chgunks
    text_chunks = [chunk["text"] for chunk in list_chunks_dicts]

    # ---                      ---
    # ---      Load model      ---    
    # ---                      ---

    # Select device: MPS (Apple Silicon), CUDA (NVIDIA), util
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    
    model = SentenceTransformer(model_name_or_path=model_name, device=device, trust_remote_code=True)

    
    # ---                            ---
    # ---    Generate embeddings     ---    
    # ---                            ---

    embeddings = model.encode(
        sentences=text_chunks,
        batch_size=batch_size,
        convert_to_numpy=True
    )

    # Convert numpy array into standard python list
    embeddings = embeddings.tolist()


    
    final_chunks = [
        {**chunk_dict, "vector": embeddings[i]} for i, chunk_dict in enumerate(list_chunks_dicts)
    ]

    return final_chunks



def generate_embeddings_w_model(
    model_name: Optional[str] = None,
    model: Optional[SentenceTransformer] = None,
    text: Union[str, List[str]] = None,
    batch_size: int = 32,
    hf_token: Optional[str] = None
):
    """Generate embeddings for text using a SentenceTransformer model."""

    if model is None:
        if model_name is None:
            raise ValueError("You must provide either a `model` or `model_name`.")

        # Select device: MPS (Apple Silicon), CUDA (NVIDIA), or CPU
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        model = SentenceTransformer(model_name, device=device, trust_remote_code=True, token=hf_token)

    embeddings = model.encode(
        sentences=text,
        batch_size=batch_size,
        convert_to_numpy=True
    )

    return embeddings.tolist()

# Legacy*
def request_local_TEI_embeddings(inference_url: str, text: Union[str, List[str]]) -> List[List[float]]:
    """
    Sends a request to the local TEI server for the generation of embeddings from text

    Args:
        inference_url (str): the url endpoint for embedding generation (non OpenAPI)
        text (Union[str, List[List[str]]]): a single string or a list of strings

    Returns:
        Union[str, List[List[str]]]: a list of lists where each list is an embedding
    """

    # TODO: inference_url should really be just "server_url", then I should do inference_url = server_url + "/embed"

    payload= {
        "inputs": text,
        "truncate": True,
    }



    # Sending the request to the server
    try:
        response = requests.post(
            url=inference_url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=10 # This is a good practice as it prevents it from hanging forever
        )
        response.raise_for_status()

    except requests.exceptions.RequestException as e:
        logger.error("Request to %s failed: %s", inference_url, e)
        raise
    

    # Handling the response safely (ensuring the expected output)
    try:
        data = response.json()
    except ValueError:
        logger.error("Unexpected response format: %s", response.text)
        raise ValueError("Bad response format")
    
    # logger.info(data) --> Always returns a list of lists as expected
    return data


# Legacy*
def request_local_STserver_embeddings(inference_url:str, text: Union[str, List[str]], batch_size: int = 32) -> List[List[float]]:
    """
    Sends a request for embeddings of text to the local Sentence-Transformers server

    Args:
        inference_url (str): the url endpoint for embeddings of the local STserver
        text (Union[str, List[str]]): a string or a list of strings to be embedded
        batch_size (int, optional): batch size. Defaults to 32.

    Returns:
        List[List[float]]: Always returns a list with a list of floats
    """
    payload = {
        "text": text,
        "batch_size": batch_size
    }



    # Sendind the request to the local Sentence Transformers server
    try:
        response = requests.post(
            url=inference_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error("Request to %s failed: %s", inference_url, e)
        raise

    
    # Verifying data format 
    try:
        data = response.json()["embeddings"]
    except ValueError:
        logger.error("Unexpected respose format %s.", response.text)
        raise ValueError("Bad response format")
    

    return data # returns a list of lists


# Legacy*
"""
This should go on another module like server_info.py
"""
def local_STserver_info(server_url:str) -> dict:
    # server url should be http://localhost:8000
    try:    
        url = server_url + "/info"
        response = requests.get(
            url=url
        )
    except requests.exceptions.RequestException as e:
        logger.error("Request to %s failed: %s", url, e)
        raise

    logger.info(response)
    logger.info(response.json())
    return response.json()

# Legacy*
"""
This should go on another module like server_info.py
"""
def local_TEI_server_info(server_url:str) -> dict:
    # server url should be http://localhost:8089
    try:    
        url = server_url + "/info"
        response = requests.get(
            url=url
        )
    except requests.exceptions.RequestException as e:
        logger.error("Request to %s failed: %s", url, e)
        raise

    logger.info(response)
    logger.info(response.json())
    logger.info("\n\nModel name: %s", response.json()["model_id"])
    return response.json()






# Legacy*
def generate_TEI_PT_wikiRAG_embeddings(
    inference_url: str,
    chunking_type: Literal["fixed_size", "fixed_size_overlap", "variable_size", "mixed_chunking", None] = "fixed_size_overlap",
    dataset_jsonl_subset1_path: Optional[str] = None,
    dataset_jsonl_subset2_path: Optional[str] = None,
    hf_token: Optional[str] = None,
    chunk_size: int = 200,
    min_char_chunk: int = 30,
    batch_size: int = 8
) -> List[Dict[str, Any]]:
    """
    [Using an inference_url of TEI - Transformers Embedding Inference]
    Generate embeddings for PT-wikiRAG documents using different chunking strategies.

    Args:
        model_name: Name of embedding model.
        chunking_type: Chunking strategy ("fixed_size", "fixed_size_overlap", "variable_size", "mixed_chunking").
        dataset_jsonl_subset1_path: Optional path to local dataset subset1.
        dataset_jsonl_subset2_path: Optional path to local dataset subset2.
        hf_token: Optional HuggingFace token if loading from HF.
        chunk_size: Chunk size for fixed-size chunking.
        min_char_chunk: Minimum characters for mixed chunking.

    Returns:
        A list of dictionaries containing:
        {
            "doc_id": str,
            "doc_name": str,
            "text": str,
            "vector": List[float]
        }
    """

    # --- Dataset loading ---
    if dataset_jsonl_subset1_path and dataset_jsonl_subset2_path:
        dataset = load_jsonl_PT_wikiRAG_locally(dataset_jsonl_subset1_path,dataset_jsonl_subset2_path)
    else:
        dataset = loading_PT_wikiRAG_from_HF(hf_token)

    # Storing the documents part of the dataset in a variable
    data = dataset["subsplit1"]


    # --- Chunking strategy mapping ---
    chunking_strategies: Dict[str, Callable[[str], List[str]]] = {
        "fixed_size": lambda text: fixed_size_chunking(text, chunk_size=chunk_size),
        "fixed_size_overlap": lambda text: fixed_size_chunking_with_overlap(text, chunk_size=chunk_size),
        "variable_size": lambda text: variable_size_chunking_character_based(
            text, characters=[r"\n{3,}"], regex_mode=True
        ),
        "mixed_chunking": lambda text: special_mixed_chunking(text, min_char_chunk=min_char_chunk),
    }

    chunk_fn = chunking_strategies[chunking_type]

    # ---  Build List of Chunk Dictionaries ---
    list_chunks_dicts = [
        {
            "doc_id": doc["doc_id"],
            "doc_name": doc["doc_name"],
            "text":  chunk
         
        }
        for doc in data
        for chunk in chunk_fn(doc["content"])
    ]

    # A list with only the "text" part of the chunks
    text_chunks = [chunk["text"] for chunk in list_chunks_dicts]
    
    # ---                               ---
    # --- Generate with local TEI serer ---    
    # ---                               ---

    # Needs to be batched because the endpoint only supports max_batch_size=32
    # Although I think this can be changed in the declaration of the server
    # I want to keep it low, not to overburden my computer 8-32
    batched_text_chunks = [
        text_chunks[i:i+batch_size]
        for i in range(0, len(text_chunks), batch_size)
    ]

    all_embeddings = []

    for batch in tqdm.tqdm(batched_text_chunks, desc="Generating embeddings"):
        batch_embeddings = request_local_TEI_embeddings(
            url=inference_url,
            text=batch
        )
        all_embeddings.extend(batch_embeddings)

    # --- Generate with local TEI serer ---

    final_chunks = [
        {**chunk_dict, "vector": all_embeddings[i]} for i, chunk_dict in enumerate(list_chunks_dicts)
    ]

    return final_chunks





# Legacy*
def generate_local_STserver_PT_wikiRAG_embeddings(
    inference_url: str,
    chunking_type: Literal["fixed_size", "fixed_size_overlap", "variable_size", "mixed_chunking", None] = "fixed_size_overlap",
    dataset_jsonl_subset1_path: Optional[str] = None,
    dataset_jsonl_subset2_path: Optional[str] = None,
    hf_token: Optional[str] = None,
    chunk_size: int = 200,
    min_char_chunk: int = 30,
    batch_size: int = 32
) -> List[Dict[str, Any]]:
    """
    [Using an inference_url of TEI - Transformers Embedding Inference]
    Generate embeddings for PT-wikiRAG documents using different chunking strategies.

    Args:
        model_name: Name of embedding model.
        chunking_type: Chunking strategy ("fixed_size", "fixed_size_overlap", "variable_size", "mixed_chunking").
        dataset_jsonl_subset1_path: Optional path to local dataset subset1.
        dataset_jsonl_subset2_path: Optional path to local dataset subset2.
        hf_token: Optional HuggingFace token if loading from HF.
        chunk_size: Chunk size for fixed-size chunking.
        min_char_chunk: Minimum characters for mixed chunking.

    Returns:
        A list of dictionaries containing:
        {
            "doc_id": str,
            "doc_name": str,
            "text": str,
            "vector": List[float]
        }
    """

    # --- Dataset loading ---
    if dataset_jsonl_subset1_path and dataset_jsonl_subset2_path:
        dataset = load_jsonl_PT_wikiRAG_locally(dataset_jsonl_subset1_path,dataset_jsonl_subset2_path)
    else:
        dataset = loading_PT_wikiRAG_from_HF(hf_token)


    data = dataset["subsplit1"]


    # --- Chunking strategy mapping ---
    chunking_strategies: Dict[str, Callable[[str], List[str]]] = {
        "fixed_size": lambda text: fixed_size_chunking(text, chunk_size=chunk_size),
        "fixed_size_overlap": lambda text: fixed_size_chunking_with_overlap(text, chunk_size=chunk_size),
        "variable_size": lambda text: variable_size_chunking_character_based(
            text, characters=[r"\n{3,}"], regex_mode=True
        ),
        "mixed_chunking": lambda text: special_mixed_chunking(text, min_char_chunk=min_char_chunk),
    }


    # Safe handling chunking stratery option
    if chunking_type not in chunking_strategies:
        raise ValueError(f"Invalid chunking_type: {chunking_type}")
    
    chunk_fn = chunking_strategies[chunking_type]

    # ---  Build List of Chunk Dictionaries ---
    list_chunks_dicts = [
        {
            "doc_id": doc["doc_id"],
            "doc_name": doc["doc_name"],
            "text":  chunk
         
        }
        for doc in data
        for chunk in chunk_fn(doc["content"])
    ]

    # A list with only the "text" part of the chgunks
    text_chunks = [chunk["text"] for chunk in list_chunks_dicts]
    
    # ---                               ---
    # --- Generate with local STserver ---    
    # ---                               ---

    # Needs to be batched because the endpoint only supports max_batch_size=32
    # Although I think this can be changed in the declaration of the server
    # I want to keep it low, not to overburden my computer 8-32
    embeddings = request_local_STserver_embeddings(
        inference_url=inference_url,
        text=text_chunks,
        batch_size = batch_size
    )

    final_chunks = [
        {**chunk_dict, "vector": embeddings[i]} for i, chunk_dict in enumerate(list_chunks_dicts)
    ]

    return final_chunks


