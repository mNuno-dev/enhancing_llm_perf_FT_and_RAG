import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer, util
import json

# TODO: Write the README.txt with instructions on:
#              - how to start the server
#              - models used
#              - quicks
#              - API call examples
#              - etc ...
#         

"""
 ------> Commands <------

 Run server:
 FastAPI + Uvicorn expects the format:
    $ uvicorn <filename>:<fastapi_app_instance>

SO:
    $ python3 -m uvicorn STserver:app --reload --host 0.0.0.0 --port 8000

"""


####                    ####
####  Global Variables  ####
####                    ####

MODEL_NAME = "BAAI/bge-m3"



# Select device: MPS (Apple Silicon), CUDA (NVIDIA), util
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Load model onto device
model = SentenceTransformer(model_name_or_path=MODEL_NAME, device=device)

app = FastAPI(title="Sentence Transformers API")

class TextInput(BaseModel):
    text: Union[str, List[str]]
    batch_size: Optional[int] = 32

class SimilatiryInput(BaseModel):
    text1: str
    text2: str

@app.get("/")
def home():
    return {"message": f"Running on {device.upper()}!"}

@app.get("/info")
def model_info():
    return {
        "model_name": MODEL_NAME,
        "device": device,
        "embedding_dim": model.get_sentence_embedding_dimension()
    }

@app.post("/embed")
def embed(input: TextInput):
    embeddings = model.encode(
        input.text,
        batch_size=input.batch_size,
        convert_to_numpy=True
    )

    if isinstance(input.text, str):
        # print("before: ", embeddings)
        # print("mid: ", list(embeddings.tolist()))
        l = []
        l.append(embeddings.tolist())
        # print("after: ", l)
        return {"embeddings": l} # return a list with single embedding
    else:
        return {"embeddings": embeddings.tolist()} # return a list of embeddings



    # # If input is a single string or a list with only one string, make sure to return a single vector
    # if isinstance(input.text, str):
    #     # print("before: ", embeddings)
    #     # print("after: ", embeddings.tolist())
    #     return {"embeddings": embeddings.tolist()} # return a single embedding
    # elif isinstance(input.text, list) and len(input.text) == 1:
    #     return {"embeddings": embeddings.tolist()[0]} # return the single embedding inside the list
    # else:
    #     return {"embeddings": embeddings.tolist()} # return a list of embeddings


@app.post("/similarity")
def similarity(input: SimilatiryInput):
    emb1 = model.encode(input.text1, convert_to_tensor=True)
    emb2 = model.encode(input.text2, convert_to_tensor=True)
    score = util.cos_sim(emb1,emb2).item()

    return {"similarity": score}