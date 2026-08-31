from embeddings import get_PTwikiRAG_DPCchunking_embeddings, get_PT_wikiRAG_LXsentenceSplit_embeddings

TOKEN_HF = os.getenv("HF_TOKEN", "")  # set HF_TOKEN as an environment variable # I will remove this later

res = get_PTwikiRAG_DPCchunking_embeddings(batch_size=32, sliding_window=True, stride=32, pooling="mean", hf_token=TOKEN_HF)

print(res[12])

# res = get_PTwikiRAG_DPCchunking_embeddings(batch_size=32, sliding_window=False, hf_token=TOKEN_HF)

# print(res[12])


