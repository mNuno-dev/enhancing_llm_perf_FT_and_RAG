from embeddings import generate_PT_wikiRAG_chunks_embeddings


fixed_size_embeddz = generate_PT_wikiRAG_chunks_embeddings(
    model_name="BAAI/bge-m3",
    chunking_type="fixed_size",
    chunk_size=200,
    overlap_factor=0.2,
    min_char_chunk=60,
    regex=r"\n{3,}",
    batch_size=32,
)

