from embeddings import get_PT_wikiRAG_LXsentenceSplit_embeddings, get_PTwikiRAG_docSegmentation_embeddings, get_PTwikiRAG_simple_line_chunking_embeddings, get_PTwikiRAG_chunkingStrat_embeddings
import multiprocessing as mp


TOKEN = "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"
# get_PT_wikiRAG_LXsentenceSplit_embeddings()
# get_PT_wikiRAG_simple_line_chunking_embeddings()
# test = get_PTwikiRAG_docSegmentation_embeddings(model="PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir", batch_size=32)


# print(test[0])

# print(test[2])


# get_PT_wikiRAG_LXsentenceSplit_embeddings()

# print("Multiprocessing start method", mp.get_start_method())


# t = "As condições do tempo levaram à intensificação, que se tornou um furacão no início de 25 de setembro de 2024. Uma intensificação mais pronunciada e rápida ocorreu quando Helene atravessou o Golfo do México no dia seguinte, atingindo a intensidade de categoria 4 na tarde de 26 de setembro. No final de 26 de setembro de 2024, Helene atingiu a costa com intensidade máxima na região de Big Bend, na Flórida, perto da cidade de Perry, com ventos máximos sustentados de 220 quilômetros por hora. Helene enfraqueceu à medida que se movia rapidamente para o interior antes de degenerar para um ciclone pós-tropical sobre o Tennessee em 27 de setembro. A tempestade então parou sobre o estado antes de se dissipar em 29 de setembro de 2024."
# # t = "As condições do tempo levaram à intensificação, que se tornou um furacão no início de 25 de setembro de 2024. Uma intensificação mais pronunciada e rápida ocorreu quando Helene atravessou o Golfo do México no dia seguinte, atingindo a intensidade de categoria 4 na tarde de 26 de setembro. No final de 26 de setembro de 2024, Helene atingiu a costa com intensidade máxima na região de Big Bend, na Flórida, perto da cidade de Perry, com ventos máximos sustentados de 220 quilômetros por hora. Helene enfraqueceu à medida que se movia rapidamente para o interior antes de degenerar para um ciclone pós-tropical sobre o Tennessee em 27 de setembro. A tempestade então parou sobre o estado antes de se dissipar em 29 de setembro de 2024.As condições do tempo levaram à intensificação, que se tornou um furacão no início de 25 de setembro de 2024. Uma intensificação mais pronunciada e rápida ocorreu quando Helene atravessou o Golfo do México no dia seguinte, atingindo a intensidade de categoria 4 na tarde de 26 de setembro. No final de 26 de setembro de 2024, Helene atingiu a costa com intensidade máxima na região de Big Bend, na Flórida, perto da cidade de Perry, com ventos máximos sustentados de 220 quilômetros por hora. Helene enfraqueceu à medida que se movia rapidamente para o interior antes de degenerar para um ciclone pós-tropical sobre o Tennessee em 27 de setembro. A tempestade então parou sobre o estado antes de se dissipar em 29 de setembro de 2024."
# # t = "chocolate"



# t_sliding_window_embeddings = generate_sliding_window_embeddings(text=t, model_name="PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir", max_tokens=128)
# # generate_sliding_window_embeddings(text="asjhdbaskjdbkasbdasd", model_name="PORTULAN/serafim-900m-portuguese-pt-sentence-encoder")


# # print(t_sliding_window_embeddings)
# print(len(t_sliding_window_embeddings))


# res = N_PTwikiRAG_docSegmentation_embeddings(model_name="PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir")


# res = get_PTwikiRAG_simple_line_chunking_embeddings(model_name="PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir",batch_size=32,sliding_window=False)

# print(res[14])

# one = get_PTwikiRAG_chunkingStrat_embeddings(model_name="PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir", batch_size=32, chunking_type="fixed_size", chunk_size=300, hf_token=TOKEN, sliding_window=True)

# two = get_PTwikiRAG_chunkingStrat_embeddings(model_name="PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir", batch_size=32, chunking_type="fixed_size", chunk_size=300, hf_token=TOKEN, sliding_window=False)

# print("one->text",one[2]["text"])
# print("one->embedding(last three)",one[2]["vector"][-3:])

# print("two->text",two[2]["text"])
# print("two->embedding(last three)",two[2]["vector"][-3:])



res = get_PT_wikiRAG_LXsentenceSplit_embeddings(model_name="PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir")

print(res[14])