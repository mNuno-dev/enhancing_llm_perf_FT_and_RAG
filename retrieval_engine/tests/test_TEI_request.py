import pytest
from embeddings.embeddings import request_local_TEI_embeddings, local_STserver_info, local_TEI_server_info



def test_TEI_single_embedding_request():
    inference_url = "http://localhost:8089/embed"
    text = "Boas bro, gostas de chocolate?"
    result = request_local_TEI_embeddings(inference_url=inference_url, text=text)

    assert type(result[0]) == list


def test_TEI_multiple_embeddings_request():
    inference_url = "http://localhost:8089/embed"
    text = ["Boas bro, gostas de chocolate?", "Eu sou o Cristiano Ronaldz"]
    result = request_local_TEI_embeddings(inference_url=inference_url, text=text)
    assert len(result) == len(text)


def test_TEI_info():
    server_url = "http://localhost:8089"
    response = local_TEI_server_info(server_url)
    assert type(response) == dict
