import logging
import os
import numpy as np

from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings

project_root = os.path.dirname(os.path.realpath(__file__))
embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
embeddings_store = LocalFileStore(f"{project_root}/embeddings-cache")
embedder = CacheBackedEmbeddings.from_bytes_store(
    embeddings_model, embeddings_store, namespace=embeddings_model.model
)


def get_embeddings(documents):
    embeddings = embedder.embed_documents(documents)
    for embedding in embeddings:
        assert is_normalized(embedding)
    return embeddings


class WordEmbeddings:
    def __init__(self, words, embeddings):
        assert len(words) == len(embeddings)
        self.words = words
        self.embeddings = embeddings

    def find_near(self, word, count):
        # TODO: Implement this
        return self.words[:count]


def is_normalized(embedding):
    norm = np.linalg.norm(embedding)
    is_normalized = np.isclose(norm, 1.0)
    return is_normalized
