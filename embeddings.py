import logging
import os

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
    return embeddings
