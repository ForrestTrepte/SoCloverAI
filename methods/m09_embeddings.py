import logging
import os

from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from english_words import get_common_words

logger = logging.getLogger("SoCloverAI")
project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def generate(temperature, pair):
    embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    embeddings_store = LocalFileStore(f"{project_root}/embeddings-cache")
    embedder = CacheBackedEmbeddings.from_bytes_store(
        embeddings_model, embeddings_store, namespace=embeddings_model.model
    )

    documents = [
        f"{pair[0]}",
        f"{pair[1]}",
        f"{pair[0]} {pair[1]}",
        f"{pair[1]} {pair[0]}",
    ]
    embeddings = embedder.embed_documents(documents)

    common_words = get_common_words(50000)

    # TODO: Search for words with embeddings related to the word pair and use those as clues

    return ["boxing"]
