import logging
import os

import numpy as np
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings

from init_openai import init_openai

logger = logging.getLogger("SoCloverAI")
init_openai()
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
        self.words = np.array(words)
        self.embeddings = embeddings

    def find_near(self, target_embeddings, count):
        # TODO: Can this replace find_near by accepting a single embedding in the list?
        distances_by_target = []
        for target_embedding in target_embeddings:
            target_embedding_column_vector = np.array([target_embedding]).T

            # Use matrix multiplication to calculate the dot product with all the word embeddings
            dot_products = self.embeddings @ target_embedding_column_vector
            # cos distance = 1 - dot product (if embeddings are normalized, which they are)
            distances = 1 - dot_products

            distances_by_target.append(distances)

        # Get max of distances for each word
        distances = np.max(distances_by_target, axis=0)

        # Get indices from ascending sort of the distances, flattened to 1d, slice to count elements
        nearest_indices = np.argsort(distances, axis=0).flatten()[:count]

        # Get the words and distances for the nearest indices
        nearest_words = self.words[nearest_indices]
        nearest_distances = distances[nearest_indices].flatten()

        result = list(zip(nearest_words, nearest_distances))
        return result


def is_normalized(embedding):
    norm = np.linalg.norm(embedding)
    is_normalized = np.isclose(norm, 1.0)
    return is_normalized
