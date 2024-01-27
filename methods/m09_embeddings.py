import logging

from embeddings_search import find_near_pair

logger = logging.getLogger("SoCloverAI")


def generate(temperature, pair):
    return find_near_pair(pair)
