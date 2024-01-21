import logging

from english_words import get_common_words
from embeddings import get_embeddings

logger = logging.getLogger("SoCloverAI")


def generate(temperature, pair):
    documents = [
        f"{pair[0]}",
        f"{pair[1]}",
        f"{pair[0]} {pair[1]}",
        f"{pair[1]} {pair[0]}",
    ]
    pair_embeddings = get_embeddings(documents)

    common_words = get_common_words(60000)
    common_words_embeddings = get_embeddings(common_words)

    # TODO: Search for words with embeddings related to the word pair and use those as clues

    return ["boxing"]
