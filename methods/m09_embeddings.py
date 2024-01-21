import logging

from english_words import get_common_words, get_common_word_embeddings
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

    common_words_count = 60000
    common_words = get_common_words(common_words_count)
    common_words_embeddings = get_common_word_embeddings(common_words_count)

    candidates = common_words_embeddings.find_near(documents[0], 10)
    # TODO: Search for words with embeddings related to the word pair and use those as clues

    return ["boxing"]
