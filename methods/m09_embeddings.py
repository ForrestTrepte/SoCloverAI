import json
import logging
import os

from gensim.models import KeyedVectors, Word2Vec
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings

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


maximum_common_words = 200000


def get_common_words(count):
    assert count <= maximum_common_words
    sorted_words = get_words_sorted_by_frequency()
    return sorted_words[:count]


def get_words_sorted_by_frequency():
    # Load a list of the most common words in English
    word_frequency_filepath = f"{project_root}/words_by_frequency.json"
    if os.path.exists(word_frequency_filepath):
        with open(word_frequency_filepath, "r") as f:
            sorted_words = json.load(f)
            # sanity check, if this changes we should delete words_by_frequency.json so it can be regenerated
            if len(sorted_words) >= maximum_common_words:
                return sorted_words
            logger.info(
                f"Setting maximum_common_words has increased. Regenerating {word_frequency_filepath}"
            )
    else:
        logger.info(f"Generating {word_frequency_filepath}")

    # If we don't have the list of the most common words in English, load a Google's pre-trained Word2Vec model
    # Note: Download the model if you don't have it; it's quite large (over 1.5GB)
    # See https://code.google.com/archive/p/word2vec/, Pre-trained word and phrase vectors for a link to download the file from Google Drive
    model = KeyedVectors.load_word2vec_format(
        f"{project_root}/downloaded_data_files/GoogleNews-vectors-negative300.bin",
        binary=True,
    )

    word_frequencies = []
    invalid_words = []
    for word in model.index_to_key:
        if len(word_frequencies) >= maximum_common_words:
            break
        if is_valid_word(word):
            word_frequencies.append((word, model.get_vecattr(word, "count")))
        else:
            invalid_words.append((word, model.get_vecattr(word, "count")))

    logger.info(f"Loaded {len(word_frequencies)} valid words")
    logger.info(f"Checking {len(invalid_words)} invalid words")
    possible_candidates = []
    for word, frequency in invalid_words:
        non_alpha_chars = {char for char in word if not char.isalpha()}
        invalid_chars = non_alpha_chars - {"-", "_", "."}
        if len(invalid_chars) > 0:
            continue
        possible_candidate = word.replace("_", "-").replace(".", "")
        possible_candidates.append((possible_candidate, frequency))
    logger.info(f"Checking {len(possible_candidates)} fixed-up candidates")
    logger.info((", ").join([word for word, _ in possible_candidates]))

    # The model is already sorted, but we'll sort again just in case the model changes in the future
    sorted_word_frequencies = sorted(word_frequencies, key=lambda x: x[1], reverse=True)

    sorted_words = [word for word, _ in sorted_word_frequencies]
    with open(word_frequency_filepath, "w") as f:
        json.dump(sorted_words, f, indent=2)
    return sorted_words


def is_valid_word(word):
    if len(word) == "$":
        return True
    if not word.isalpha():
        return False
    return True
