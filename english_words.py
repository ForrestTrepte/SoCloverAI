import json
import logging
import os

import numpy as np
from gensim.models import KeyedVectors, Word2Vec
from word_forms.word_forms import get_word_forms

from embeddings import WordEmbeddings, get_embeddings

maximum_common_words = 200000
maximum_common_word_embeddings = 60000
logger = logging.getLogger("SoCloverAI")
project_root = os.path.dirname(os.path.realpath(__file__))


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
    model_filename = (
        f"{project_root}/downloaded_data_files/GoogleNews-vectors-negative300.bin"
    )
    logger.info(f"Loading {model_filename}")
    model = KeyedVectors.load_word2vec_format(model_filename, binary=True)

    word_frequencies = []
    invalid_words = []
    for word in model.index_to_key:
        if len(word_frequencies) >= maximum_common_words:
            break
        if is_valid_word(word):
            word_frequencies.append((word, model.get_vecattr(word, "count")))
        else:
            invalid_words.append(word)

    logger.info(f"Loaded {len(word_frequencies)} valid words")
    logger.info(f"Filtered out {len(invalid_words)} invalid words")
    logger.debug(
        (", ").join(
            [
                word
                for word in invalid_words
                if not contains_problematic_characters(word)
            ]
        )
    )

    # The model is already sorted, but we'll sort again just in case the model changes in the future
    sorted_word_frequencies = sorted(word_frequencies, key=lambda x: x[1], reverse=True)

    sorted_words = [word for word, _ in sorted_word_frequencies]
    with open(word_frequency_filepath, "w") as f:
        json.dump(sorted_words, f, indent=2)
    return sorted_words


def is_valid_word(word):
    if contains_problematic_characters(word):
        return False
    if not word.isalpha():
        if len(word) == 1:
            return True  # accept a single symbol (e.g. $)
        return False
    return True


def contains_problematic_characters(input_string, encoding="cp1252"):
    try:
        input_string.encode(encoding)
        return False
    except UnicodeEncodeError as e:
        return True


def get_common_word_embeddings(count):
    assert count <= maximum_common_word_embeddings
    common_words = get_common_words(maximum_common_word_embeddings)
    word_embeddings_filepath = f"{project_root}/words_by_frequency_embeddings.npz"
    is_loaded = False
    if os.path.exists(word_embeddings_filepath):
        embeddings_data = np.load(word_embeddings_filepath)
        embeddings_stacked = embeddings_data["embeddings"]
        if len(embeddings_stacked) >= maximum_common_word_embeddings:
            is_loaded = True
        else:
            logger.info(
                f"Setting maximum_common_word_embeddings has increased. Regenerating {word_embeddings_filepath}"
            )

    if not is_loaded:
        logger.info(f"Getting embeddings")
        embeddings = get_embeddings(common_words)
        assert len(common_words) == len(embeddings)
        embeddings_stacked = np.stack(embeddings)

        logger.info(f"Saving {word_embeddings_filepath}")
        np.savez(word_embeddings_filepath, embeddings=embeddings_stacked)
        logger.info(f"Saved {word_embeddings_filepath}")

    result = WordEmbeddings(common_words[:count], embeddings_stacked[:count])
    return result


def remove_word_forms_of(base_word, words_to_remove_from):
    def is_any_word_form_in(word_forms, word):
        for word_form in word_forms:
            if word_form.lower() in word.lower():
                return True
        return False

    word_forms = get_word_forms_list(base_word)
    result = []
    for candidate_word in words_to_remove_from:
        # remove words that are a form of the base word
        if is_any_word_form_in(word_forms, candidate_word):
            continue
        # also remove words that have a form in the base word
        candidate_word_forms = get_word_forms_list(candidate_word)
        if is_any_word_form_in(candidate_word_forms, base_word):
            continue
        result.append(candidate_word)

    return result


def get_word_forms_list(base_word):
    word_forms_by_type = get_word_forms(base_word)
    word_forms = set.union(*word_forms_by_type.values())
    return word_forms
