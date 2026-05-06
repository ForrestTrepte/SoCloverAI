from wordfreq import top_n_list, zipf_frequency


max_words = 50_000


def get_words() -> list[str]:
    """Get a list of the most common words in English, sorted by frequency."""
    result = top_n_list("en", max_words, ascii_only=True)
    return result
