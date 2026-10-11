from typing import Callable

from GenerateKeywordCards2.embeddings import WordEmbeddings


def get_distance_threshold_association_predictor(
    embeddings: WordEmbeddings, distance_threshold: float
) -> Callable[[str, str], bool]:
    def predictor(word: str, associated_word: str) -> bool:
        return embeddings.distance(word, associated_word) <= distance_threshold

    return predictor
