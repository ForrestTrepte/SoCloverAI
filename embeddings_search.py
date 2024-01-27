import logging
from typing import Dict, List, Tuple

from embeddings import get_embeddings
from english_words import get_common_word_embeddings, remove_word_forms_of

logger = logging.getLogger("SoCloverAI")


class CandidatesByDistance:
    def __init__(self) -> None:
        self.candidates: Dict[str, float] = {}

    def add(self, candidate: str, distance: float) -> None:
        # Get current distance or 2.0 if not in set (2.0 is larger than any cos distance)
        current_distance = self.candidates.get(candidate, 2.0)
        if distance < current_distance:
            self.candidates[candidate] = distance

    def get_sorted_candidates(self) -> List[Tuple[str, float]]:
        result = sorted(self.candidates.items(), key=lambda x: x[1])
        return result


def find_near_pair(word0: str, word1: str) -> List[str]:
    pair_documents: List[str] = [
        f"{word0}",
        f"{word1}",
        f"{word0} {word1}",
        f"{word1} {word0}",
    ]
    pair_embeddings = get_embeddings(pair_documents)

    common_words_count = 60000
    common_words_embeddings = get_common_word_embeddings(common_words_count)

    all_candidates = CandidatesByDistance()

    search_index_sets = [
        [0, 1],  # Find near both individual words
        [2],  # Find near pair of words in original order
        [3],  # Find near pair of words in reverse order
    ]

    for search_index_set in search_index_sets:
        words = [pair_documents[i] for i in search_index_set]
        logger.info(f"        Find near: {' & '.join(words)}")

        embeddings = [pair_embeddings[i] for i in search_index_set]
        candidates_and_distances = common_words_embeddings.find_near(embeddings, 20)
        candidate_names = [candidate for candidate, _ in candidates_and_distances]
        logger.info(f"          {', '.join(candidate_names)}")
        for candidate, distance in candidates_and_distances:
            all_candidates.add(candidate, distance)

    sorted_candidates = all_candidates.get_sorted_candidates()
    sorted_candidate_words = [candidate for candidate, _ in sorted_candidates]

    valid_candidates = sorted_candidate_words
    valid_candidates = remove_word_forms_of(word0, valid_candidates)
    valid_candidates = remove_word_forms_of(word1, valid_candidates)

    return valid_candidates
