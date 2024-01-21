import logging

from english_words import get_common_words, get_common_word_embeddings
from embeddings import get_embeddings

logger = logging.getLogger("SoCloverAI")


class CandidatesByDistance:
    def __init__(self):
        self.candidates = {}

    def add(self, candidate, distance):
        # Get current distance or 2.0 if not in set (2.0 is larger than any cos distance)
        current_distance = self.candidates.get(candidate, 2.0)
        if distance < current_distance:
            self.candidates[candidate] = distance

    def get_sorted_candidates(self):
        result = sorted(self.candidates.items(), key=lambda x: x[1])
        return result


def generate(temperature, pair):
    individual_documents = [
        f"{pair[0]}",
        f"{pair[1]}",
    ]
    individual_embeddings = get_embeddings(individual_documents)

    together_documents = [
        f"{pair[0]} {pair[1]}",
        f"{pair[1]} {pair[0]}",
    ]
    together_embeddings = get_embeddings(together_documents)

    common_words_count = 60000
    common_words = get_common_words(common_words_count)
    common_words_embeddings = get_common_word_embeddings(common_words_count)

    all_candidates = CandidatesByDistance()

    # Find using together embeddings
    for i in range(2):
        logger.info(f"        Find near: {together_documents[i]}")
        candidates_and_distances = common_words_embeddings.find_near(
            together_embeddings[i], 20
        )
        candidate_names = [candidate for candidate, _ in candidates_and_distances]
        logger.info(f"          {', '.join(candidate_names)}")
        for candidate, distance in candidates_and_distances:
            all_candidates.add(candidate, distance)

    sorted_candidates = all_candidates.get_sorted_candidates()
    result = [candidate for candidate, _ in sorted_candidates]
    return result
