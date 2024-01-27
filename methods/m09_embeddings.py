from typing import List, Tuple

from embeddings_search import find_near_pair


def generate(temperature: float, pair: Tuple[str, str]) -> List[str]:
    return find_near_pair(pair[0], pair[1])
