import random
from typing import List, Tuple

from clover_words import CloverWords


class Board:
    def __init__(self, pairs: List[Tuple[str, str]]) -> None:
        self.pairs = pairs


def get_random_pair() -> Tuple[str, str]:
    words = random.sample(CloverWords, 2)
    pair = (words[0], words[1])
    return pair


def get_random_board() -> Board:
    words = random.sample(CloverWords, 8)
    pairs = [(words[i], words[i + 1]) for i in range(0, 8, 2)]
    return Board(pairs)
