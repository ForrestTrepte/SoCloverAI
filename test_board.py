import random

from board import get_random_pair, get_random_board
from clover_words import CloverWords


def test_get_random_pair():
    random.seed("Clover")
    pair = get_random_pair()
    assert pair[0] != pair[1]
    assert pair[0] in CloverWords
    assert pair[1] in CloverWords


def test_get_random_board():
    random.seed("Clover")
    board = get_random_board()
    assert len(board.pairs) == 4
    words = set()
    for pair in board.pairs:
        words.add(pair[0])
        words.add(pair[1])
    assert len(words) == 8
    for word in CloverWords:
        assert word in CloverWords
