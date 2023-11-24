import random

from board import get_random_board
from words import Words


def test_get_random_board():
    random.seed("Clover")
    board = get_random_board()
    assert len(board.pairs) == 4
    words = set()
    for pair in board.pairs:
        words.add(pair[0])
        words.add(pair[1])
    assert len(words) == 8
    for word in Words:
        assert word in Words
