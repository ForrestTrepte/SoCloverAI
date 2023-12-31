import random

from words import Words


class Board:
    def __init__(self, pairs):
        self.pairs = pairs


def get_random_pair():
    words = random.sample(Words, 2)
    pair = (words[0], words[1])
    return pair


def get_random_board():
    words = random.sample(Words, 8)
    pairs = [(words[i], words[i + 1]) for i in range(0, 8, 2)]
    return Board(pairs)
