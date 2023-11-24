import random
from board import get_random_pair
from m01_direct import m01_basic

trials = 10
methods = [m01_basic]


def generate_clues(trials, methods):
    for method in methods:
        print(f"Testing {method.__name__}")
        for trial in range(trials):
            print(f"  Trial {trial + 1} of {trials}")
            pair = get_random_pair()
            clue = method(pair)
            assert isinstance(clue, str)
            print(f"    {pair[0]} {pair[1]} -> {clue}")


if __name__ == "__main__":
    random.seed("Clover")
    generate_clues(trials, methods)
