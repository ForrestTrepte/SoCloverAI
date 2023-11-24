import random
from board import get_random_board
from m01_direct import m01_direct

trials = 10
methods = [m01_direct]

if __name__ == "__main__":
    random.seed("Clover")
    for method in methods:
        print(f"Testing {method.__name__}")
        for trial in range(trials):
            print(f"  Trial {trial + 1} of {trials}")
            board = get_random_board()
            clues = method(board)
            assert len(clues) == 4
            for pair, clue in zip(board.pairs, clues):
                assert isinstance(clue, str)
                assert not clue in pair
                print(f"    {pair[0]} {pair[1]} -> {clue}")
