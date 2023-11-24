import random
from board import get_random_pair
from m01_direct import m01_basic
from llm import set_trial, dump_cache_stats_since_last_call

trials = 10
methods = [m01_basic]
temperatures = [0.0, 0.5, 0.9]


def generate_clues(trials, methods):
    for method in methods:
        print(f"Testing {method.__name__}")
        for temperature in temperatures:
            for trial in range(trials):
                print(f"  Trial {trial + 1} of {trials}")
                set_trial(trial)
                pair = get_random_pair()
                clue = method(temperature, pair)
                assert isinstance(clue, str)
                print(f"    {pair[0]} {pair[1]} -> {clue}")


if __name__ == "__main__":
    random.seed("Clover")
    generate_clues(trials, methods)
    dump_cache_stats_since_last_call()
