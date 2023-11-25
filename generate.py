import random

from board import get_random_pair
from llm import dump_cache_stats_since_last_call, set_trial
from m01_direct import m01_basic
from results import Clue, Configuration, Results

methods = [m01_basic]
temperatures = [0.0, 0.5, 0.9]
trials = 3
pairs_per_trial = 10


def generate():
    random.seed("Clover")
    test_pairs = [get_random_pair() for _ in range(pairs_per_trial)]

    configurations = []
    for method in methods:
        print(f"Generating {method.__name__}")
        for temperature in temperatures:
            print(f"  Temperature {temperature}")
            clues = []
            for trial in range(trials):
                print(f"    Trial {trial + 1} of {trials}")
                set_trial(trial)
                for test_pair in test_pairs:
                    clue = generate_clue(method, temperature, test_pair)
                    clues.append(clue)
            configuration = Configuration(
                method=method.__name__, temperature=temperature, trials=clues
            )
            configurations.append(configuration)
    dump_cache_stats_since_last_call()
    return Results(configurations=configurations)


def generate_clue(method, temperature, pair):
    clue = method(temperature, pair)
    assert isinstance(clue, str)
    print(f"        {pair[0]} {pair[1]} -> {clue}")
    clue = Clue(Word0=pair[0], Word1=pair[1], Clue=clue)
    return clue
