import importlib
import os
import random

from board import get_random_pair
from llm import dump_cache_stats_since_last_call, set_trial
from methods import m01_direct, m02_expert, m03_clue_criteria
from results import Clue, Configuration, Results

temperatures = [0.0, 0.5, 0.9]
trials = 3
pairs_per_trial = 10

# Get all generate functions from the methods folder
methods = []
for filename in os.listdir("methods"):
    if filename.startswith("m") and filename.endswith(".py"):
        module_name = filename[:-3]
        module = importlib.import_module(f"methods.{module_name}")
        method = getattr(module, "generate")
        methods.append(method)


def generate():
    random.seed("Clover")
    test_pairs = [get_random_pair() for _ in range(pairs_per_trial)]

    configurations = []
    for method in methods:
        method_name = method.__module__
        methods_prefix = "methods."
        if method_name.startswith(methods_prefix):
            method_name = method_name[len(methods_prefix) :]
        print(f"Generating {method_name}")
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
                method=method_name, temperature=temperature, trials=clues
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
