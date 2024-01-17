import importlib
import logging
import os
import random
import shutil

from board import get_random_pair
from llm import dump_cache_stats_since_last_call, set_trial
from log_to_file import log_to_file
from results import Clue, Configuration, Results

logger = logging.getLogger("SoCloverAI")


def clear_logs():
    if os.path.isdir("logs"):
        shutil.rmtree("logs")


def get_methods(allowed_prefixes):
    """Get all generate functions from the methods folder"""
    methods = []
    for filename in sorted(
        os.listdir("methods"), reverse=True
    ):  # reverse order to see results being generated for latest methods first
        if not filename.startswith("m") or not filename.endswith(".py"):
            continue
        if allowed_prefixes:
            if not any(filename.startswith(prefix) for prefix in allowed_prefixes):
                continue
        module_name = filename[:-3]
        module = importlib.import_module(f"methods.{module_name}")
        method = getattr(module, "generate")
        methods.append(method)
    return methods


def generate_with_standard_settings():
    """Generate with standard settings and random words for development and evaluation of methods."""
    temperatures = [0.0, 0.5, 0.9]
    trials = 3
    pairs_per_trial = 10
    random.seed("Clover")
    test_pairs = [get_random_pair() for _ in range(pairs_per_trial)]
    methods = get_methods(allowed_prefixes=None)
    results = generate(test_pairs, temperatures, trials, methods)
    return results


def generate(test_pairs, temperatures, trials, methods):
    random.seed("Clover")

    configurations = []
    for method in methods:
        method_name = method.__module__
        methods_prefix = "methods."
        if method_name.startswith(methods_prefix):
            method_name = method_name[len(methods_prefix) :]
        logger.info(f"Generating {method_name}")
        for temperature in temperatures:
            logger.info(f"  Temperature {temperature}")
            clues = []
            for trial in range(trials):
                logger.info(f"    Trial {trial + 1} of {trials}")
                set_trial(trial)
                for test_pair in test_pairs:
                    log_to_file(
                        f"logs/{method_name}/{test_pair[0]}-{test_pair[1]}-t{temperature}-{trial}.log"
                    )
                    clue = generate_clue(method, temperature, test_pair)
                    log_to_file(None)
                    clues.append(clue)
            configuration = Configuration(
                method=method_name, temperature=temperature, trials=clues
            )
            configurations.append(configuration)
    dump_cache_stats_since_last_call()
    # Reverse order so that the latest methods are last
    return Results(configurations=configurations[::-1])


def generate_clue(method, temperature, pair):
    candidates = method(temperature, pair)
    for candidate in candidates:
        assert isinstance(candidate, str)
    best = candidates[0]
    candidates_str = ", ".join(candidates[1:])
    logger.info(f"        {pair[0]} {pair[1]} -> {best} ({candidates_str})")
    clue = Clue(Word0=pair[0], Word1=pair[1], Clue=best)
    return clue
