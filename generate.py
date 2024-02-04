import importlib
import inspect
import logging
import os
import random
import shutil
from typing import Any, Callable, Coroutine, List, Optional, Tuple

from board import get_random_pair
from llm import dump_cache_stats_since_last_call, set_trial
from log_to_file import log_to_file
from results import Clue, Configuration, Results

logger = logging.getLogger("SoCloverAI")
GenerateType = Callable[[float, Tuple[str, str]], Coroutine[None, None, list[str]]]


def clear_logs() -> None:
    if os.path.isdir("logs"):
        shutil.rmtree("logs")


def get_methods(
    allowed_prefixes: Optional[List[str]],
) -> List[GenerateType]:
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
        method = get_generate_method(method)
        methods.append(method)
    return methods


def get_generate_method(method: Any) -> GenerateType:
    expected_arg_types = [float, Tuple[str, str]]
    expected_return_type = List[str]
    assert callable(method)
    sig = inspect.signature(method)
    assert len(sig.parameters) == len(expected_arg_types)
    for i, parameter in enumerate(sig.parameters.values()):
        assert parameter.annotation == expected_arg_types[i]
    assert sig.return_annotation == expected_return_type
    result: GenerateType = method
    return result


async def generate_with_standard_settings() -> Results:
    """Generate with standard settings and random words for development and evaluation of methods."""
    temperatures = [0.0, 0.5, 0.9]
    trials = 3
    pairs_per_trial = 10
    random.seed("Clover")
    test_pairs = [get_random_pair() for _ in range(pairs_per_trial)]
    methods = get_methods(allowed_prefixes=None)
    results = await generate(test_pairs, temperatures, trials, methods)
    return results


async def generate(
    test_pairs: List[Tuple[str, str]],
    temperatures: List[float],
    trials: int,
    methods: list[GenerateType],
) -> Results:
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
            clues: List[Clue] = []
            for trial in range(trials):
                logger.info(f"    Trial {trial + 1} of {trials}")
                set_trial(trial)
                for test_pair in test_pairs:
                    log_to_file(
                        f"logs/{method_name}/{test_pair[0]}-{test_pair[1]}-t{temperature}-{trial}.log"
                    )
                    pair_clues = await generate_clues(method, temperature, test_pair)
                    clues.extend(pair_clues)
                    log_to_file(None)
            configuration = Configuration(
                method=method_name, temperature=temperature, trials=clues
            )
            configurations.append(configuration)
    dump_cache_stats_since_last_call()
    return Results(configurations=configurations)


async def generate_clues(
    method: GenerateType, temperature: float, pair: Tuple[str, str]
) -> List[Clue]:
    candidates = await method(temperature, pair)
    for candidate in candidates:
        assert isinstance(candidate, str)
    logger.info(f"        {pair[0]} {pair[1]} -> {', '.join(candidates)}")
    clues = [
        Clue(Word0=pair[0], Word1=pair[1], ClueWord=candidate)
        for candidate in candidates
    ]
    return clues
