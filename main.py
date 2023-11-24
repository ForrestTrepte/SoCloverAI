import os
import random

import numpy as np

from board import get_random_pair
from llm import dump_cache_stats_since_last_call, set_trial
from m01_direct import m01_basic
from results import Clue, Configuration, Evaluations, Results

trials = 10
methods = [m01_basic]
temperatures = [0.0, 0.5, 0.9]
evaluation_filename = "evaluations.json"
percentiles = [0, 1, 5, 10, 50, 90, 95, 99, 100]


def generate(trials, methods):
    configurations = []
    for method in methods:
        print(f"Testing {method.__name__}")
        for temperature in temperatures:
            print(f"  Temperature {temperature}")
            clues = []
            for trial in range(trials):
                print(f"    Trial {trial + 1} of {trials}")
                clue = generate_clue(method, temperature, trial)
                clues.append(clue)
            configuration = Configuration(
                method=method.__name__, temperature=temperature, trials=clues
            )
            configurations.append(configuration)
    return Results(configurations=configurations)


def generate_clue(method, temperature, trial):
    set_trial(trial)
    pair = get_random_pair()
    clue = method(temperature, pair)
    assert isinstance(clue, str)
    print(f"      {pair[0]} {pair[1]} -> {clue}")
    clue = Clue(Word0=pair[0], Word1=pair[1], Clue=clue)
    return clue


def evaluate(results):
    evaluations_dict = None
    while True:
        evaluations_dict = get_completed_evaluations_dict(results)
        if evaluations_dict:
            break
        print("Please evaluate the clues with null scores in {evaluation_filename}")
        input("Press enter when done")

    print("Evaluating results")
    score_results(results, evaluations_dict)
    evaluate_results(results)


def get_completed_evaluations_dict(results):
    evaluations_dict = load_evaluations_dict()
    is_scored = True
    for configuration in results.configurations:
        for clue in configuration.trials:
            clue_tuple = clue.as_tuple()
            if (
                clue_tuple in evaluations_dict
                and evaluations_dict[clue_tuple] is not None
            ):
                continue
            else:
                evaluations_dict[clue_tuple] = None
                is_scored = False
    save_evaluations_dict(evaluations_dict)
    if not is_scored:
        return None
    return evaluations_dict


def load_evaluations_dict():
    if os.path.isfile(evaluation_filename):
        with open(evaluation_filename, "r") as f:
            evaluations_json = f.read()
            evaluations_pydantic = Evaluations.model_validate_json(evaluations_json)
    else:
        evaluations_pydantic = Evaluations(clues=[])
    evaluations_dict = {}
    for clue in evaluations_pydantic.clues:
        evaluations_dict[clue.as_tuple()] = clue.Score
    return evaluations_dict


def save_evaluations_dict(evaluations_dict):
    evaluations_pydantic = Evaluations(clues=[])
    for clue_tuple, score in sorted(evaluations_dict.items()):
        clue = Clue.from_tuple(clue_tuple)
        clue.Score = score
        evaluations_pydantic.clues.append(clue)
    with open(evaluation_filename, "w") as f:
        f.write(evaluations_pydantic.model_dump_json(indent=2))


def score_results(results, evaluations_dict):
    for configuration in results.configurations:
        for clue in configuration.trials:
            clue_tuple = clue.as_tuple()
            clue.Score = evaluations_dict[clue_tuple]


def evaluate_results(results):
    for configuration in results.configurations:
        evaluate_configuration(configuration)


def evaluate_configuration(configuration):
    print(
        f"Method, Temperature, {', '.join([f'{percentile}%' for percentile in percentiles])}"
    )
    scores = [clue.Score for clue in configuration.trials]
    percentile_scores = [
        np.percentile(scores, percentile) for percentile in percentiles
    ]
    percentile_scores_str = [f"{score:.2f}" for score in percentile_scores]
    print(
        f"{configuration.method}, {configuration.temperature}, {', '.join(percentile_scores_str)}"
    )


if __name__ == "__main__":
    random.seed("Clover")
    results = generate(trials, methods)
    dump_cache_stats_since_last_call()
    evaluate(results)
