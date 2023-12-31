import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from results import Rating, Clue, Evaluations

evaluation_filename = "evaluations.json"
percentiles = [10, 50, 90]


def evaluate(results):
    evaluations_dict = None
    while True:
        evaluations_dict = get_completed_evaluations_dict(results)
        if evaluations_dict:
            break
        print(f"Please evaluate the clues with null scores in {evaluation_filename}")
        input("Press enter when done")

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
                evaluations_dict[clue_tuple] = Rating(Score=None, Legal=None)
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
        evaluations_dict[clue.as_tuple()] = clue.Rating
    return evaluations_dict


def save_evaluations_dict(evaluations_dict):
    evaluations_pydantic = Evaluations(clues=[])
    for clue_tuple, rating in sorted(evaluations_dict.items()):
        clue = Clue.from_tuple(clue_tuple)
        clue.Rating = rating
        evaluations_pydantic.clues.append(clue)
    with open(evaluation_filename, "w") as f:
        f.write(evaluations_pydantic.model_dump_json(indent=2))


def score_results(results, evaluations_dict):
    for configuration in results.configurations:
        for clue in configuration.trials:
            clue_tuple = clue.as_tuple()
            clue.Rating = evaluations_dict[clue_tuple]


def evaluate_results(results):
    print(
        f"Method, Temperature, {', '.join([f'{percentile}%' for percentile in percentiles])}"
    )
    for configuration in results.configurations:
        evaluate_configuration(configuration)

    plot_results(results, "average_bar")
    plot_results(results, "box")
    plot_results(results, "violin")


def evaluate_configuration(configuration):
    scores = [clue.Rating.get_adjusted_score() for clue in configuration.trials]
    percentile_scores = [
        np.percentile(scores, percentile) for percentile in percentiles
    ]
    percentile_scores_str = [f"{score:.2f}" for score in percentile_scores]
    print(
        f"{configuration.method}, {configuration.temperature}, {', '.join(percentile_scores_str)}"
    )


def plot_results(results, plot_type):
    # Prepare dataframe with method and score columns
    df_scores = []
    for configuration in results.configurations:
        method_name = f"{configuration.method} t{configuration.temperature}"
        for clue in configuration.trials:
            df_scores.append(
                {"Method": method_name, "Score": clue.Rating.get_adjusted_score()}
            )
    df = pd.DataFrame(df_scores)

    # Plot based on the selected plot type
    plt.figure(figsize=(10, 6))
    if plot_type == "box":
        sns.boxplot(x="Method", y="Score", data=df)
        plt.title("Clue Scores Distribution by Method and Temperature (box)")
    elif plot_type == "violin":
        sns.violinplot(x="Method", y="Score", data=df)
        plt.title("Clue Scores Distribution by Method and Temperature (violin)")
    elif plot_type == "average_bar":
        # Calculate the averages for each method
        df_avg = df.groupby("Method", as_index=False).mean()
        sns.barplot(x="Method", y="Score", data=df_avg)
        plt.title("Average Clue Scores by Method and Temperature")

    plt.ylabel("Score")
    plt.xticks(rotation=90)
    plt.show()
