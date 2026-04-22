from collections import defaultdict
from typing import List, Tuple

import generate
from results import Results


async def play(pairs: List[Tuple[str, str]]) -> None:
    methods = []
    methods.extend(generate.get_methods(["m06"]))
    methods.extend(generate.get_methods(["m08"]))
    methods.extend(generate.get_methods(["m09"]))
    results = await generate.parallel_generate(
        test_pairs=pairs,
        temperatures=[0.0, 0.5, 0.9],
        trials=1,
        methods=methods,
    )

    output_results(results)


def output_results(results: Results) -> None:
    clues_by_pair = defaultdict(list)
    for configuration in results.configurations:
        for trial in configuration.trials:
            pair = (trial.Word0, trial.Word1)
            clues_by_pair[pair].append(trial.ClueWord)

    for pair, clues in clues_by_pair.items():
        unique_clues = []
        seen = set()
        for clue in clues:
            if clue not in seen:
                unique_clues.append(clue)
                seen.add(clue)
        clues_str = ", ".join(unique_clues)
        print(f"{pair[0]}, {pair[1]}: {clues_str}")
