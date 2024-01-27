from typing import List, Tuple

from llm import predict

template = """
You are playing the game So Clover. Create clues for the following word pair:
{word0} {word1}

Output your clue CLUE in the following format:
Best: CLUE

Your clue should be a single word.
It cannot be a made up word. It should be in a dictionary or be in common use in pop culture.
For example, brand names, names of people, acronyms, numbers, and onomatopoeia are OK.
Your word cannot be in the same family or contain the words from the word pair.
Use lowercase unless it is a proper noun or if capitalizing the word would help the guesser understand the clue.
"""


def generate(temperature: float, pair: Tuple[str, str]) -> List[str]:
    prediction = predict(temperature, template, word0=pair[0], word1=pair[1])
    return prediction
