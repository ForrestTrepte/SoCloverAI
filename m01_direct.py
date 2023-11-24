from llm import predict

template = """
You are playing the game So Clover. Create clues for the following word pair:
{word0} {word1}

Output your clue CLUE in the following format:
Best: CLUE

Your clue should be a single word. Use lowercase unless it is a proper noun or capitalization the word would help the guesser understand the clue.
"""


def m01_basic(temperature, pair):
    prediction = predict(temperature, template, word0=pair[0], word1=pair[1])
    return prediction
