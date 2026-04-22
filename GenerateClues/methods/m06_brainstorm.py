from typing import List, Tuple

from llm import predict

template = """
You are playing the game So Clover. Create clues for the following word pair:
{word0} {word1}

Begin by brainstorming as many possible connections as you can think of as follows:
1. Think of the two words '{word0} {word1}' or '{word1} {word0}' as a single concept. Output a list of words that come to mind.
2. Think of {word0} individually. Output a list of associated words. Consider different meanings, common phrases, idioms, and pop culture involving the word.
3. Think of {word1} individually. Output a list of associated words. Consider different meanings, common phrases, idioms, and pop culture involving the word.
4. Reflecing on 1-3, output additional clue ideas that have an association with both {word0} and {word1}.
5. Output a discussion of which clue ideas are the most tightly associated with both words.

FINALLY, output your best clue CLUE in the following format:
Best: CLUE

Your clue should be a single word.
It cannot be a made up word. It should be in a dictionary or be in common use in pop culture.
For example, brand names, names of people, acronyms, numbers, and onomatopoeia are OK.
Your word cannot be in the same family or contain the words from the word pair.
Use lowercase unless it is a proper noun or if capitalizing the word would help the guesser understand the clue.
"""


async def generate(temperature: float, pair: Tuple[str, str]) -> List[str]:
    prediction = await predict(temperature, template, word0=pair[0], word1=pair[1])
    return prediction
