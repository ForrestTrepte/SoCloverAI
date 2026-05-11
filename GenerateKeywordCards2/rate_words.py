from typing import Literal, TypeAlias

from pydantic import BaseModel

from GenerateKeywordCards2.async_rng import AsyncRng

from .get_prompt import get_prompt
from .llm import generate_structured_async
from .llm_metadata import LlmMetadata

Ratings: TypeAlias = dict[str, float]


async def rate_words(
    model: str,
    prompt_name: str,
    words: list[str],
    reasoning_effort: Literal[
        "none", "minimal", "low", "medium", "high", "xhigh", "default"
    ],
    batch_size: int,
    rng: AsyncRng,
) -> tuple[dict[str, Ratings], LlmMetadata]:
    local_rng = rng.unwrap()
    words_shuffled = words.copy()
    local_rng.shuffle(words_shuffled)

    ratings = {}
    metadata = LlmMetadata.zero()
    for i in range(0, len(words), batch_size):
        batch_words = words_shuffled[i : i + batch_size]
        batch_ratings, batch_metadata = await _rate_words_batch(
            model, prompt_name, batch_words, reasoning_effort
        )
        ratings.update(batch_ratings)
        metadata += batch_metadata

    unrated = [w for w in words_shuffled if w not in ratings]
    if unrated:
        print(f"Warning: Re-rating {len(unrated)} unrated words")
        for i in range(0, len(unrated), batch_size):
            batch_words = unrated[i : i + batch_size]
            batch_ratings, batch_metadata = await _rate_words_batch(
                model, prompt_name, batch_words, reasoning_effort
            )
            ratings.update(batch_ratings)
            metadata += batch_metadata

    assert set(ratings.keys()) == set(words)
    return ratings, metadata


async def _rate_words_batch(
    model: str,
    prompt_name: str,
    words: list[str],
    reasoning_effort: Literal[
        "none", "minimal", "low", "medium", "high", "xhigh", "default"
    ],
) -> tuple[dict[str, Ratings], LlmMetadata]:

    class WordRating(BaseModel):
        word: str
        rating: float

    class WordRatingList(BaseModel):
        ratings: list[WordRating]

    prompt = get_prompt(prompt_name, "rate_words", {})

    response, metadata = await generate_structured_async(
        model=model,
        system_message=prompt.format(words=words),
        user_message=f"{words}",
        reasoning_effort="low",
        trial=0,
        response_format=WordRatingList,
    )

    result_ratings = {}
    words_set = set(words)
    is_valid = True
    for rating in response.ratings:
        if rating.word not in words_set:
            print(f"Error: rated word '{rating.word}' not in input words")
            is_valid = False
            continue

        if rating.word in result_ratings:
            print(f"Warning: duplicate rating for word '{rating.word}'")
            continue

        result_ratings[rating.word] = {
            "overall": rating.rating,
        }

    unrated_words = words_set - set(result_ratings.keys())
    if unrated_words:
        print(f"Warning: the following words were not rated: {unrated_words}")

    if not is_valid:
        raise ValueError("Invalid ratings response")

    return (result_ratings, metadata)
