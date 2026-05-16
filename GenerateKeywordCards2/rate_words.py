from typing import TypeAlias

from pydantic import BaseModel

from GenerateKeywordCards2.async_rng import AsyncRng

# from asyncio import TaskGroup  <-- Doesn't work in Jupyter, use CompatTaskGroup instead.
from GenerateKeywordCards2.compat_task_group import CompatTaskGroup as TaskGroup

from .get_prompt import get_prompt
from .llm import ReasoningEffort, generate_structured_async
from .llm_metadata import LlmMetadata

Ratings: TypeAlias = dict[str, float]
RatingsByWord: TypeAlias = dict[str, Ratings]


async def rate_words(
    model: str,
    prompt_name: str,
    words: list[str],
    reasoning_effort: ReasoningEffort,
    batch_size: int,
    rng: AsyncRng,
) -> tuple[RatingsByWord, LlmMetadata]:
    local_rng = rng.unwrap()
    words_shuffled = words.copy()
    local_rng.shuffle(words_shuffled)

    ratings = {}
    metadata = LlmMetadata.zero()
    batch_ratings, batch_metadata = await rate_words_batches(
        model, prompt_name, words_shuffled, reasoning_effort, batch_size
    )
    ratings.update(batch_ratings)
    metadata += batch_metadata

    unrated = [w for w in words_shuffled if w not in ratings]
    if unrated:
        print(f"Warning: Re-rating {len(unrated)} unrated words")
        batch_ratings, batch_metadata = await rate_words_batches(
            model, prompt_name, unrated, reasoning_effort, batch_size
        )
        ratings.update(batch_ratings)
        metadata += batch_metadata

    assert set(ratings.keys()) == set(words)
    return ratings, metadata


async def rate_words_batches(
    model: str,
    prompt_name: str,
    words: list[str],
    reasoning_effort: ReasoningEffort,
    batch_size: int,
) -> tuple[RatingsByWord, LlmMetadata]:
    tasks = []
    ratings = {}
    metadata = LlmMetadata.zero()
    async with TaskGroup() as tg:
        for i in range(0, len(words), batch_size):
            batch_words = words[i : i + batch_size]
            tasks.append(
                tg.create_task(
                    _rate_words_batch(
                        model, prompt_name, batch_words, reasoning_effort
                    ),
                    eager_start=True,
                )
            )

    for task in tasks:
        batch_ratings, batch_metadata = task.result()
        ratings.update(batch_ratings)
        metadata += batch_metadata

    return ratings, metadata


async def _rate_words_batch(
    model: str,
    prompt_name: str,
    words: list[str],
    reasoning_effort: ReasoningEffort,
) -> tuple[RatingsByWord, LlmMetadata]:

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
        reasoning_effort=reasoning_effort,
        trial=0,
        response_format=WordRatingList,
        response_format_fallback_description=(
            "\nRespond with a JSON object matching the following format:\n"
            '{"ratings": [{"word": "example", "rating": 0.5}, ...]}'
        ),
    )

    result_ratings = {}
    words_set = set(words)
    for rating in response.ratings:
        if rating.word not in words_set:
            print(f"Warning: rated word '{rating.word}' not in input words")
            continue

        if rating.word in result_ratings:
            print(f"Warning: duplicate rating for word '{rating.word}'")
            continue

        result_ratings[rating.word] = {
            "overall": rating.rating,
        }

    unrated_words = words_set - set(result_ratings.keys())
    if unrated_words:
        print(f"Warning: the following words were not rated: {sorted(unrated_words)}")

    return (result_ratings, metadata)
