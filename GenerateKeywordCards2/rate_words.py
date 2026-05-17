from typing import TypeAlias

from pydantic import BaseModel, Field

from GenerateKeywordCards2.async_rng import AsyncRng

# from asyncio import TaskGroup  <-- Doesn't work in Jupyter, use CompatTaskGroup instead.
from GenerateKeywordCards2.compat_task_group import CompatTaskGroup as TaskGroup

from .get_prompt import get_prompt
from .llm import ReasoningEffort, generate_structured_async
from .llm_metadata import LlmMetadata

Ratings: TypeAlias = dict[str, float]
RatingsByWord: TypeAlias = dict[str, Ratings]


class WordAssociationAspects(BaseModel):
    semantic_category: float = Field(alias="Semantic/Category")
    functional: float = Field(alias="Functional")
    multiple_meanings: float = Field(alias="Multiple Meanings")
    metaphorical_symbolic: float = Field(alias="Metaphorical/Symbolic")
    idioms_phrases: float = Field(alias="Idioms/Phrases")
    wordplay: float = Field(alias="Wordplay")
    visual: float = Field(alias="Visual")
    emotional: float = Field(alias="Emotional")
    cultural_historical: float = Field(alias="Cultural/Historical")
    overall: float = Field(alias="Overall")


class WordGameplayAspects(BaseModel):
    recognizability: float = Field(alias="Recognizability")
    inoffensiveness: float = Field(alias="Inoffensiveness")
    evocativeness: float = Field(alias="Evocativeness")
    fun: float = Field(alias="Fun")
    overall: float = Field(alias="Overall")


class WordAspects(BaseModel):
    word: str
    associations: WordAssociationAspects
    gameplay: WordGameplayAspects

    def to_dict(self) -> Ratings:
        associations_dict = self.associations.model_dump(by_alias=True)
        associations_dict_with_prefix = {
            f"Association {k}": v for k, v in associations_dict.items()
        }
        gameplay_dict = self.gameplay.model_dump(by_alias=True)
        gameplay_dict_with_prefix = {
            f"Gameplay {k}": v for k, v in gameplay_dict.items()
        }
        result = associations_dict_with_prefix | gameplay_dict_with_prefix
        return result


class WordAspectsList(BaseModel):
    ratings: list[WordAspects]


response_format_fallback_description_overall = (
    "\nRespond with a JSON object containing a ratings array matching the following format:\n"
    '{"ratings": [{"word": "example", "rating": 5}, ...]}'
)

response_format_fallback_description_aspects = (
    "\nRespond with a JSON object containing a ratings array matching the following format:\n"
    '{"ratings": [{"word": "example", "associations": {"Semantic/Category": 5, "Functional": 5, "Multiple Meanings": 5, "Metaphorical/Symbolic": 5, "Idioms/Phrases": 5, "Wordplay": 5, "Visual": 5, "Emotional": 5, "Cultural/Historical": 5, "Overall": 5}, "gameplay": {"Recognizability": 5, "Evocativeness": 5, "Fun": 5, "Overall": 5}}, ...]}'
    "IMPORTANT OUTPUT CONTRACT: Be sure to respond with a JSON object containing the ratings array, and not just the array by itself."
)


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
    batch_ratings, batch_metadata = await _rate_words_batches(
        model, prompt_name, words_shuffled, reasoning_effort, batch_size
    )
    ratings.update(batch_ratings)
    metadata += batch_metadata

    rerating_count = 0
    maximum_rerating_passes = 2
    unrated = [w for w in words_shuffled if w not in ratings]
    while len(unrated) > 0 and rerating_count < maximum_rerating_passes:
        print(
            f"Warning: Re-rating {len(unrated)} unrated words (pass {rerating_count + 1})"
        )
        batch_ratings, batch_metadata = await _rate_words_batches(
            model, prompt_name, unrated, reasoning_effort, batch_size
        )

        if len(batch_ratings) == 0:
            print(
                "Error: no ratings obtained in re-rating pass, additional passes would just return the same cached result"
            )
            break

        ratings.update(batch_ratings)
        metadata += batch_metadata
        unrated = [w for w in words_shuffled if w not in ratings]
        rerating_count += 1

    if len(unrated) > 0:
        message = f"The following {len(unrated)} words were still not rated after {rerating_count} re-rating passes: {sorted(unrated)}"
        print(f"Error: {message}")
        raise RuntimeError(message)

    assert set(ratings.keys()) == set(words)
    return ratings, metadata


async def _rate_words_batches(
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

    # CONSIDER: Should response format models be defined at global scope instead of inside this function?
    #   Unfortunately, moving them invalidates the LLM cache.
    #   For now, we'll keep the WordRating format here and place additional formats such as WordAspects at global scope.
    class WordRating(BaseModel):
        word: str
        rating: float

        def to_dict(self) -> Ratings:
            result = {"overall": self.rating}
            return result

    class WordRatingList(BaseModel):
        ratings: list[WordRating]

    prompt = get_prompt(prompt_name, "rate_words", {})

    is_aspects = prompt_name.startswith("aspects_")
    response_format = WordAspectsList if is_aspects else WordRatingList
    fallback = (
        response_format_fallback_description_aspects
        if is_aspects
        else response_format_fallback_description_overall
    )

    print(
        f"> {model} {prompt_name} {reasoning_effort}: {len(words)} words [{words[0]}...]"
    )
    response, metadata = await generate_structured_async(
        model=model,
        system_message=prompt.format(words=words),
        user_message=f"{words}",
        reasoning_effort=reasoning_effort,
        trial=0,
        response_format=response_format,
        response_format_fallback_description=fallback,
    )
    assert isinstance(response, response_format)

    result_ratings = {}
    words_set = set(words)
    for rating in response.ratings:
        if rating.word not in words_set:
            print(f"Warning: rated word '{rating.word}' not in input words")
            continue

        if rating.word in result_ratings:
            print(f"Warning: duplicate rating for word '{rating.word}'")
            continue

        result_ratings[rating.word] = rating.to_dict()

    unrated_words = words_set - set(result_ratings.keys())
    if unrated_words:
        print(
            f"Warning: the following {len(unrated_words)} words were not rated: {sorted(unrated_words)}"
        )

    print(
        f"< {model} {prompt_name} {reasoning_effort}: {len(result_ratings)} words [{words[0]}...]"
    )

    return (result_ratings, metadata)
