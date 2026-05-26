from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Iterator

from GenerateKeywordCards2.async_rng import AsyncRng

# from asyncio import TaskGroup  <-- Doesn't work in Jupyter, use CompatTaskGroup instead.
from GenerateKeywordCards2.compat_task_group import CompatTaskGroup as TaskGroup
from GenerateKeywordCards2.llm import ReasoningEffort
from GenerateKeywordCards2.llm_metadata import LlmMetadata
from GenerateKeywordCards2.rate_words import RatingsByWord, rate_words


@dataclass(frozen=True, order=True)
class ModelParams:
    model: str
    reasoning_effort: ReasoningEffort

    # batch size used with prompts that have a single overall rating
    batch_size_overall: int
    # batch size used with aspects prompts that have multiple ratings per word
    batch_size_aspects: int

    def __str__(self) -> str:
        return f"{self.model} {self.reasoning_effort} {self.batch_size_overall} {self.batch_size_aspects}"

    def short_str(self) -> str:
        model_slash_split = self.model.split("/")
        model_name = model_slash_split[-1]
        model_dash_split = model_name.split("-")
        base_model_name = model_dash_split[0]
        for model_dash_part in model_dash_split[1:]:
            if model_dash_part.startswith("20") and len(model_dash_part) >= 4:
                # likely a date, stop here
                break
            base_model_name += f"-{model_dash_part}"

        return base_model_name


@dataclass(frozen=True, order=True)
class VariationParams:
    model_params: ModelParams
    prompt_name: str

    @property
    def is_aspects(self) -> bool:
        return self.prompt_name.startswith("aspects_")

    def get_batch_size(self) -> int:
        result = (
            self.model_params.batch_size_aspects
            if self.is_aspects
            else self.model_params.batch_size_overall
        )
        return result

    def __str__(self) -> str:
        return f"{self.model_params} {self.prompt_name}"

    def short_str(self) -> str:
        return f"{self.model_params.short_str()} {self.prompt_name}"


@dataclass(frozen=True)
class RateWordsWithVariationsResult:
    ratings_by_word: RatingsByWord
    metadata: LlmMetadata


async def rate_words_with_variations(
    models_params: list[ModelParams],
    prompt_names: list[str],
    words: list[str],
) -> dict[VariationParams, RateWordsWithVariationsResult]:
    tasks_by_variation_params = {}
    async with TaskGroup() as tg:
        for model_params in models_params:
            for prompt_name in prompt_names:
                variation_params = VariationParams(model_params, prompt_name)
                print(f"* Rating {variation_params}...")
                tasks_by_variation_params[variation_params] = tg.create_task(
                    rate_words(
                        model_params.model,
                        prompt_name,
                        words,
                        model_params.reasoning_effort,
                        variation_params.get_batch_size(),
                        rng=AsyncRng(123),
                    ),
                    eager_start=True,
                )

    result = {}
    for variation_params in tasks_by_variation_params:
        task = tasks_by_variation_params[variation_params]
        ratings, metadata = task.result()
        result[variation_params] = RateWordsWithVariationsResult(ratings, metadata)

    return result


@dataclass(frozen=True)
class VariationView:
    """
    View abstraction for treating aspects results as two virtual variations.

    For aspects prompts, this class exposes two views:
    1. Overall rating comes from Association Overall,
    2. Overall rating comes from Gameplay Overall.
    """

    variation_params: VariationParams
    variation_result: RateWordsWithVariationsResult

    # False for non-aspects.
    # For aspects: True for Association Overall, False for Gameplay Overall.
    is_association: bool

    @property
    def overall_key(self) -> str:
        if not self.variation_params.is_aspects:
            return "overall"
        return "Association Overall" if self.is_association else "Gameplay Overall"

    def overall_rating(self, word: str) -> float:
        return self.variation_result.ratings_by_word[word][self.overall_key]

    def short_str(self) -> str:
        if not self.variation_params.is_aspects:
            return self.variation_params.short_str()
        suffix = "assoc" if self.is_association else "gamepl"
        return f"{self.variation_params.short_str()} [{suffix}]"

    def get_hash_key(self) -> tuple[VariationParams, bool]:
        # Helper for using VariationView as a dict key.
        # The underlying variation_params and is_association should be sufficient to uniquely identify the view.
        return (self.variation_params, self.is_association)


def iter_variation_views(
    results_by_variation: dict[VariationParams, RateWordsWithVariationsResult],
) -> Iterator[VariationView]:
    """
    Iterate over VariationView items, treating aspects variations as two virtual variations each.
    """
    for vp in sorted(results_by_variation.keys()):
        vr = results_by_variation[vp]
        if vp.prompt_name.startswith("aspects_"):
            # Two items per aspects variation
            yield VariationView(vp, vr, is_association=True)  # Association Overall
            yield VariationView(vp, vr, is_association=False)  # Gameplay Overall
        else:
            # One item for non-aspects variations
            yield VariationView(vp, vr, is_association=False)


def filter_results(
    results: Iterator[VariationView],
    names: list[str],
) -> Iterator[VariationView]:
    """
    Filter VariationView items based on a list of names.
    """
    for result in results:
        if result.short_str() in names:
            yield result


def combine_results(
    results: Iterator[VariationView],
) -> RateWordsWithVariationsResult:
    """
    Combine ratings from multiple variations by averaging them.
    """
    metadata = LlmMetadata.zero()
    seen_variation_params = set()
    variation_ratings_by_word = defaultdict(list)
    for result in results:
        # Don't double-count metadata from multiple views of the same variation.
        #   e.g. Association Overall and Gameplay Overall views of the same aspects variation.
        if result.variation_params not in seen_variation_params:
            seen_variation_params.add(result.variation_params)
            metadata += result.variation_result.metadata

        for word in result.variation_result.ratings_by_word.keys():
            overall_rating = result.overall_rating(word)
            variation_ratings_by_word[word].append(overall_rating)

    # Average the ratings for each word
    averaged_ratings_by_word = {
        word: {"overall": mean(ratings)}
        for word, ratings in variation_ratings_by_word.items()
    }

    return RateWordsWithVariationsResult(
        ratings_by_word=averaged_ratings_by_word, metadata=metadata
    )
