from dataclasses import dataclass

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
    batch_size: int

    def __str__(self) -> str:
        return f"{self.model} {self.reasoning_effort} {self.batch_size}"

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
                        model_params.batch_size,
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
