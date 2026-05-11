from typing import Any

from litellm import ModelResponse
from litellm.cost_calculator import completion_cost


custom_cost_mapping_by_model = {
    # TODO: Remove deepseek custom mappings once https://github.com/BerriAI/litellm/issues/26709 is resolved
    "deepseek-v4-flash": [0.14, 0.28],
    "deepseek-v4-pro": [0.435, 0.87],
}


class LlmMetadata:
    def __init__(
        self,
        cache_hits: int,
        cache_misses: int,
        cached_cost: float,
        uncached_cost: float,
        output_tokens: int,
    ):
        self.cache_hits = cache_hits
        self.cache_misses = cache_misses
        self.cached_cost = cached_cost
        self.uncached_cost = uncached_cost
        self.output_tokens = output_tokens

    @property
    def total_requests(self) -> int:
        return self.cache_hits + self.cache_misses

    @property
    def total_cost(self) -> float:
        return self.cached_cost + self.uncached_cost

    def __add__(self, other: Any) -> "LlmMetadata":
        if other == 0:
            return self
        if not isinstance(other, LlmMetadata):
            return NotImplemented
        result = LlmMetadata(
            cache_hits=self.cache_hits + other.cache_hits,
            cache_misses=self.cache_misses + other.cache_misses,
            cached_cost=self.cached_cost + other.cached_cost,
            uncached_cost=self.uncached_cost + other.uncached_cost,
            output_tokens=self.output_tokens + other.output_tokens,
        )
        return result

    def __radd__(self, other: Any) -> "LlmMetadata":
        return self.__add__(other)

    def __str__(self) -> str:
        result = f"{self.cache_hits}/{self.total_requests} cache hits, ${self.uncached_cost:.4f} uncached cost, ${self.total_cost:.4f} total cost, {self.output_tokens} output tokens"
        return result

    @classmethod
    def from_response(cls, response: ModelResponse) -> "LlmMetadata":
        cache_hit = response._hidden_params.get("cache_hit", False)

        assert hasattr(response, "usage")
        if response.model in custom_cost_mapping_by_model:
            prices = custom_cost_mapping_by_model[response.model]
            cost = (
                response.usage.prompt_tokens / 1_000_000 * prices[0]
                + response.usage.completion_tokens / 1_000_000 * prices[1]
            )
        else:
            cost = completion_cost(response)

        output_tokens = response.usage.completion_tokens
        if cache_hit:
            return cls(
                cache_hits=1,
                cache_misses=0,
                cached_cost=cost,
                uncached_cost=0.0,
                output_tokens=output_tokens,
            )
        else:
            return cls(
                cache_hits=0,
                cache_misses=1,
                cached_cost=0.0,
                uncached_cost=cost,
                output_tokens=output_tokens,
            )

    @classmethod
    def zero(cls) -> "LlmMetadata":
        return cls(0, 0, 0.0, 0.0, 0)
