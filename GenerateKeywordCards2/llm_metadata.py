from typing import Any

from litellm import ModelResponse
from litellm.cost_calculator import completion_cost


class LlmMetadata:
    def __init__(
        self,
        cache_hits: int,
        cache_misses: int,
        cached_cost: float,
        uncached_cost: float,
    ):
        self.cache_hits = cache_hits
        self.cache_misses = cache_misses
        self.cached_cost = cached_cost
        self.uncached_cost = uncached_cost

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
        )
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __str__(self) -> str:
        result = f"{self.cache_hits}/{self.total_requests} cache hits, ${self.uncached_cost:.4f} uncached cost, ${self.total_cost:.4f} total cost"
        return result

    @classmethod
    def from_response(cls, response: ModelResponse) -> "LlmMetadata":
        cache_hit = response._hidden_params.get("cache_hit", False)
        cost = completion_cost(response)
        if cache_hit:
            return cls(
                cache_hits=1,
                cache_misses=0,
                cached_cost=cost,
                uncached_cost=0.0,
            )
        else:
            return cls(
                cache_hits=0,
                cache_misses=1,
                cached_cost=0.0,
                uncached_cost=cost,
            )
