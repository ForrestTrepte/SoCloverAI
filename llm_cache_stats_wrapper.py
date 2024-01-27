from typing import Any, Dict, List, Optional
from langchain_core.caches import BaseCache, RETURN_VAL_TYPE
from collections import defaultdict
import tiktoken
import re


class Stat:
    def __init__(self) -> None:
        self.count = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def add_tokens(self, input_tokens: int, output_tokens: int) -> None:
        self.count += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens


class LlmCacheStatsWrapper:
    """Wrapper for an LLM cache that tracks the number of cache hits and tokens used."""

    def __init__(self, inner_cache: BaseCache) -> None:
        """
        Initialize the wrapper.

        Args:
            inner_cache: The cache to wrap.
            encoding: The encoding used by the model.
        """
        self.inner_cache = inner_cache
        self.cache_hits_by_model_name: Dict[str, Stat] = defaultdict(Stat)
        self.cache_misses_by_model_name: Dict[str, Stat] = defaultdict(Stat)
        self.model_name_re = re.compile(
            r"'model_name',\s*'(?P<model_name_format_1>[^']+)'|\"model_name\":\s*\"(?P<model_name_format_2>[^\"]+)\""
        )
        self.encodings: Dict[str, tiktoken.Encoding] = {}

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """
        Look up based on prompt and llm_string.

        Args:
            prompt: The prompt to use as the key.
            llm_string: The llm_string to use as part of the key.

        Returns:
            The value in the cache, or None if not found.
        """
        result = self.inner_cache.lookup(prompt, llm_string)
        if result:
            self.add_tokens(True, prompt, llm_string, result)
        return result

    def update(self, prompt: str, llm_string: str, return_val: Any) -> None:
        """
        Update cache based on prompt and llm_string and persist to JSON text file.

        Args:
            prompt: The prompt to use as the key.
            llm_string: The llm_string to use as part of the key.
            return_val: The value to store in the cache.
        """
        self.inner_cache.update(prompt, llm_string, return_val)
        self.add_tokens(False, prompt, llm_string, return_val)

    def add_tokens(
        self,
        is_hit: bool,
        prompt: str,
        llm_string: str,
        result: RETURN_VAL_TYPE,
    ) -> None:
        model_name_match = self.model_name_re.search(llm_string)
        if not model_name_match:
            raise ValueError(f"Could not find model_name in llm_string: {llm_string}")
        model_name = model_name_match.group(
            "model_name_format_1"
        ) or model_name_match.group("model_name_format_2")
        if model_name not in self.encodings:
            self.encodings[model_name] = tiktoken.encoding_for_model(model_name)
        encoding = self.encodings[model_name]

        input_tokens = len(encoding.encode(prompt))
        output_tokens = 0
        for generation in result:
            output_tokens += len(encoding.encode(generation.text))

        (
            self.cache_hits_by_model_name[model_name]
            if is_hit
            else self.cache_misses_by_model_name[model_name]
        ).add_tokens(input_tokens, output_tokens)

    def clear_cache_stats(self) -> None:
        self.cache_hits_by_model_name = defaultdict(Stat)
        self.cache_misses_by_model_name = defaultdict(Stat)

    def get_cache_stats_summary(self) -> str:
        result = ""

        all_cache_hits = self.cache_hits_by_model_name.values()
        all_cache_misses = self.cache_misses_by_model_name.values()
        result += f"LLM Cache: {sum([x.count for x in all_cache_hits])} hits, {sum([x.count for x in all_cache_misses])} misses\n"

        all_stats = list(all_cache_hits) + list(all_cache_misses)
        result += f"           {sum([x.input_tokens for x in all_cache_misses])} new input tokens, {sum([x.output_tokens for x in all_cache_misses])} new output tokens, {sum([x.input_tokens for x in all_stats])} total input tokens, {sum([x.output_tokens for x in all_stats])} total output tokens\n"

        try:
            miss_cost = 0.0
            total_cost = 0.0
            for model_name, cache_misses in self.cache_misses_by_model_name.items():
                cost = self._get_model_cost(model_name, cache_misses)
                miss_cost += cost
                total_cost += cost
            for model_name, cache_hits in self.cache_hits_by_model_name.items():
                cost = self._get_model_cost(model_name, cache_hits)
                total_cost += cost
            result += f"           new (this run) API cost: ${miss_cost:.2f}, total (including previously-cached runs) API cost: ${total_cost:.2f}\n"
        except ValueError as e:
            result += f"           Can't estimate cost: {e}\n"
        return result

    # from https://openai.com/pricing as of 11/7/23
    # (input cost, output cost) in USD per 1000 tokens
    _token_cost_by_model = {
        "gpt-4-1106-preview": (0.01, 0.03),
        "gpt-4": (0.03, 0.06),
        "gpt-4-32k": (0.06, 0.12),
        "gpt-3.5-turbo": (0.0030, 0.0060),
        "gpt-3.5-turbo-1106": (0.0010, 0.0020),
        "gpt-3.5-turbo-instruct": (0.0015, 0.0020),
    }

    @classmethod
    def _get_model_cost(cls, model_name: str, cache_misses: Stat) -> float:
        if not model_name in cls._token_cost_by_model:
            raise ValueError(f"Unknown model name: {model_name}")

        token_cost = cls._token_cost_by_model[model_name]
        cost = (
            cache_misses.input_tokens * token_cost[0]
            + cache_misses.output_tokens * token_cost[1]
        ) / 1000.0
        return cost
