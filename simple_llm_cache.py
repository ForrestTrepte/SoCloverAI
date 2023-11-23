import json
import os
from typing import Any, Dict, List, Optional
from langchain.cache import InMemoryCache, RETURN_VAL_TYPE
from langchain.load.dump import dumps
from langchain.load.load import loads
from langchain.schema import Generation

class SimpleLlmCache(InMemoryCache):
    """Cache that stores things in memory and persists to a JSON text file."""

    def __init__(self, filename: str = "llm-cache.json") -> None:
        """
        Initialize with empty cache and filename for JSON text file.
        """
        self._cache: Dict[str, List[str]] = {}
        self._trial = 0
        self._filename = filename
        try:
            with open(self._filename, 'r') as f:
                self._cache = json.load(f)
        except FileNotFoundError:
            pass

    def set_trial(self, trial: int) -> None:
        """Set a trial index. This permits generating new results for the same prompt and llm_string.
        For example, when testing LLM prompts this can used to generate multiple different responses for the prompt,
        while still caching the results for each trial."""
        self._trial = trial

    def _get_key(self, prompt: str, llm_string: str) -> str:
        """
        Convert the tuple key to a string key.

        Args:
            prompt: The prompt to use as the key.
            llm_string: The llm_string to use as part of the key.

        Returns:
            The string key.
        """
        return f"trial {self._trial} ::: {prompt} ::: {llm_string})"
    
    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """
        Look up based on prompt and llm_string.

        Args:
            prompt: The prompt to use as the key.
            llm_string: The llm_string to use as part of the key.

        Returns:
            The value in the cache, or None if not found.
        """
        key = self._get_key(prompt, llm_string)
        generations = []
        value = self._cache.get(key, None)
        if not value:
            return None
        for generation in value:
            generations.append(loads(generation))
        return generations

    def update(self, prompt: str, llm_string: str, return_val: Any) -> None:
        """
        Update cache based on prompt and llm_string and persist to JSON text file.

        Args:
            prompt: The prompt to use as the key.
            llm_string: The llm_string to use as part of the key.
            return_val: The value to store in the cache.
        """
        key = self._get_key(prompt, llm_string)
        generations = []
        for generation in return_val:
            generations.append(dumps(generation))
        self._cache[key] = generations
        with open(self._filename, 'w') as f:
            json.dump(self._cache, f, indent=4)

    def clear(self) -> None:
        """
        Clear cache and remove JSON text file.
        """
        try:
            self._cache = {}
            os.remove(self._filename)
        except FileNotFoundError:
            pass
