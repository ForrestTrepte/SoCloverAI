import asyncio
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import litellm
import pytest
from litellm.caching.caching import Cache
from litellm.types.caching import LiteLLMCacheType

from . import embeddings
from .embeddings import (
    _RateLimiter,
    embed_words_async,
    is_embedding_cached,
)

MODEL = "openai/text-embedding-3-small"


@pytest.fixture
def disk_cache(tmp_path: Path) -> Iterator[Cache]:
    original_cache = litellm.cache
    litellm.cache = Cache(type=LiteLLMCacheType.DISK, disk_cache_dir=str(tmp_path))
    yield litellm.cache
    litellm.cache = original_cache


async def _wait_until_cached(model: str, word: str) -> None:
    # litellm writes to its cache in a background task after the response returns.
    for _ in range(100):
        if is_embedding_cached(model, word):
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"{word!r} never appeared in the cache")


class TestIsEmbeddingCached:
    @pytest.mark.asyncio
    async def test_predicts_hits_and_misses_consistently_with_litellm(
        self, disk_cache: Cache
    ) -> None:
        assert not is_embedding_cached(MODEL, "apple")

        response = await litellm.aembedding(
            model=MODEL, input=["apple"], mock_response=[0.1, 0.2, 0.3]
        )
        assert not response._hidden_params.get("cache_hit")
        await _wait_until_cached(MODEL, "apple")

        assert is_embedding_cached(MODEL, "apple")
        assert not is_embedding_cached(MODEL, "banana")
        # Cache keys include the model.
        assert not is_embedding_cached("openai/text-embedding-3-large", "apple")

        # The prediction agrees with what litellm reports on a real call.
        cached_response = await litellm.aembedding(model=MODEL, input=["apple"])
        assert cached_response._hidden_params.get("cache_hit") is True


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_spaces_acquisitions(self) -> None:
        limiter = _RateLimiter(per_minute=600)  # 0.1 s apart
        start = time.monotonic()
        await asyncio.gather(*(limiter.acquire() for _ in range(4)))
        elapsed = time.monotonic() - start
        assert 0.25 <= elapsed < 0.6
