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
def embeddings_cache(tmp_path: Path) -> Iterator[Cache]:
    cache = Cache(type=LiteLLMCacheType.DISK, disk_cache_dir=str(tmp_path))
    embeddings.set_embeddings_cache(cache)
    yield cache
    embeddings.set_embeddings_cache(None)


class TestEmbeddingsCache:
    @pytest.mark.asyncio
    async def test_caches_per_word_and_bypasses_global_cache(
        self, embeddings_cache: Cache, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        global_cache = Cache(
            type=LiteLLMCacheType.DISK, disk_cache_dir=str(tmp_path / "global")
        )
        monkeypatch.setattr(litellm, "cache", global_cache)
        calls: list[dict[str, Any]] = []
        real_aembedding = litellm.aembedding

        async def spy(**kwargs: Any) -> Any:
            calls.append(kwargs)
            return await real_aembedding(**kwargs, mock_response=[0.1, 0.2, 0.3])

        monkeypatch.setattr(embeddings, "aembedding", spy)

        assert not is_embedding_cached(MODEL, "apple")
        _, usage = await embed_words_async(MODEL, ["apple"])
        assert (usage.cache_hits, usage.cache_misses) == (0, 1)
        assert is_embedding_cached(MODEL, "apple")
        assert not is_embedding_cached(MODEL, "banana")
        # Cache keys include the model.
        assert not is_embedding_cached("openai/text-embedding-3-large", "apple")

        _, usage = await embed_words_async(MODEL, ["apple", "banana"])
        assert (usage.cache_hits, usage.cache_misses) == (1, 1)
        assert [c["input"] for c in calls] == [["apple"], ["banana"]]
        assert all(c["cache"] == {"no-cache": True, "no-store": True} for c in calls)

        # Nothing was written to litellm's global cache.
        await asyncio.sleep(0.2)
        assert len(global_cache.cache.disk_cache) == 0  # type: ignore[attr-defined]


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_spaces_acquisitions(self) -> None:
        limiter = _RateLimiter(per_minute=600)  # 0.1 s apart
        start = time.monotonic()
        await asyncio.gather(*(limiter.acquire() for _ in range(4)))
        elapsed = time.monotonic() - start
        assert 0.25 <= elapsed < 0.6
