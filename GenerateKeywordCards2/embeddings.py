"""
Word-embedding-based association helper, built on litellm.

Two intended uses (see Associations.ipynb):
  1. Compare embedding-derived nearest neighbors against SWOW human association
     norms (see swow.py), to see how well embeddings approximate human associations.
  2. Score candidate words with graph-theory-like metrics (centrality, neighbor
     diversity) to find words whose associations span a large, diverse set of
     other words -- good candidates for So Clover! keyword expansion.
"""

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from litellm import aembedding

from .swow import SWOWAssociations

try:
    from . import notes
except ImportError:
    pass

if TYPE_CHECKING:
    import networkx as nx  # type: ignore[import-untyped]

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

maximum_concurrent_embedding_requests = 25
_embedding_semaphore = asyncio.Semaphore(maximum_concurrent_embedding_requests)


@dataclass
class EmbeddingUsage:
    cache_hits: int
    cache_misses: int
    cached_cost: float
    uncached_cost: float
    prompt_tokens: int

    @property
    def total_cost(self) -> float:
        return self.cached_cost + self.uncached_cost

    @classmethod
    def zero(cls) -> "EmbeddingUsage":
        return cls(
            cache_hits=0,
            cache_misses=0,
            cached_cost=0.0,
            uncached_cost=0.0,
            prompt_tokens=0,
        )

    def __add__(self, other: "EmbeddingUsage") -> "EmbeddingUsage":
        return EmbeddingUsage(
            cache_hits=self.cache_hits + other.cache_hits,
            cache_misses=self.cache_misses + other.cache_misses,
            cached_cost=self.cached_cost + other.cached_cost,
            uncached_cost=self.uncached_cost + other.uncached_cost,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
        )

    def __str__(self) -> str:
        total = self.cache_hits + self.cache_misses
        return f"{self.cache_hits}/{total} cache hits, ${self.uncached_cost:.7f} uncached cost, ${self.total_cost:.7f} total cost, {self.prompt_tokens:,} prompt tokens"


async def _embed_one_async(model: str, word: str) -> tuple[list[float], EmbeddingUsage]:
    from litellm.cost_calculator import completion_cost

    async with _embedding_semaphore:
        response = await aembedding(model=model, input=[word])

    cache_hit = response._hidden_params.get("cache_hit", False)
    cost = completion_cost(completion_response=response, call_type="aembedding")
    usage = EmbeddingUsage(
        cache_hits=1 if cache_hit else 0,
        cache_misses=0 if cache_hit else 1,
        cached_cost=cost if cache_hit else 0.0,
        uncached_cost=0.0 if cache_hit else cost,
        prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
    )
    embedding = response.data[0]["embedding"]
    return embedding, usage


async def embed_words_async(
    words: list[str], model: str = DEFAULT_EMBEDDING_MODEL
) -> tuple[np.ndarray, EmbeddingUsage]:
    """
    Compute L2-normalized embeddings for a list of (distinct) words via litellm.

    Requests are made one word at a time so that litellm's response cache
    (e.g. `litellm.cache = litellm.Cache(type="disk")`) can dedupe at the word
    level as the vocabulary grows across notebook runs, rather than caching
    whole batches keyed on the exact set of words requested together.
    """
    assert len(words) == len(set(words)), "words must be distinct"

    results = await asyncio.gather(*(_embed_one_async(model, word) for word in words))
    raw_embeddings = np.array([embedding for embedding, _ in results], dtype=np.float64)
    total_usage: EmbeddingUsage = sum(
        (usage for _, usage in results), EmbeddingUsage.zero()
    )

    norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
    normalized = raw_embeddings / norms
    return normalized, total_usage


class WordEmbeddings:
    """
    A vocabulary of words with L2-normalized embeddings, supporting nearest-neighbor
    lookup and graph-theory-like centrality/diversity scoring.
    """

    def __init__(self, words: list[str], embeddings: np.ndarray) -> None:
        assert len(words) == len(embeddings)
        self.words = list(words)
        self._index_by_word = {word: i for i, word in enumerate(self.words)}
        self.embeddings = embeddings

    @classmethod
    async def from_words_async(
        cls, words: list[str], model: str = DEFAULT_EMBEDDING_MODEL
    ) -> tuple["WordEmbeddings", EmbeddingUsage]:
        deduped_words = list(dict.fromkeys(words))
        embeddings, usage = await embed_words_async(deduped_words, model=model)
        return cls(deduped_words, embeddings), usage

    def __len__(self) -> int:
        return len(self.words)

    def __contains__(self, word: str) -> bool:
        return word in self._index_by_word

    def get_embedding(self, word: str) -> np.ndarray:
        return cast(np.ndarray, self.embeddings[self._index_by_word[word]])

    def similarity(self, word_a: str, word_b: str) -> float:
        return float(self.get_embedding(word_a) @ self.get_embedding(word_b))

    def nearest(
        self, word: str, count: int = 10, exclude_self: bool = True
    ) -> list[tuple[str, float]]:
        return self.nearest_to_embedding(
            self.get_embedding(word),
            count,
            exclude_index=self._index_by_word[word] if exclude_self else None,
        )

    def nearest_to_embedding(
        self, embedding: np.ndarray, count: int, exclude_index: int | None = None
    ) -> list[tuple[str, float]]:
        similarities = self.embeddings @ embedding
        if exclude_index is not None:
            similarities = similarities.copy()
            similarities[exclude_index] = -1.0
        top_indices = np.argsort(similarities)[::-1][:count]
        return [(self.words[i], float(similarities[i])) for i in top_indices]

    def centrality_scores(self, k: int = 10) -> dict[str, float]:
        """
        Mean cosine similarity to each word's k nearest neighbors: how tightly a
        word sits amid a well-populated region of the embedding space (hub-ness).
        """
        similarity_matrix = self.embeddings @ self.embeddings.T
        np.fill_diagonal(similarity_matrix, -1.0)
        top_k_similarities = np.partition(similarity_matrix, -k, axis=1)[:, -k:]
        scores = top_k_similarities.mean(axis=1)
        return dict(zip(self.words, (float(s) for s in scores)))

    def neighbor_diversity_scores(self, k: int = 10) -> dict[str, float]:
        """
        For each word, how spread out (mutually dissimilar) its k nearest neighbors
        are from one another: 1 - mean pairwise similarity among the neighbor set.

        A word with both high centrality (many close neighbors) and high neighbor
        diversity (those neighbors are themselves unrelated to each other) bridges
        multiple distinct concepts -- a candidate whose associations "span a large
        amount of diverse words".
        """
        similarity_matrix = self.embeddings @ self.embeddings.T
        n = len(self.words)
        self_excluded = similarity_matrix.copy()
        np.fill_diagonal(self_excluded, -1.0)
        neighbor_indices = np.argpartition(self_excluded, -k, axis=1)[:, -k:]

        scores = np.empty(n)
        for i in range(n):
            neighbor_embeddings = self.embeddings[neighbor_indices[i]]
            pairwise = neighbor_embeddings @ neighbor_embeddings.T
            off_diagonal_mask = ~np.eye(k, dtype=bool)
            scores[i] = 1.0 - pairwise[off_diagonal_mask].mean()
        return dict(zip(self.words, (float(s) for s in scores)))

    def bridging_scores(self, k: int = 10) -> dict[str, float]:
        """
        Combines centrality and neighbor diversity (each rank-normalized to [0, 1]
        then averaged) into a single score for surfacing hub words whose nearest
        neighbors span diverse regions of the embedding space.
        """
        centrality = self.centrality_scores(k)
        diversity = self.neighbor_diversity_scores(k)

        def rank_normalize(values: dict[str, float]) -> dict[str, float]:
            ordered_words = sorted(values, key=lambda w: values[w])
            n = len(ordered_words)
            return {word: i / (n - 1) for i, word in enumerate(ordered_words)}

        centrality_rank = rank_normalize(centrality)
        diversity_rank = rank_normalize(diversity)
        return {
            word: (centrality_rank[word] + diversity_rank[word]) / 2
            for word in self.words
        }

    def to_networkx_graph(self, k: int = 10) -> "nx.Graph[str]":
        """
        Builds an undirected weighted k-nearest-neighbor graph (edge weight = cosine
        similarity) for use with full graph-theoretic algorithms (betweenness
        centrality, community detection, ...) from networkx.

        Requires the optional `networkx` package.
        """
        try:
            import networkx as nx
        except ImportError as e:
            raise ImportError(
                "to_networkx_graph requires the 'networkx' package: pip install networkx"
            ) from e

        graph = nx.Graph()
        graph.add_nodes_from(self.words)
        similarity_matrix = self.embeddings @ self.embeddings.T
        self_excluded = similarity_matrix.copy()
        np.fill_diagonal(self_excluded, -1.0)
        neighbor_indices = np.argpartition(self_excluded, -k, axis=1)[:, -k:]
        for i, word in enumerate(self.words):
            for j in neighbor_indices[i]:
                neighbor_word = self.words[j]
                weight = float(similarity_matrix[i, j])
                graph.add_edge(word, neighbor_word, weight=weight)
        return graph


@dataclass(frozen=True)
class SwowEmbeddingComparison:
    word: str
    embedding_neighbors: list[str]
    swow_responses: list[str]
    overlap: list[str]

    @property
    def overlap_count(self) -> int:
        return len(self.overlap)

    @property
    def jaccard(self) -> float:
        union_size = len(set(self.embedding_neighbors) | set(self.swow_responses))
        if union_size == 0:
            return 0.0
        return len(self.overlap) / union_size


def compare_with_swow(
    embeddings: WordEmbeddings,
    swow: SWOWAssociations,
    word: str,
    count: int = 10,
    direction: Literal["forward", "backward"] = "forward",
) -> SwowEmbeddingComparison:
    """
    Compares a word's top embedding neighbors against its SWOW human association
    responses (forward: cue -> response, backward: response -> cue), to gauge how
    well embedding similarity approximates human free-association norms.
    """
    embedding_neighbors = [w for w, _ in embeddings.nearest(word, count)]

    word_associations = swow.associations.get(word)
    responses = getattr(word_associations, direction, {}) if word_associations else {}
    swow_responses = list(responses.keys())[:count]

    overlap = [w for w in embedding_neighbors if w in swow_responses]
    return SwowEmbeddingComparison(
        word=word,
        embedding_neighbors=embedding_neighbors,
        swow_responses=swow_responses,
        overlap=overlap,
    )


def summarize_swow_comparisons(
    comparisons: list[SwowEmbeddingComparison],
) -> dict[str, float]:
    """Aggregate overlap/jaccard statistics across many compare_with_swow() results."""
    if not comparisons:
        return {"mean_overlap_count": 0.0, "mean_jaccard": 0.0}
    return {
        "mean_overlap_count": sum(c.overlap_count for c in comparisons)
        / len(comparisons),
        "mean_jaccard": sum(c.jaccard for c in comparisons) / len(comparisons),
    }
