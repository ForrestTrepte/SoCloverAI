"""
Discover additional So Clover! keyword candidates using pre-computed embeddings.

Searches the 60k common English word list for words that are:
  - HIGH CENTRALITY (versatile): many neighbors within a cosine similarity threshold
  - HIGH NOVELTY: far from the 880 existing game keywords on average

These words are likely to be versatile (rich association potential) while
feeling fresh relative to the base game.

Usage:
    cd GenerateWords
    python discover_via_embeddings.py

    # Tune parameters:
    python discover_via_embeddings.py --centrality-k 100 --novelty-threshold 0.15 --top-n 100

    # Preview without writing to candidates.csv:
    python discover_via_embeddings.py --dry-run
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).parent
load_dotenv(SCRIPT_DIR.parent / ".env")
GENERATE_CLUES_DIR = SCRIPT_DIR.parent / "GenerateClues"
CANDIDATES_CSV = SCRIPT_DIR / "candidates.csv"
EXISTING_KEYWORDS_CSV = SCRIPT_DIR / "CloverExistingKeywords.csv"

EMBEDDINGS_NPZ = GENERATE_CLUES_DIR / "words_by_frequency_embeddings.npz"
WORDS_JSON = GENERATE_CLUES_DIR / "words_by_frequency.json"

# Only consider words in the top N by frequency — rarer words are less useful as keywords
FREQUENCY_CUTOFF = 15000

# Cosine similarity threshold for counting a word as a "neighbor" (centrality)
DEFAULT_CENTRALITY_K = 50

# Minimum average cosine distance from existing keywords (0 = identical, 2 = opposite)
DEFAULT_NOVELTY_THRESHOLD = 0.12

DEFAULT_TOP_N = 75


def load_exclusion_set() -> set[str]:
    exclusions: set[str] = set()
    with open(EXISTING_KEYWORDS_CSV) as f:
        for line in f:
            word = line.strip()
            if word:
                exclusions.add(word.lower())
    return exclusions


def load_current_candidates() -> set[str]:
    if not CANDIDATES_CSV.exists():
        return set()
    candidates: set[str] = set()
    with open(CANDIDATES_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidates.add(row["word"].lower())
    return candidates


def append_candidates(new_rows: list[dict[str, str]]) -> None:
    with open(CANDIDATES_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "category", "source", "notes"])
        for row in new_rows:
            writer.writerow(row)


def is_useful_word(word: str) -> bool:
    """Filter out words that are unlikely to make good keywords."""
    if not word.isalpha():
        return False
    if len(word) < 3:
        return False
    # Skip stop words and very common function words
    skip = {
        "the", "and", "for", "that", "this", "with", "from", "but", "not",
        "are", "was", "were", "has", "have", "had", "its", "our", "his",
        "her", "they", "them", "their", "said", "also", "been", "will",
        "can", "may", "any", "all", "one", "two", "more", "most", "such",
        "than", "then", "when", "into", "out", "what", "who", "how",
        "did", "does", "would", "could", "should", "about", "which",
        "him", "she", "you", "your", "there", "use", "used", "being",
        "each", "both", "per", "via", "yet", "nor", "own",
    }
    return word.lower() not in skip


def compute_centrality(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Mean cosine similarity to each word's k nearest neighbors (excluding self)."""
    print(f"Computing centrality scores (top-{k} mean similarity)...")
    batch_size = 1000
    n = len(embeddings)
    scores = np.zeros(n)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sim = embeddings[start:end] @ embeddings.T  # (batch, N)
        # Zero out self-similarity so it doesn't rank as a neighbor
        for local_i in range(end - start):
            sim[local_i, start + local_i] = -1.0
        top_k_sim = np.partition(sim, -k, axis=1)[:, -k:]
        scores[start:end] = top_k_sim.mean(axis=1)
    return scores


def compute_novelty(
    candidate_embeddings: np.ndarray,
    existing_embeddings: np.ndarray,
) -> np.ndarray:
    """Average cosine distance from each candidate to the existing 880 keywords."""
    # similarity: (N_candidates, N_existing)
    print("Computing novelty scores...")
    batch_size = 1000
    n = len(candidate_embeddings)
    avg_distances = np.zeros(n)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sim = candidate_embeddings[start:end] @ existing_embeddings.T
        avg_distances[start:end] = (1 - sim).mean(axis=1)
    return avg_distances


def get_existing_embeddings(
    existing_words: set[str],
    all_words: list[str],
    all_embeddings: np.ndarray,
) -> np.ndarray:
    """Extract embeddings for the 880 existing keywords from the pre-computed array."""
    n_embeddings = len(all_embeddings)
    word_to_idx = {w.lower(): i for i, w in enumerate(all_words[:n_embeddings])}
    indices = []
    missing = []
    for word in existing_words:
        idx = word_to_idx.get(word.lower())
        if idx is not None:
            indices.append(idx)
        else:
            missing.append(word)
    if missing:
        print(f"  Warning: {len(missing)} existing keywords skipped in novelty check (not in embedding vocab): {', '.join(sorted(missing))}")
    return all_embeddings[indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover So Clover! keyword candidates via embeddings")
    parser.add_argument("--centrality-k", type=int, default=DEFAULT_CENTRALITY_K,
                        help=f"Number of nearest neighbors for centrality scoring (default: {DEFAULT_CENTRALITY_K})")
    parser.add_argument("--novelty-threshold", type=float, default=DEFAULT_NOVELTY_THRESHOLD,
                        help=f"Min avg cosine distance from existing keywords (default: {DEFAULT_NOVELTY_THRESHOLD})")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                        help=f"Number of top candidates to surface (default: {DEFAULT_TOP_N})")
    parser.add_argument("--frequency-cutoff", type=int, default=FREQUENCY_CUTOFF,
                        help=f"Only consider words ranked within top N by frequency (default: {FREQUENCY_CUTOFF})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print results without writing to candidates.csv")
    args = parser.parse_args()

    if not EMBEDDINGS_NPZ.exists():
        print(f"Error: embeddings file not found at {EMBEDDINGS_NPZ}", file=sys.stderr)
        sys.exit(1)

    print("Loading embeddings...")
    data = np.load(EMBEDDINGS_NPZ)
    all_embeddings: np.ndarray = data["embeddings"]

    with open(WORDS_JSON) as f:
        all_words: list[str] = json.load(f)

    n_vocab = min(args.frequency_cutoff, len(all_words))
    words = all_words[:n_vocab]
    embeddings = all_embeddings[:n_vocab]
    print(f"Loaded {n_vocab} words with embeddings of shape {embeddings.shape}")

    exclusions = load_exclusion_set()
    current_candidates = load_current_candidates()
    all_excluded = exclusions | current_candidates

    # Build mask of usable candidate words
    usable_mask = np.array([
        is_useful_word(w) and w.lower() not in all_excluded
        for w in words
    ])
    usable_indices = np.where(usable_mask)[0]
    usable_words = [words[i] for i in usable_indices]
    usable_embeddings = embeddings[usable_indices]
    print(f"Candidate pool after filtering: {len(usable_words)} words")

    # Centrality: mean similarity to top-k nearest neighbors
    centrality = compute_centrality(usable_embeddings, args.centrality_k)

    # Novelty: distance from existing keywords
    existing_embeddings = get_existing_embeddings(exclusions, all_words, all_embeddings)
    novelty = compute_novelty(usable_embeddings, existing_embeddings)

    # Filter by minimum novelty, then rank by centrality (versatility first)
    novel_mask = novelty >= args.novelty_threshold
    filtered_words = [w for w, m in zip(usable_words, novel_mask) if m]
    filtered_centrality = centrality[novel_mask]
    filtered_novelty = novelty[novel_mask]

    print(f"\nWords passing novelty threshold ({args.novelty_threshold}): {len(filtered_words)}")
    if len(filtered_novelty) > 0:
        lo, q1, med, q3, hi = np.percentile(filtered_novelty, [0, 25, 50, 75, 100])
        print(f"Novelty  min={lo:.4f}  q1={q1:.4f}  median={med:.4f}  q3={q3:.4f}  max={hi:.4f}")
        bins = 8
        counts, edges = np.histogram(filtered_novelty, bins=bins)
        bar_max = counts.max()
        bar_width = 20
        for i in range(bins):
            bar = round(counts[i] / bar_max * bar_width) if bar_max > 0 else 0
            print(f"  {edges[i]:.4f}–{edges[i+1]:.4f} | {'█' * bar:{bar_width}}  {counts[i]}")

    freq_rank = {w: i for i, w in enumerate(all_words)}

    # Sort by centrality descending
    order = np.argsort(filtered_centrality)[::-1]
    top_words = [filtered_words[i] for i in order[:args.top_n]]
    top_centrality = filtered_centrality[order[:args.top_n]]
    top_novelty = filtered_novelty[order[:args.top_n]]

    print(f"\nTop {len(top_words)} candidates (ranked by centrality):")
    print(f"{'Word':<20} {'Freq rank':>10} {'Centrality':>12} {'Novelty':>10}")
    print("-" * 56)
    for word, cent, nov in zip(top_words, top_centrality, top_novelty):
        rank = freq_rank.get(word.lower(), freq_rank.get(word, -1))
        print(f"{word:<20} {rank:>10,} {cent:>12.4f} {nov:>10.4f}")

    if not args.dry_run:
        new_rows = [
            {
                "word": word,
                "category": "embeddings",
                "source": "embeddings",
                "notes": "",
            }
            for word, cent, nov in zip(top_words, top_centrality, top_novelty)
        ]
        append_candidates(new_rows)
        print(f"\nAppended {len(new_rows)} words to {CANDIDATES_CSV}")
    else:
        print("\n(Dry run — nothing written to candidates.csv)")


if __name__ == "__main__":
    main()
