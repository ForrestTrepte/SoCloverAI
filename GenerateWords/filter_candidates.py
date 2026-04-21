"""
Filter and organize So Clover! keyword candidates for human review.

Reads candidates.csv and produces candidates_review.csv with:
  - Existing-keyword duplicates removed
  - Near-duplicate pairs flagged (semantically similar candidates)
  - Rows organized by category
  - A 'duplicate_flag' column noting any similar words in the same pool

The output is meant for manual curation — you decide which near-duplicates to keep.

Usage:
    cd GenerateWords
    python filter_candidates.py

    # Tune similarity threshold for near-duplicate flagging:
    python filter_candidates.py --similarity-threshold 0.85
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).parent
load_dotenv(SCRIPT_DIR.parent / ".env")
GENERATE_CLUES_DIR = SCRIPT_DIR.parent / "GenerateClues"
CANDIDATES_CSV = SCRIPT_DIR / "candidates.csv"
REVIEW_CSV = SCRIPT_DIR / "candidates_review.csv"
EXISTING_KEYWORDS_CSV = SCRIPT_DIR.parent / "GenerateCards" / "CloverExistingKeywords.csv"

EMBEDDINGS_NPZ = GENERATE_CLUES_DIR / "words_by_frequency_embeddings.npz"
WORDS_JSON = GENERATE_CLUES_DIR / "words_by_frequency.json"

DEFAULT_SIMILARITY_THRESHOLD = 0.82

CATEGORY_ORDER = [
    "tech", "noun", "verb", "adjective", "whimsical",
    "animal", "food", "risque", "abstract", "embeddings", "llm",
]


def load_exclusion_set() -> set[str]:
    exclusions: set[str] = set()
    with open(EXISTING_KEYWORDS_CSV) as f:
        for line in f:
            word = line.strip()
            if word:
                exclusions.add(word.lower())
    return exclusions


def load_candidates() -> list[dict[str, str]]:
    rows = []
    with open(CANDIDATES_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def get_embedding(word: str, word_to_idx: dict[str, int], all_embeddings: np.ndarray) -> np.ndarray | None:
    idx = word_to_idx.get(word.lower())
    if idx is None:
        return None
    return all_embeddings[idx]


def find_near_duplicates(
    words: list[str],
    word_to_idx: dict[str, int],
    all_embeddings: np.ndarray,
    threshold: float,
) -> dict[str, list[str]]:
    """For each word, list other candidate words that are cosine-similar above threshold."""
    embeddings_list = []
    valid_words = []
    for word in words:
        emb = get_embedding(word, word_to_idx, all_embeddings)
        if emb is not None:
            embeddings_list.append(emb)
            valid_words.append(word)

    if not embeddings_list:
        return {}

    emb_matrix = np.stack(embeddings_list)  # (N, D)
    sim_matrix = emb_matrix @ emb_matrix.T  # (N, N)

    near_dupes: dict[str, list[str]] = defaultdict(list)
    n = len(valid_words)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                near_dupes[valid_words[i]].append(valid_words[j])
                near_dupes[valid_words[j]].append(valid_words[i])

    return dict(near_dupes)


def category_sort_key(row: dict[str, str]) -> int:
    cat = row.get("category", "").lower()
    for i, c in enumerate(CATEGORY_ORDER):
        if c in cat:
            return i
    return len(CATEGORY_ORDER)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter and organize So Clover! keyword candidates")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help=f"Cosine similarity above which two candidates are flagged as near-duplicates (default: {DEFAULT_SIMILARITY_THRESHOLD})",
    )
    args = parser.parse_args()

    if not CANDIDATES_CSV.exists():
        print(f"Error: {CANDIDATES_CSV} not found", file=sys.stderr)
        sys.exit(1)

    exclusions = load_exclusion_set()
    candidates = load_candidates()
    print(f"Loaded {len(candidates)} candidates")

    # Remove existing-keyword duplicates
    before = len(candidates)
    candidates = [r for r in candidates if r["word"].lower() not in exclusions]
    removed = before - len(candidates)
    if removed:
        print(f"Removed {removed} words that appear in existing keyword list")

    # Deduplicate by word (keep first occurrence)
    seen: set[str] = set()
    deduped = []
    for row in candidates:
        key = row["word"].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    if len(deduped) < len(candidates):
        print(f"Removed {len(candidates) - len(deduped)} exact duplicate rows")
    candidates = deduped

    print(f"Candidates after deduplication: {len(candidates)}")

    # Load embeddings for near-duplicate detection
    near_dupes: dict[str, list[str]] = {}
    if EMBEDDINGS_NPZ.exists():
        print("Loading embeddings for near-duplicate detection...")
        data = np.load(EMBEDDINGS_NPZ)
        all_embeddings: np.ndarray = data["embeddings"]
        with open(WORDS_JSON) as f:
            all_words: list[str] = json.load(f)
        n_embeddings = len(all_embeddings)
        word_to_idx = {w.lower(): i for i, w in enumerate(all_words[:n_embeddings])}

        words = [r["word"] for r in candidates]
        near_dupes = find_near_duplicates(words, word_to_idx, all_embeddings, args.similarity_threshold)
        flagged = sum(1 for v in near_dupes.values() if v)
        print(f"Found {flagged} words with at least one near-duplicate (threshold={args.similarity_threshold})")
    else:
        print("Warning: embeddings file not found — skipping near-duplicate detection")

    # Sort by category
    candidates.sort(key=category_sort_key)

    # Write review CSV
    fieldnames = ["word", "category", "source", "near_duplicates", "notes"]
    with open(REVIEW_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in candidates:
            dupes = near_dupes.get(row["word"], near_dupes.get(row["word"].lower(), []))
            writer.writerow({
                "word": row["word"],
                "category": row.get("category", ""),
                "source": row.get("source", ""),
                "near_duplicates": ", ".join(dupes) if dupes else "",
                "notes": row.get("notes", ""),
            })

    print(f"\nWrote {len(candidates)} candidates to {REVIEW_CSV}")
    print("\nCategory breakdown:")
    by_cat: dict[str, int] = defaultdict(int)
    for row in candidates:
        by_cat[row.get("category", "unknown")] += 1
    for cat, count in sorted(by_cat.items()):
        print(f"  {cat:<20} {count}")


if __name__ == "__main__":
    main()
