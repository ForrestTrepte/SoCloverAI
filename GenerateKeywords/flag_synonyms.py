"""
Flag near-synonym pairs among keyword candidates for human review.

Computes pairwise cosine similarity between all candidates and prints pairs
above a threshold, grouped so you can decide which to keep or remove.

Usage:
    cd GenerateKeywords
    python flag_synonyms.py

    # Tune threshold (lower = more pairs flagged):
    python flag_synonyms.py --threshold 0.88

    # Write flagged pairs to CSV for easier review:
    python flag_synonyms.py --output synonym_pairs.csv
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
EMBEDDINGS_NPZ = GENERATE_CLUES_DIR / "words_by_frequency_embeddings.npz"
WORDS_JSON = GENERATE_CLUES_DIR / "words_by_frequency.json"

DEFAULT_THRESHOLD = 0.88


def load_candidates() -> list[dict[str, str]]:
    with open(CANDIDATES_CSV) as f:
        return list(csv.DictReader(f))


def build_word_index(all_words: list[str], n_embeddings: int) -> dict[str, int]:
    return {w.lower(): i for i, w in enumerate(all_words[:n_embeddings])}


def get_embedding(word: str, word_to_idx: dict[str, int], embeddings: np.ndarray) -> np.ndarray | None:
    idx = word_to_idx.get(word.lower())
    return embeddings[idx] if idx is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Flag near-synonym candidate pairs")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Cosine similarity threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--output", type=str, default=None,
                        help="Write flagged pairs to this CSV file")
    args = parser.parse_args()

    candidates = load_candidates()
    words = [r["word"] for r in candidates]
    cat_of = {r["word"]: r["category"] for r in candidates}

    if not EMBEDDINGS_NPZ.exists():
        print(f"Error: embeddings not found at {EMBEDDINGS_NPZ}", file=sys.stderr)
        sys.exit(1)

    print("Loading embeddings...")
    data = np.load(EMBEDDINGS_NPZ)
    all_embeddings: np.ndarray = data["embeddings"]
    with open(WORDS_JSON) as f:
        all_words: list[str] = json.load(f)
    word_to_idx = build_word_index(all_words, len(all_embeddings))

    emb_list = []
    valid_words = []
    missing = []
    for word in words:
        emb = get_embedding(word, word_to_idx, all_embeddings)
        if emb is not None:
            emb_list.append(emb)
            valid_words.append(word)
        else:
            missing.append(word)

    if missing:
        print(f"Warning: {len(missing)} words not in embedding vocab, skipped: {', '.join(missing)}")

    emb_matrix = np.stack(emb_list)
    sim_matrix = emb_matrix @ emb_matrix.T

    pairs: list[tuple[str, str, float]] = []
    n = len(valid_words)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= args.threshold:
                pairs.append((valid_words[i], valid_words[j], float(sim_matrix[i, j])))

    pairs.sort(key=lambda x: -x[2])

    print(f"\nFound {len(pairs)} near-synonym pairs (threshold={args.threshold}):\n")
    print(f"{'Word A':<20} {'Word B':<20} {'Category A':<18} {'Category B':<18} {'Similarity':>10}")
    print("-" * 90)
    for a, b, sim in pairs:
        print(f"{a:<20} {b:<20} {cat_of.get(a,''):<18} {cat_of.get(b,''):<18} {sim:>10.4f}")

    if args.output:
        out_path = SCRIPT_DIR / args.output
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["word_a", "word_b", "category_a", "category_b", "similarity", "keep"])
            for a, b, sim in pairs:
                writer.writerow([a, b, cat_of.get(a, ""), cat_of.get(b, ""), f"{sim:.4f}", ""])
        print(f"\nWrote {len(pairs)} pairs to {out_path}")
    else:
        print(f"\nTip: run with --output synonym_pairs.csv to save for review")


if __name__ == "__main__":
    main()
