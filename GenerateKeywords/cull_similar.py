"""
Cull near-duplicate words from the shortlist to produce the final keyword set.

Reads candidates_shortlist.csv and repeatedly removes the lower-rated word from
the most similar pair, stopping when no pair exceeds the similarity threshold or
the remove limit is reached. Writes the result to candidates_final.csv.

Usage:
    cd GenerateKeywords
    python cull_similar.py

    # Tune stopping criteria:
    python cull_similar.py --threshold 0.82 --max-remove 15

    # Preview without writing:
    python cull_similar.py --dry-run
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
SHORTLIST_CSV = SCRIPT_DIR / "candidates_shortlist.csv"
FINAL_CSV = SCRIPT_DIR / "candidates_final.csv"
EMBEDDINGS_NPZ = GENERATE_CLUES_DIR / "words_by_frequency_embeddings.npz"
WORDS_JSON = GENERATE_CLUES_DIR / "words_by_frequency.json"

DEFAULT_THRESHOLD = 0.85
DEFAULT_MAX_REMOVE = 10
DEFAULT_HUMAN_WEIGHT = 2.0
DEFAULT_LLM_WEIGHT = 1.0

FINAL_FIELDNAMES = ["word", "category", "source", "notes", "human_rating", "llm_rating", "llm_notes"]


def load_shortlist() -> list[dict[str, str]]:
    with open(SHORTLIST_CSV) as f:
        return list(csv.DictReader(f))


def weighted_score(row: dict[str, str], human_weight: float, llm_weight: float) -> float:
    h = float(row["human_rating"]) if row.get("human_rating", "").strip() else 0.0
    l = float(row["llm_rating"]) if row.get("llm_rating", "").strip() else 0.0
    return (human_weight * h + llm_weight * l) / (human_weight + llm_weight)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cull near-duplicate keywords from shortlist")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Cosine similarity above which a pair is considered redundant (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--max-remove", type=int, default=DEFAULT_MAX_REMOVE,
                        help=f"Maximum words to remove (default: {DEFAULT_MAX_REMOVE})")
    parser.add_argument("--human-weight", type=float, default=DEFAULT_HUMAN_WEIGHT,
                        help=f"Weight for human rating when choosing which of a pair to drop (default: {DEFAULT_HUMAN_WEIGHT})")
    parser.add_argument("--llm-weight", type=float, default=DEFAULT_LLM_WEIGHT,
                        help=f"Weight for LLM rating (default: {DEFAULT_LLM_WEIGHT})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print results without writing candidates_final.csv")
    args = parser.parse_args()

    rows = load_shortlist()
    print(f"Loaded {len(rows)} words from shortlist\n")

    if not EMBEDDINGS_NPZ.exists():
        print(f"Error: embeddings not found at {EMBEDDINGS_NPZ}", file=sys.stderr)
        sys.exit(1)

    print("Loading embeddings...")
    data = np.load(EMBEDDINGS_NPZ)
    all_embeddings: np.ndarray = data["embeddings"]
    with open(WORDS_JSON) as f:
        all_words: list[str] = json.load(f)
    n_embeddings = len(all_embeddings)
    word_to_idx = {w.lower(): i for i, w in enumerate(all_words[:n_embeddings])}

    # Partition rows into those with and without embeddings
    has_emb: list[dict[str, str]] = []
    no_emb: list[dict[str, str]] = []
    emb_list: list[np.ndarray] = []

    for row in rows:
        idx = word_to_idx.get(row["word"].lower())
        if idx is not None:
            has_emb.append(row)
            emb_list.append(all_embeddings[idx])
        else:
            no_emb.append(row)

    if no_emb:
        print(f"Warning: {len(no_emb)} words not in embedding vocab, kept unconditionally: "
              f"{', '.join(r['word'] for r in no_emb)}\n")

    emb_matrix = np.stack(emb_list)  # (N, D)
    scores = np.array([weighted_score(r, args.human_weight, args.llm_weight) for r in has_emb])
    forced = np.array([r.get("human_rating", "").strip() == "5" for r in has_emb])

    # Active mask — False means this word has been culled
    active = np.ones(len(has_emb), dtype=bool)
    removed: list[tuple[str, str, float]] = []  # (removed_word, kept_word, similarity)

    for _ in range(args.max_remove):
        active_idx = np.where(active)[0]

        # Compute pairwise similarities among active words only
        active_emb = emb_matrix[active_idx]
        sim = active_emb @ active_emb.T
        np.fill_diagonal(sim, -1.0)

        # Mask out pairs where both words are force-includes — neither can be dropped
        active_forced = forced[active_idx]
        both_forced = np.outer(active_forced, active_forced)
        sim[both_forced] = -1.0

        max_sim = sim.max()
        if max_sim < args.threshold:
            break

        # Find the highest-similarity pair
        local_i, local_j = np.unravel_index(sim.argmax(), sim.shape)
        gi, gj = active_idx[local_i], active_idx[local_j]

        # Always keep a force-include; otherwise remove the lower-scored word,
        # tie-breaking by keeping the earlier shortlist position
        if forced[gi]:
            drop, keep = gj, gi
        elif forced[gj]:
            drop, keep = gi, gj
        elif scores[gi] >= scores[gj]:
            drop, keep = gj, gi
        else:
            drop, keep = gi, gj

        active[drop] = False
        removed.append((has_emb[drop]["word"], has_emb[keep]["word"], float(max_sim)))
        print(f"  Removed '{has_emb[drop]['word']}' (score={scores[drop]:.2f})  "
              f"— too similar to '{has_emb[keep]['word']}' (score={scores[keep]:.2f})  "
              f"sim={max_sim:.4f}")

    kept_from_emb = [has_emb[i] for i in range(len(has_emb)) if active[i]]
    final = kept_from_emb + no_emb

    print(f"\nRemoved {len(removed)} words. Final set: {len(final)} words "
          f"(threshold={args.threshold}, max_remove={args.max_remove})\n")

    if not removed:
        print("No pairs exceeded the similarity threshold — nothing was culled.")

    # Sort by category then score for output
    final.sort(key=lambda r: (
        r.get("category", ""),
        -weighted_score(r, args.human_weight, args.llm_weight),
    ))

    print(f"{'Category':<20} {'Count':>6}  Words")
    print("-" * 80)
    from collections import defaultdict
    by_cat: dict[str, list[str]] = defaultdict(list)
    for r in final:
        by_cat[r.get("category", "")].append(r["word"])
    for cat in sorted(by_cat):
        words = by_cat[cat]
        print(f"  {cat:<18} {len(words):>6}  {', '.join(words)}")

    if not args.dry_run:
        with open(FINAL_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FINAL_FIELDNAMES)
            writer.writeheader()
            writer.writerows(final)
        print(f"\nWrote {len(final)} words to {FINAL_CSV}")
    else:
        print(f"\n(Dry run — nothing written)")


if __name__ == "__main__":
    main()
